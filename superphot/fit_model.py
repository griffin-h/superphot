#!/usr/bin/env python

import os
import argparse
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from .util import read_light_curve, select_event_data, filter_colors, PHASE_MIN, PHASE_MAX
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
# Default fitting parameters
FILTERS = 'griz'
ITERATIONS = 10000
TUNING = 25000
WALKERS = 25


def flux_model(t, A, beta, gamma, t_0, tau_rise, tau_fall):
    """
    Calculate the flux given amplitude, plateau slope, plateau duration, start time, rise time, and fall time using
    theano.switch. Parameters.type = TensorType(float64, scalar).

    Parameters
    ----------
    t : 1-D numpy array
        Time.
    A : TensorVariable
        Amplitude of the light curve.
    beta : TensorVariable
        Light curve slope during the plateau, normalized by the amplitude.
    gamma : TensorVariable
        The duration of the plateau after the light curve peaks.
    t_0 : TransformedRV
        Start time, which is very close to when the actual light curve peak flux occurs.
    tau_rise : TensorVariable
        Exponential rise time to peak.
    tau_fall : TensorVariable
        Exponential decay time after the plateau ends.

    Returns
    -------
    flux_model : symbolic Tensor
        The predicted flux from the given model.

    """
    phase = t - t_0
    flux_model = A / (1. + tt.exp(-phase / tau_rise)) * \
        tt.switch(phase < gamma, 1. - beta * phase, (1. - beta * gamma) * tt.exp((gamma - phase) / tau_fall))
    return flux_model


class LogUniform(pm.distributions.continuous.BoundedContinuous):
    R"""
    Continuous log-uniform log-likelihood.

    The pdf of this distribution is

    .. math::

       f(x \mid lower, upper) = \frac{1}{[\log(upper)-\log(lower)]x}

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        plt.style.use('seaborn-darkgrid')
        x = np.linspace(1., 300., 500)
        ls = [3., 150.]
        us = [100., 250.]
        for l, u in zip(ls, us):
            y = np.zeros(500)
            inside = (x<u) & (x>l)
            y[inside] = 1. / ((np.log(u) - np.log(l)) * x[inside])
            plt.plot(x, y, label='lower = {}, upper = {}'.format(l, u))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.legend(loc=1)
        plt.show()

    ========  =====================================
    Support   :math:`x \in [lower, upper]`
    Mean      :math:`\dfrac{upper - lower}{\log(upper) - \log(lower)}`
    ========  =====================================

    Parameters
    ----------
    lower : float
        Lower limit.
    upper : float
        Upper limit.
    """

    def __init__(self, lower=1., upper=np.e, *args, **kwargs):
        if lower <= 0. or upper <= 0.:
            raise ValueError('LogUniform bounds must be positive')
        log_lower = tt.log(lower)
        log_upper = tt.log(upper)
        self.logdist = pm.Uniform.dist(lower=log_lower, upper=log_upper)
        self.median = tt.exp(self.logdist.median)
        self.mean = (upper - lower) / (log_upper - log_lower)

        super().__init__(lower=lower, upper=upper, *args, **kwargs)

    def random(self, point=None, size=None):
        """
        Draw random values from LogUniform distribution.

        Parameters
        ----------
        point : dict, optional
            Dict of variable values on which random values are to be
            conditioned (uses default point if not specified).
        size : int, optional
            Desired size of random sample (returns one sample if not
            specified).

        Returns
        -------
        array
        """
        return tt.exp(self.logdist.random(point=point, size=size))

    def logp(self, value):
        """
        Calculate log-probability of LogUniform distribution at specified value.

        Parameters
        ----------
        value : numeric
            Value for which log-probability is calculated.

        Returns
        -------
        TensorVariable
        """
        log_value = tt.log(value)
        return self.logdist.logp(log_value) - log_value


def setup_model1(obs, max_flux=None):
    """
    Set up the PyMC3 model object, which contains the priors and the likelihood.

    Parameters
    ----------
    obs : astropy.table.Table
        Astropy table containing the light curve data.
    max_flux : float, optional
        The maximum flux observed in any filter. The amplitude prior is 100 * `max_flux`. If None, the maximum flux in
        the input table is used, even though it does not contain all the filters.

    Returns
    -------
    model : pymc3.Model
        PyMC3 model object for the input data. Use this to run the MCMC.
    """
    obs_time = obs['PHASE'].data
    obs_flux = obs['FLUXCAL'].data
    obs_unc = obs['FLUXCALERR'].data
    if max_flux is None:
        max_flux = obs_flux.max()
    if max_flux <= 0.01:
        raise ValueError('The maximum flux is very low. Cannot fit the model.')

    with pm.Model() as model:
        A = LogUniform(name='Amplitude', lower=1., upper=100. * max_flux)
        beta = pm.Uniform(name='Plateau Slope', lower=0., upper=0.01)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        gamma = pm.Mixture(name='Plateau Duration', w=tt.constant([2., 1.]) / 3., testval=1.,
                           comp_dists=[BoundedNormal.dist(mu=5., sigma=5.), BoundedNormal.dist(mu=60., sigma=30.)])
        t_0 = pm.Uniform(name='Start Time', lower=PHASE_MIN, upper=PHASE_MAX)
        tau_rise = pm.Uniform(name='Rise Time', lower=0.01, upper=50.)
        tau_fall = pm.Uniform(name='Fall Time', lower=1., upper=300.)
        extra_sigma = pm.HalfNormal(name='Intrinsic Scatter', sigma=1.)
        parameters = [A, beta, gamma, t_0, tau_rise, tau_fall]

        exp_flux = flux_model(obs_time, *parameters)
        sigma = tt.sqrt(tt.pow(extra_sigma, 2.) + tt.pow(obs_unc, 2.))
        pm.Normal(name='Flux_Posterior', mu=exp_flux, sigma=sigma, observed=obs_flux)

    return model, parameters


def make_new_priors(traces, parameters, res=100):
    """
    For each parameter, combine the posteriors for the four filters and use that as the new prior.

    Parameters
    ----------
    traces : list
        List of MultiTrace objects for each filter.
    parameters : list
        List of Theano variables for which to combine the posteriors. (Only names of the parameters are used.)
    res : int, optional
        Number of points to sample the KDE for the new priors.

    Returns
    -------
    x_priors : list
        List of Numpy arrays containing the x-values of the new priors.
    y_priors : list
        List of Numpy arrays containing the y-values of the new priors.
    old_posteriors : list
        List of Numpy arrays containing the y-values of the old posteriors.
    """
    x_priors = []
    y_priors = []
    old_posteriors = []
    for param in parameters:
        trace_values = [trace[param.name] for trace in traces]
        combined_trace = np.concatenate(trace_values)
        x = np.linspace(combined_trace.min(), combined_trace.max(), res)
        y_comb = gaussian_kde(combined_trace)(x)
        x_priors.append(x)
        y_priors.append(y_comb)
        old_posteriors.append(trace_values)
    return x_priors, y_priors, old_posteriors


def plot_priors(x_priors, y_priors, old_posteriors, parameters, saveto=None):
    """
    Overplot the old priors, the old posteriors in each filter, and the new priors for each parameter.

    Parameters
    ----------
    x_priors : list
        List of Numpy arrays containing the x-values of the new priors.
    y_priors : list
        List of Numpy arrays containing the y-values of the new priors.
    old_posteriors : list
        List of Numpy arrays containing the y-values of the old posteriors.
    parameters : list
        List of Theano variables for which to combine the posteriors.
    saveto : str, optional
        Filename to which to save the plot. If None, display the plot instead of saving it.
    """
    fig, axarr = plt.subplots(len(parameters) // 3 + bool(len(parameters) % 3), 3)
    for param, x, y_comb, trace_values, ax in zip(parameters, x_priors, y_priors, old_posteriors, axarr.flatten()):
        y_orig = np.exp(param.distribution.logp(x).eval())
        ax.plot(x, y_orig, color='gray', lw=1, ls='--')
        for flt, trace_flt in zip('griz', trace_values):
            y_filt = gaussian_kde(trace_flt)(x)
            ax.plot(x, y_filt, color=filter_colors[flt], lw=1, ls=':')
        ax.plot(x, y_comb, color='gray')
        ax.set_xlabel(param.name)
        ax.set_yticks([])
    fig.tight_layout()
    if saveto:
        fig.savefig(saveto)
    else:
        plt.show()
    plt.close(fig)


def setup_model2(obs, parameters, x_priors, y_priors):
    """
    Set up a PyMC3 model for observations in a given filter using the given priors and parameter names.

    Parameters
    ----------
    obs : astropy.table.Table
        Astropy table containing the light curve data.
    parameters : list
        List of Theano variables for which to create new parameters. (Only names of the parameters are used.)
    x_priors : list
        List of Numpy arrays containing the x-values of the priors.
    y_priors : list
        List of Numpy arrays containing the y-values of the priors.

    Returns
    -------
    model : pymc3.Model
        PyMC3 model object for the input data. Use this to run the MCMC.
    """
    obs_time = obs['PHASE'].data
    obs_flux = obs['FLUXCAL'].data
    obs_unc = obs['FLUXCALERR'].data

    with pm.Model() as model:
        new_params = []
        for param, x, y in zip(parameters, x_priors, y_priors):
            new_param = pm.Interpolated(name=param.name, x_points=x, pdf_points=y)
            new_params.append(new_param)

        exp_flux = flux_model(obs_time, *new_params)
        extra_sigma = pm.HalfNormal(name='Intrinsic Scatter', sigma=1.)
        sigma = tt.sqrt(tt.pow(extra_sigma, 2.) + tt.pow(obs_unc, 2.))
        pm.Normal(name='Flux_Posterior', mu=exp_flux, sigma=sigma, observed=obs_flux)

    return model, new_params


def sample_or_load_trace(model, trace_file, force=False, iterations=ITERATIONS, walkers=WALKERS, tuning=TUNING):
    """
    Run a Metropolis Hastings MCMC for the given model with a certain number iterations, burn in (tuning), and walkers.

    If the MCMC has already been run, read and return the existing trace (unless `force=True`).

    Parameters
    ----------
    model : pymc3.Model
        PyMC3 model object for the input data.
    trace_file : str
        Path where the trace will be stored. If this path exists, load the trace from there instead.
    force : bool, optional
        Resample the model even if `trace_file` already exists.
    iterations : int, optional
        The number of iterations after tuning.
    walkers : int, optional
        The number of cores and walkers used.
    tuning : int, optional
        The number of iterations used for tuning.

    Returns
    -------
    trace : pymc3.MultiTrace
        The PyMC3 trace object for the MCMC run.
    """
    basename = os.path.basename(trace_file)
    with model:
        if not os.path.exists(trace_file) or force:
            logging.info(f'Starting fit for {basename}')
            trace = pm.sample(iterations, tune=tuning, cores=walkers, chains=walkers, step=pm.Metropolis())
            pm.save_trace(trace, trace_file, overwrite=True)
        else:
            trace = pm.load_trace(trace_file)
            logging.info(f'Loaded trace from {trace_file}')
    return trace


def produce_lc(time, trace, align_to_t0=False):
    """
    Load the stored PyMC3 traces and produce model light curves from the parameters.

    Parameters
    ----------
    time : numpy.array
        Range of times (in days, with respect to PEAKMJD) over which the model should be calculated.
    trace : numpy.array
        PyMC3 trace stored as an array, with parameters as the last dimension.
    align_to_t0 : bool, optional
        Interpret `time` as days with respect to t_0 instead of PEAKMJD.

    Returns
    -------
    lc : numpy.array
        Model light curves. Time is the last dimension.
    """
    tt.config.compute_test_value = 'ignore'
    mytensor = tt.TensorType('float64', (False,) * (trace.ndim - 1) + (True,))
    parameters = [mytensor() for _ in range(trace.shape[-1])]
    flux = flux_model(time, *parameters)
    param_values = {param: values[..., np.newaxis] for param, values in zip(parameters, np.moveaxis(trace, -1, 0))}
    if align_to_t0:
        param_values[parameters[3]] = np.zeros_like(param_values[parameters[3]])
    lc = flux.eval(param_values)
    return lc


def sample_posterior(trace, rand_num):
    """
    Randomly sample the parameters from the stored MCMC traces.

    Parameters
    ----------
    trace : numpy.ndarray, shape=(nsteps, nfilters, nparams)
        PyMC3 trace stored as 3-D array with shape .
    rand_num : int
        The number of light curves randomly extracted from the MCMC run.

    Returns
    -------
    trace_rand : numpy.ndarray, shape=(rand_num, nfilters, nparams)
        3-D array containing a random sampling of parameters for each filter.

    """
    i_rand = np.random.randint(trace.shape[0], size=rand_num)
    trace_rand = trace[i_rand]
    return trace_rand


def plot_model_lcs(obs, trace, parameters, size=100, ax=None, fltr=None, ls=None, phase_min=PHASE_MIN,
                   phase_max=PHASE_MAX):
    """
    Plot sample light curves from a fit compared to the observations.

    Parameters
    ----------
    obs : astropy.table.Table
        Astropy table containing the observed light curve in a single filter.
    trace : pymc3.MultiTrace
        PyMC3 trace object containing values for the fit parameters.
    parameters : list
        List of Theano variables in the PyMC3 model.
    size : int, optional
        Number of draws from the posterior to plot. Default: 100.
    ax : matplotlib.axes.Axes, optional
        Axes object on which to plot the light curves. If None, create new Axes.
    fltr : str, optional
        Filter these data were observed in. Only used to label and color the plot.
    ls : str, optional
        Line style for the model light curves. Default: solid line.
    phase_min, phase_max : float, optional
        Time range over which to plot the light curves.
    """
    x = np.arange(phase_min, phase_max)
    trace_values = np.transpose([trace.get_values(var) for var in parameters])
    params = sample_posterior(trace_values, size)
    y = produce_lc(x, params).T
    if ax is None:
        ax = plt.axes()
    color = filter_colors.get(fltr)
    ax.plot(x, y, 'k', alpha=0.2, color=color, ls=ls)
    ax.errorbar(obs['PHASE'], obs['FLUXCAL'], obs['FLUXCALERR'], fmt='o', color=color)
    ax.set_xlim(phase_min - 9., phase_max + 9.)
    if fltr:
        ax.text(0.95, 0.95, fltr, transform=ax.transAxes, ha='right', va='top')

    # autoscale y-axis without errorbars
    if len(obs):
        ymin = min(obs['FLUXCAL'].min(), y.min())
        ymax = max(obs['FLUXCAL'].max(), y.max())
        height = ymax - ymin
        ax.set_ylim(ymin - 0.08 * height, ymax + 0.08 * height)


def plot_final_fits(t, traces1, traces2, parameters, outfile=None, filters=FILTERS):
    """
    Make a four-panel plot showing sample light curves from each of the two fitting iterations compared to observations.

    Parameters
    ----------
    t : astropy.table.Table
        Astropy table containing the observed light curve.
    traces1, traces2 : list
        Lists of the trace objects (for each filter) from which to generate the model light curves.
    parameters : list
        List of Theano variables in the PyMC3 model.
    outfile : str, optional
        Filename to which to save the plot. If None, display the plot instead of saving it.
    filters : str, optional
        Filters corresponding to the traces in `trace1` and `trace2`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for the plot (can be added to a multipage PDF).
    """
    fig, axes = plt.subplots(2, 2, sharex=True)
    basename = os.path.splitext(os.path.basename(outfile))[0]
    fig.text(0.5, 0.95, basename, ha='center', va='bottom', size='large')
    for fltr, trace1, trace2, ax in zip(filters, traces1, traces2, axes.flatten()):
        obs = t[t['FLT'] == fltr]
        plot_model_lcs(obs, trace1, parameters, size=10, ax=ax, fltr=fltr, ls=':')
        plot_model_lcs(obs, trace2, parameters, size=10, ax=ax, fltr=fltr)
        ax.set_ylabel('Flux')
    for ax in axes[-1]:
        ax.set_xlabel('Phase')
    for ax in axes[:, -1]:
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
    fig.tight_layout(w_pad=0, h_pad=0, rect=(0, 0, 1, 0.97))
    if outfile:
        fig.savefig(outfile)
    else:
        plt.show()
    return fig


def diagnostics(obs, trace, parameters, filename='.', show=False):
    """Make some diagnostic plots for the PyMC3 fitting.

    Parameters
    ----------
    obs : astropy.table.Table
        Observed light curve data in a single filter.
    trace : pymc3.MultiTrace
        Trace object that is the result of the PyMC3 fit.
    parameters : list
        List of Theano variables in the PyMC3 model.
    filename : str, optional
        Directory in which to save the output plots and summary. Not used if `show=True`.
    show : bool, optional
        If True, show the plots instead of saving them.
    """
    f1 = pm.traceplot(trace, figsize=(6., 7.)).flat[0].get_figure()
    f2 = pm.pairplot(trace, kind='hexbin', textsize=6, figsize=(6., 6.)).flat[0].get_figure()
    f3 = pm.plot_posterior(trace, textsize=6, figsize=(6., 4.)).flat[0].get_figure()
    summary = pm.summary(trace)

    f4 = plt.figure()
    plot_model_lcs(obs, trace, parameters)

    if show:
        print(summary)
        plt.show()
    else:
        with open(os.path.join(filename, 'summary.txt'), 'w') as f:
            f.write(summary.to_string() + '\n')
        f1.savefig(os.path.join(filename, 'trace.pdf'))
        f2.savefig(os.path.join(filename, 'corner.pdf'))
        f3.savefig(os.path.join(filename, 'posterior.pdf'))
        f4.savefig(os.path.join(filename, 'lightcurve.pdf'))
        plt.close('all')


def two_iteration_mcmc(light_curve, outfile, filters=FILTERS, force=False, force_second=False, do_diagnostics=True,
                       iterations=ITERATIONS, walkers=WALKERS, tuning=TUNING):
    """
    Fit the model to the observed light curve. Then combine the posteriors for each filter and use that as the new prior
    for a second iteration of fitting.

    Parameters
    ----------
    light_curve : astropy.table.Table
        Astropy table containing the observed light curve.
    outfile : str
        Path where the trace will be stored. This should include a blank field ({{}}) that will be replaced with the 
        iteration number and filter name. Diagnostic plots will also be saved according to this pattern.
    filters : str, optional
        Light curve filters to fit. If the observed light curve does not contain one or more of these filters, the 
        posteriors of the remaining filters will be combined and used in place of the missing ones.
    force : bool, optional
        Redo the fit (both iterations) even if results are already stored in `outfile`. Default: False.
    force_second : bool, optional
        Redo only the second iteration of the fit, even if the results are already stored in `outfile`. Default: False.
    do_diagnostics : bool, optional
        Produce and save some diagnostic plots. Default: True.
    iterations : int, optional
        The number of iterations after tuning.
    walkers : int, optional
        The number of cores and walkers used.
    tuning : int, optional
        The number of iterations used for tuning.

    Returns
    -------
    traces1, traces2 : list
        Lists of the PyMC3 trace objects for each filter for the first and second fitting iterations, respectively.
    parameters : list
        List of Theano variables in the PyMC3 model.
    """
    t = select_event_data(light_curve)
    traces1 = []
    for fltr in filters:
        obs = t[t['FLT'] == fltr]
        if not len(obs):
            logging.warning(f'No {fltr}-band points. Skipping fit.')
            continue
        model1, parameters1 = setup_model1(obs, t['FLUXCAL'].max())
        outfile1 = outfile.format('_1' + fltr)
        trace1 = sample_or_load_trace(model1, outfile1, force, iterations, walkers, tuning)
        traces1.append(trace1)
        if do_diagnostics:
            diagnostics(obs, trace1, parameters1, outfile1)

    x_priors, y_priors, old_posteriors = make_new_priors(traces1, parameters1)
    if do_diagnostics:
        plot_priors(x_priors, y_priors, old_posteriors, parameters1, outfile.format('_priors.pdf'))
    logging.info('Starting second iteration of fitting')

    traces2 = []
    for fltr in filters:
        obs = t[t['FLT'] == fltr]
        if not len(obs):
            logging.warning(f'No {fltr}-band points. Skipping fit.')
            continue
        model2, parameters2 = setup_model2(obs, parameters1, x_priors, y_priors)
        outfile2 = outfile.format('_2' + fltr)
        trace2 = sample_or_load_trace(model2, outfile2, force or force_second, iterations, walkers, tuning)
        traces2.append(trace2)
        if do_diagnostics:
            diagnostics(obs, trace2, parameters2, outfile2)

    return traces1, traces2, parameters2


def plot_diagnostics():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Path to PyMC3 trace directory')
    parser.add_argument('--snana-path', type=str, default='.', help='Path to SNANA files')
    parser.add_argument('--show', action='store_true', help='Show the plots instead of saving to a file')
    args = parser.parse_args()

    _, _, ps1id, fltr = os.path.split(args.filename.strip('/'))[-1].split('_')
    fltr = fltr.strip('12')
    snana_file = os.path.join(args.snana_path, f'PS1_PS1MD_{ps1id}.snana.dat')
    t_full = read_light_curve(snana_file)
    t = select_event_data(t_full)
    obs = t[t['FLT'] == fltr]
    model, parameters = setup_model1(obs, t['FLUXCAL'].max())
    trace = pm.load_trace(args.filename, model)
    diagnostics(obs, trace, parameters, args.filename, args.show)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', type=str, help='Input SNANA files')
    parser.add_argument('--filters', type=str, default=FILTERS, help='Filters to fit (choose from griz)')
    parser.add_argument('--iterations', type=int, default=ITERATIONS, help='Number of steps after burn-in')
    parser.add_argument('--tuning', type=int, default=TUNING, help='Number of burn-in steps')
    parser.add_argument('--walkers', type=int, default=WALKERS, help='Number of walkers')
    parser.add_argument('--output-dir', type=str, default='.', help='Path in which to save the PyMC3 trace data')
    parser.add_argument('--zmin', type=float, help='Do not fit the transient if redshift <= zmin in the header')
    parser.add_argument('-f', '--force', action='store_true', help='redo the fit even if the trace is already saved')
    parser.add_argument('-2', '--force-second', action='store_true',
                        help='redo only the second iteration of fitting even if the trace is already saved')
    parser.add_argument('--no-plots', action='store_false', dest='plots', help='Save some diagnostic plots')
    args = parser.parse_args()

    pdf = PdfPages('lc_fits.pdf', keep_empty=False)
    for filename in args.filenames:
        basename = os.path.basename(filename).split('.')[0]
        outfile = os.path.join(args.output_dir, basename + '{}')
        light_curve = read_light_curve(filename)
        if args.zmin is not None and light_curve.meta['REDSHIFT'] <= args.zmin:
            raise ValueError(f'Skipping file with redshift {light_curve.meta["REDSHIFT"]}: {filename}')
        traces1, traces2, parameters = two_iteration_mcmc(light_curve, outfile, filters=args.filters, force=args.force,
                                                          force_second=args.force_second, do_diagnostics=args.plots,
                                                          iterations=args.iterations, walkers=args.walkers,
                                                          tuning=args.tuning)
        if args.plots:
            fig = plot_final_fits(light_curve, traces1, traces2, parameters, outfile.format('.pdf'), args.filters)
            pdf.savefig(fig)
            plt.close(fig)
    pdf.close()
