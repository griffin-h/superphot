#!/usr/bin/env python

import os
import argparse
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from .util import read_light_curve, select_event_data, filter_colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import gaussian_kde
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


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


def setup_model(obs, max_flux=None):
    """
    Run a Metropolis Hastings MCMC for a file in a single filter with a certain number iterations, burn in (tuning),
    and walkers. The period and multiplier are 180 and 20, respectively.

    Parameters
    ----------
    obs : astropy.table.Table
        Astropy table containing the light curve data.

    Returns
    -------
    model : pymc3.Model
        PyMC3 model object for the input data. Use this to run the MCMC.
    """
    obs_time = obs['PHASE'].filled().data
    obs_flux = obs['FLUXCAL'].filled().data
    obs_unc = obs['FLUXCALERR'].filled().data
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
        t_0 = pm.Uniform(name='Start Time', lower=-50., upper=180.)
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
    """
    x_priors = []
    y_priors = []
    fig, axarr = plt.subplots(len(parameters) // 3 + bool(len(parameters) % 3), 3)
    for param, ax in zip(parameters, axarr.flatten()):
        trace_values = [trace[param.name] for trace in traces]
        combined_trace = np.concatenate(trace_values)
        x = np.linspace(combined_trace.min(), combined_trace.max(), res)
        y_orig = np.exp(param.distribution.logp(x).eval())
        ax.plot(x, y_orig, color='gray', lw=1, ls='--')
        for flt, trace_flt in zip('griz', trace_values):
            y_filt = gaussian_kde(trace_flt)(x)
            ax.plot(x, y_filt, color=filter_colors[flt], lw=1, ls=':')
        y_comb = gaussian_kde(combined_trace)(x)
        ax.plot(x, y_comb, color='gray')
        ax.set_xlabel(param.name)
        x_priors.append(x)
        y_priors.append(y_comb)
    fig.tight_layout()
    return x_priors, y_priors, fig


def setup_new_model(obs, parameters, x_priors, y_priors):
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
    obs_time = obs['PHASE'].filled().data
    obs_flux = obs['FLUXCAL'].filled().data
    obs_unc = obs['FLUXCALERR'].filled().data

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


def run_mcmc(model, iterations, tuning, walkers):
    """
    Run a Metropolis Hastings MCMC for the given model with a certain number iterations, burn in (tuning), and walkers.

    Parameters
    ----------
    model : pymc3.Model
        PyMC3 model object for the input data from `setup_model`.
    iterations : int
        The number of iterations after tuning.
    tuning : int
        The number of iterations used for tuning.
    walkers : int
        The number of cores and walkers used.

    Returns
    -------
    trace : MultiTrace
        The trace has a shape (len(varnames), walkers, iterations) and contains every iteration for each walker for all
        parameters.
    """
    with model:
        trace = pm.sample(iterations, tune=tuning, cores=walkers, chains=walkers, step=pm.Metropolis())
    return trace


def produce_lc(time, trace, align_to_t0=False):
    """
    Load the stored PyMC3 traces and produce model light curves from the parameters.

    Parameters
    ----------
    time : numpy.array
        Range of times (in days, with respect to PEAKMJD) over which the model should be calculated.
    trace : numpy.array
        PyMC3 trace stored as 3-D array with shape (nfilters, nsteps, nparams).
    align_to_t0 : bool, optional
        Interpret `time` as days with respect to t_0 instead of PEAKMJD.

    Returns
    -------
    lc : numpy.array
        Model light curves. Shape = (len(trace) * nwalkers, nfilters, len(time)).
    """
    tt.config.compute_test_value = 'ignore'
    mytensor = tt.TensorType('float64', (False, False, True))
    parameters = [mytensor() for _ in range(6)]
    flux = flux_model(time, *parameters)
    param_values = {param: values[:, :, np.newaxis] for param, values in zip(parameters, np.moveaxis(trace, 2, 0))}
    if align_to_t0:
        param_values[parameters[3]] = np.zeros_like(param_values[parameters[3]])
    lc = flux.eval(param_values)
    return lc


def sample_posterior(trace, rand_num):
    """
    Randomly sample the parameters from the stored MCMC traces.

    Parameters
    ----------
    trace : numpy.ndarray, shape=(nfilters, nsteps, nparams)
        PyMC3 trace stored as 3-D array with shape .
    rand_num : int
        The number of light curves randomly extracted from the MCMC run.

    Returns
    -------
    trace_rand : numpy.ndarray, shape=(nfilters, rand_num, nparams)
        3-D array containing a random sampling of parameters for each filter.

    """
    i_rand = np.random.randint(trace.shape[1], size=rand_num)
    trace_rand = trace[:, i_rand]
    return trace_rand


def plot_model_lcs(obs, trace, parameters, size=100, ax=None, fltr=None, ls=None, phase_min=-50., phase_max=180.):
    x = np.arange(phase_min, phase_max)
    trace_values = np.transpose([trace.get_values(var) for var in parameters])[np.newaxis, :, :]
    params = sample_posterior(trace_values, size)
    y = produce_lc(x, params)[0].T
    if ax is None:
        ax = plt.axes()
    color = filter_colors.get(fltr)
    ax.plot(x, y, 'k', alpha=0.1, color=color, ls=ls)
    ax.errorbar(obs['PHASE'], obs['FLUXCAL'], obs['FLUXCALERR'], fmt='o', color=color)
    ax.set_xlabel('Phase')
    ax.set_ylabel('Flux')
    if fltr:
        ax.text(0.95, 0.95, fltr, transform=ax.transAxes, ha='right', va='top')

    # autoscale y-axis without errorbars
    if len(obs):
        ymin = min(obs['FLUXCAL'].min(), y.min())
        ymax = max(obs['FLUXCAL'].max(), y.max())
        height = ymax - ymin
        ax.set_ylim(ymin - 0.08 * height, ymax + 0.08 * height)


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
    f1 = pm.traceplot(trace, textsize=6, figsize=(6., 7.)).flat[0].get_figure()
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
    model, parameters = setup_model(obs, t['FLUXCAL'].max())
    trace = pm.load_trace(args.filename, model)
    diagnostics(obs, trace, parameters, args.filename, args.show)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', type=str, help='Input SNANA files')
    parser.add_argument('--filters', type=str, default='griz', help='Filters to fit (choose from griz)')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of steps after burn-in')
    parser.add_argument('--tuning', type=int, default=25000, help='Number of burn-in steps')
    parser.add_argument('--walkers', type=int, default=25, help='Number of walkers')
    parser.add_argument('--output-dir', type=str, default='.', help='Path in which to save the PyMC3 trace data')
    parser.add_argument('--ignore-redshift', action='store_false', dest='require_redshift',
                        help='Fit the transient even though its redshift is not measured')
    parser.add_argument('-f', '--force', action='store_true', help='redo the fit even if the trace is already saved')
    parser.add_argument('-2', '--force-second', action='store_true',
                        help='redo only the second iteration of fitting even if the trace is already saved')
    args = parser.parse_args()

    pdf = PdfPages('lc_fits.pdf')
    for filename in args.filenames:
        basename = os.path.basename(filename).split('.')[0]
        outfile = os.path.join(args.output_dir, basename + '_{}')
        t_full = read_light_curve(filename)
        t = select_event_data(t_full)
        if t.meta['REDSHIFT'] <= 0. and args.require_redshift:
            raise ValueError('Skipping file with no redshift ' + filename)
        max_flux = t['FLUXCAL'].max()
        fig, axes = plt.subplots(2, 2, sharex=True)
        fig.text(0.5, 0.95, f'{basename} ($z={t.meta["REDSHIFT"]:.3f}$)', ha='center', va='bottom', size='large')
        traces = []
        for fltr, ax in zip(args.filters, axes.flatten()):
            obs = t[t['FLT'] == fltr]
            if not len(obs):
                logging.warning(f'No {fltr}-band points. Skipping fit.')
                continue
            model, parameters = setup_model(obs, max_flux)
            outfile1 = outfile.format('1' + fltr)
            if not os.path.exists(outfile1) or args.force:
                logging.info(f'Starting {fltr}-band fit, first iteration')
                trace = run_mcmc(model, args.iterations, args.tuning, args.walkers)
                pm.save_trace(trace, outfile1, overwrite=True)
                diagnostics(obs, trace, parameters, outfile1)
            else:
                trace = pm.load_trace(outfile1, model)
                logging.info(f'Loaded {fltr}-band trace from {outfile1}')
            traces.append(trace)
            plot_model_lcs(obs, trace, parameters, size=10, ax=ax, fltr=fltr, ls=':')

        x_priors, y_priors, prior_fig = make_new_priors(traces, parameters)
        prior_fig.savefig(os.path.join(args.output_dir, basename + '_priors.pdf'))
        plt.close(prior_fig)
        logging.info('Starting second iteration of fitting')

        for fltr, ax in zip(args.filters, axes.flatten()):
            obs = t[t['FLT'] == fltr]
            if not len(obs):
                logging.warning(f'No {fltr}-band points. Skipping fit.')
                continue
            new_model, new_params = setup_new_model(obs, parameters, x_priors, y_priors)
            outfile2 = outfile.format('2' + fltr)
            if not os.path.exists(outfile2) or args.force or args.force_second:
                logging.info(f'Starting {fltr}-band fit, second iteration')
                trace = run_mcmc(new_model, args.iterations, args.tuning, args.walkers)
                pm.save_trace(trace, outfile2, overwrite=True)
                diagnostics(obs, trace, new_params, outfile2)
            else:
                logging.info(f'Loaded {fltr}-band trace from {outfile2}')
                trace = pm.load_trace(outfile2, new_model)
            plot_model_lcs(obs, trace, new_params, size=10, ax=ax, fltr=fltr)

        fig.tight_layout(w_pad=0, h_pad=0, rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(args.output_dir, basename + '_final.pdf'))
        pdf.savefig(fig)
        plt.close(fig)
    pdf.close()
