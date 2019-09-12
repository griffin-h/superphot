#!/usr/bin/env python

# FREDERICK DAUPHIN
# DEPARTMENT OF PHYSICS, CARNEGIE MELLON UNIVERSITY
# ADVISOR: DR. GRIFFIN HOSSEINZADEH
# MENTOR: PROF. EDO BERGER
# CENTER FOR ASTROPHYSICS | HARVARD & SMITHSONIAN
# REU 2019 INTERN PROGRAM

import os
import argparse
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from util import light_curve_event_data
import matplotlib.pyplot as plt


def flux_model(t, A, delta, gamma, t_0, tau_rise, tau_fall):
    """
    Calculate the flux given amplitude, plateau decrease, plateau duration, start time, rise time, and fall time using
    theano.switch. Parameters.type = TensorType(float64, scalar).

    Parameters
    ----------
    t : 1-D numpy array
        Time.
    A : TensorVariable
        Amplitude of the light curve.
    delta : TensorVariable
        Fractional decrease in the light curve flux during the plateau.
    gamma : TensorVariable
        The duration of the plateau after the light curve peaks.
    t_0 : TransformedRV
        Start time, which is very close to when the actaul light curve peak flux occurs.
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
    flux_model = A / (1. + tt.exp(-phase / tau_rise)) * tt.switch(phase < gamma,
                                                                  (1 - delta * phase / gamma),
                                                                  (1 - delta) * tt.exp(-(phase - gamma) / tau_fall))
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
        x = np.linspace(-3, 3, 500)
        ls = [0., -2]
        us = [2., 1]
        for l, u in zip(ls, us):
            y = np.zeros(500)
            y[(x<u) & (x>l)] = 1. / ((np.log(u) - np.log(l)) * x)
            plt.plot(x, y, label='lower = {}, upper = {}'.format(l, u))
        plt.xlabel('x', fontsize=12)
        plt.ylabel('f(x)', fontsize=12)
        plt.ylim(0, 1)
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


def setup_model(obs):
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

    with pm.Model() as model:
        A = LogUniform(name='Amplitude', lower=1., upper=100. * obs_flux.max())
        delta = pm.Uniform(name='Plateau Decrease', lower=0., upper=1.)
        BoundedNormal = pm.Bound(pm.Normal, lower=0.)
        gamma = pm.Mixture(name='Plateau Duration', w=tt.constant([2., 1.]) / 3., testval=1.,
                           comp_dists=[BoundedNormal.dist(mu=5., sigma=5.), BoundedNormal.dist(mu=60., sigma=30.)])
        t_0 = pm.Uniform(name='Start Time', lower=-50., upper=300.)
        tau_rise = LogUniform(name='Rise Time', lower=0.01, upper=50.)
        tau_fall = LogUniform(name='Fall Time', lower=1., upper=300.)
        extra_sigma = pm.HalfNormal(name='Intrinsic Scatter', sigma=1.)
        parameters = [A, delta, gamma, t_0, tau_rise, tau_fall]
        varnames = [p.name for p in parameters]

        exp_flux = flux_model(obs_time, *parameters)
        sigma = tt.sqrt(tt.pow(extra_sigma, 2.) + tt.pow(obs_unc, 2.))
        pm.Normal(name='Flux_Posterior', mu=exp_flux, sigma=sigma, observed=obs_flux)

    return model, varnames


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


def diagnostics(obs, trace, varnames, filename='.', show=False):
    """Make some diagnostic plots for the PyMC3 fitting.

    Parameters
    ----------
    obs : astropy.table.Table
        Observed light curve data in a single filter.
    trace : pymc3.MultiTrace
        Trace object that is the result of the PyMC3 fit.
    varnames : list
        List of variable names in the PyMC3 model.
    filename : str, optional
        Directory in which to save the output plots and summary. Not used if `show=True`.
    show : bool, optional
        If True, show the plots instead of saving them.
    """
    pm.traceplot(trace, textsize=6, figsize=(6., 7.))
    pm.pairplot(trace, textsize=6, figsize=(6., 6.))
    pm.plot_posterior(trace, textsize=6, figsize=(6., 4.))
    summary = pm.summary(trace)

    x = np.arange(obs['PHASE'].min(), obs['PHASE'].max())
    plt.figure()
    for i in np.random.randint(0, len(trace), size=100):
        params = [trace[j][i] for j in varnames]
        y = flux_model(x, *params)
        plt.plot(x, y.eval(), 'k', alpha=0.1)
    plt.errorbar(obs['PHASE'], obs['FLUXCAL'], obs['FLUXCALERR'], fmt='o')
    plt.xlabel('Phase')
    plt.ylabel('Flux')

    if show:
        print(summary)
        plt.show()
    else:
        with open(os.path.join(filename, 'summary.txt'), 'w') as f:
            f.write(summary.to_string() + '\n')
        figure_filename = os.path.join(filename, 'Figure_{:d}.pdf')
        for i in range(4):
            fig = plt.figure(i + 1)
            fig.savefig(figure_filename.format(i + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', type=str, help='Input SNANA files')
    parser.add_argument('--filters', type=str, default='griz', help='Filters to fit (choose from griz)')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of steps after burn-in')
    parser.add_argument('--tuning', type=int, default=25000, help='Number of burn-in steps')
    parser.add_argument('--walkers', type=int, default=25, help='Number of walkers')
    parser.add_argument('--output-dir', type=str, default='.', help='Path in which to save the PyMC3 trace data')
    parser.add_argument('--ignore-redshift', action='store_false', dest='require_redshift',
                        help='Fit the transient even though its redshift is not measured')
    args = parser.parse_args()

    for filename in args.filenames:
        outfile = os.path.join(args.output_dir, os.path.basename(filename).replace('.snana.dat', '_{}'))
        t = light_curve_event_data(filename)
        if t.meta['REDSHIFT'] < 0. and args.require_redshift:
            raise ValueError('Skipping file with no redshift ' + filename)
        for fltr in args.filters:
            obs = t[t['FLT'] == fltr]
            model, varnames = setup_model(obs)
            trace = run_mcmc(model, args.iterations, args.tuning, args.walkers)
            pm.save_trace(trace, outfile.format(fltr), overwrite=True)
            diagnostics(obs, trace, varnames, outfile.format(fltr))
