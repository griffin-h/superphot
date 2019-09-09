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
from theano.tensor import switch
from util import light_curve_event_data


def flux_model(t, A, B, gamma, t_0, tau_rise, tau_fall):
    """
    Calculate the flux given amplitude, plateau slope, plateau duration, start time, rise time, and fall time using
    theano.switch. Parameters.type = TensorType(float64, scalar).

    Parameters
    ----------
    t : 1-D numpy array
        Time.
    A : TensorVariable
        Amplitude of the light curve.
    B : TensorVariable
        Slope of the plateau after the light curve peaks.
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
    t_1 = t_0 + gamma
    flux_model = switch((t < t_1),
                        ((A + B * (t - t_0)) /
                         (1. + np.exp((t - t_0) / -tau_rise))),
                        ((A + B * gamma) * np.exp((t - gamma - t_0) / -tau_fall) /
                         (1. + np.exp((t - t_0) / -tau_rise))))
    return flux_model


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
    obs_time = obs['MJD'].filled().data
    obs_flux = obs['FLUXCAL'].filled().data
    obs_unc = obs['FLUXCALERR'].filled().data

    with pm.Model() as model:
        log_A = pm.Uniform(name='Log(Amplitude)', lower=0, upper=6)
        arctan_beta = pm.Uniform(name='Arctan(Plateau Slope)', lower=-1.56, upper=0)
        log_gamma = pm.Uniform(name='Log(Plateau Duration)', lower=-3, upper=3)
        t_0 = pm.Uniform(name='Start Time', lower=-50, upper=50)
        log_tau_rise = pm.Uniform(name='Log(Rise Time)', lower=-3, upper=3)
        log_tau_fall = pm.Uniform(name='Log(Fall Time)', lower=-3, upper=3)
        extra_sigma = pm.HalfNormal(name='sigma', sigma=1)
        parameters = [log_A, arctan_beta, log_gamma, t_0, log_tau_rise, log_tau_fall]
        varnames = [p.name for p in parameters]

        A = 10. ** log_A
        beta = np.tan(arctan_beta)
        gamma = 10. ** log_gamma
        tau_rise = 10. ** log_tau_rise
        tau_fall = 10. ** log_tau_fall

        exp_flux = flux_model(obs_time, A, beta, gamma, t_0, tau_rise, tau_fall)
        sigma = np.sqrt(extra_sigma ** 2. + obs_unc ** 2.)
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
            model, _ = setup_model(obs)
            trace = run_mcmc(model, args.iterations, args.tuning, args.walkers)
            pm.save_trace(trace, outfile.format(fltr), overwrite=True)
