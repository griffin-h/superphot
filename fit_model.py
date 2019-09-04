#!/usr/bin/env python
# coding: utf-8
# file = '/data/reu/fdauphin/PS1_MDS/PS1_Final/PS1_PS1MD_PSc' + file_num + '.snana.dat'

# FREDERICK DAUPHIN
# DEPARTMENT OF PHYSICS, CARNEGIE MELLON UNIVERSITY
# ADVISOR: DR. GRIFFIN HOSSEINZADEH
# MENOTR: PROF. EDO BERGER
# CENTER FOR ASTROPHYSICS HARVARD AND SMITHSONIAN
# REU 2019 INTERN PROGRAM
# LAST MODIFIED: 08/27/19
# CHECK COMPLETED

# IMPORTS
# OS IS USED FOR MANIPULATING PATH NAMES
# SYS USES THE FILE NAME ON THE TERMINAL COMMAND LINE TO RUN THE SCRIPT
# NUMPY CONTAINS MATHEMATICAL FUNCTIONS, INCLUDING MATRIX MANIPULATION FUNCTIONS
# PYMC3 IS A MARKOV CHAIN MONTE CARLO PROCESS LIBRARY THAT WE USED TO EXTRACT PARAMETERS FROM OUR MODEL
# THEANO/.TENSOR IS A COMPONENT OF PYMC3 AND IS NEEDED TO EXPLICITLY CODE OUR MODEL
# MAD_STD IS THE STANDARD DEVIATION ESTIMATED USING MEDIAN ABSOLUTE DEVIATION, WHICH WE USED TO CUT OUTLIER FLUX POINTS
# LST_FILTER IS A LIST OF THE FOUR FILTERS USED FOR OUR LIGHT CURVES
# VARNAME IS A DICTIONARY CONTAINING ALL OF THE VARIABLE NAMES FOR OUR PARAMETERS IN PYMC3

import os
import sys
import argparse
import numpy as np
import pymc3 as pm
from theano.tensor import switch
from util import light_curve_event_data

varname = ['Log(Amplitude)', 'Arctan(Plateau Slope)', 'Log(Plateau Duration)',
           'Start Time', 'Log(Rise Time)', 'Log(Fall Time)']


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


def setup_model(file, fltr):
    """
    Run a Metropolis Hastings MCMC for a file in a single filter with a certain number iterations, burn in (tuning),
    and walkers. The period and multiplier are 180 and 20, respectively.

    Parameters
    ----------
    file : str
        Name of .snana.dat file containing the light curve data.
    fltr: int
        Integer 0-3, corresponding to the filters g, r, i, z.

    Returns
    -------
    model : pymc3.Model
        PyMC3 model object for the input data. Use this to run the MCMC.
    """
    obs = light_curve_event_data(file, fltr)

    with pm.Model() as model:
        obs_time = obs['MJD'].filled().data
        obs_flux = obs['FLUXCAL'].filled().data
        obs_unc = obs['FLUXCALERR'].filled().data

        A = 10 ** pm.Uniform('Log(Amplitude)', lower=0, upper=6)
        B = np.tan(pm.Uniform('Arctan(Plateau Slope)', lower=-1.56, upper=0))
        gamma = 10 ** pm.Uniform('Log(Plateau Duration)', lower=-3, upper=3)
        t_0 = pm.Uniform('Start Time', lower=-50, upper=50)
        tau_rise = 10 ** pm.Uniform('Log(Rise Time)', lower=-3, upper=3)
        tau_fall = 10 ** pm.Uniform('Log(Fall Time)', lower=-3, upper=3)
        sigma = np.sqrt(pm.HalfNormal('sigma', sigma=1) ** 2 + obs_unc ** 2)
        parameters = [A, B, gamma, t_0, tau_rise, tau_fall]

        exp_flux = flux_model(obs_time, *parameters)

        pm.Normal('Flux_Posterior', mu=exp_flux, sigma=sigma, observed=obs_flux)

    return model


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
    parser.add_argument('filename', help='Input SNANA file')
    parser.add_argument('--filters', nargs='+', type=int, default=[0, 1, 2, 3],
                        help='Filters to fit (g=0, r=1, i=2, z=3)')
    parser.add_argument('--iterations', type=int, default=10000, help='Number of steps after burn-in')
    parser.add_argument('--tuning', type=int, default=25000, help='Number of burn-in steps')
    parser.add_argument('--walkers', type=int, default=25, help='Number of walkers')
    args = parser.parse_args()

    outfile = os.path.basename(args.filename).replace('.snana.dat', '_F{:d}')
    for fltr in args.filters:
        model = setup_model(args.filename, fltr)
        trace = run_mcmc(model, args.iterations, args.tuning, args.walkers)
        pm.save_trace(trace, outfile.format(fltr), overwrite=True)
