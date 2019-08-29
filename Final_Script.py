#!/usr/bin/env python
# coding: utf-8
#file = '/data/reu/fdauphin/PS1_MDS/PS1_Final/PS1_PS1MD_PSc' + file_num + '.snana.dat'

#FREDERICK DAUPHIN
#DEPARTMENT OF PHYSICS, CARNEGIE MELLON UNIVERSITY
#ADVISOR: DR. GRIFFIN HOSSEINZADEH
#MENOTR: PROF. EDO BERGER
#CENTER FOR ASTROPHYSICS HARVARD AND SMITHSONIAN
#REU 2019 INTERN PROGRAM
#LAST MODIFIED: 08/27/19
#CHECK COMPLETED

#PURPOSE:

# In[1]:


#IMPORTS
    #OS IS USED FOR MANIPULATING PATH NAMES
    #SYS USES THE FILE NAME ON THE TERMINAL COMMAND LINE TO RUN THE SCRIPT
    #NUMPY CONTAINS MATHEMATICAL FUNCTIONS, INCLUDING MATRIX MANIPULATION FUNCTIONS
    #PYMC3 IS A MARKOV CHAIN MONTE CARLO PROCESS LIBRARY THAT WE USED TO EXTRACT PARAMETERS FROM OUR MODEL
    #THEANO/.TENSOR IS A COMPONENT OF PYMC3 AND IS NEEDED TO EXPLICITLY CODE OUR MODEL
    #MAD_STD IS THE STANDARD DEVIATION ESTIMATED USING MEDIAN ABSOLUTE DEVIATION, WHICH WE USED TO CUT OUTLIER FLUX POINTS
    #LST_FILTER IS A LIST OF THE FOUR FILTERS USED FOR OUR LIGHT CURVES
    #VARNAME IS A DICTIONARY CONTAINING ALL OF THE VARIABLE NAMES FOR OUR PARAMETERS IN PYMC3

import os
import sys
import numpy as np
import pymc3 as pm
import theano
import theano.tensor as T
from astropy.stats import mad_std
lst_filter = ['g', 'r', 'i', 'z']
varname = ['Log(Amplitude)', 'Arctan(Plateau Slope)', 'Log(Plateau Duration)', 
           'Start Time', 'Log(Rise Time)', 'Log(Fall Time)']


# In[2]:


def s2c(file):
    '''
    Make a PS1_PS1MD_PSc file into a 2-D list.
    
    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
        
    Returns
    -------
    lst : list
        2-D list containing the light curve data.
        
    '''
    with open(file) as infile:
        lst = []
        for line in infile:
            lst.append(line.split())
    return lst

def filter_func(lst, fltr):
    '''
    Make a 2-D list containing data only from one filter.
    
    Parameters
    ----------
    lst : list 
        2-D list containing the light curve data.
    fltr: string
        One of the elements from lst_filter (g, r, i, z).
        
    Returns
    -------
    lst_new : list
        2-D list containing data from only one filter.
        
    '''
    lst_fltr = []
    for row in lst:
        if row[2] == fltr:
            lst_fltr.append(row)
    return lst_fltr

def peak_event(lst, period, peak_time):
    '''
    Make a 2-D list containing only data from the observational period with the peak flux.
    
    Parameters
    ----------
    lst : list 
        2-D list containing the light curve data.
    period: float
        Half the length of time from which you want to observe.
    peak_time: float
        The estimated light curve peak time (which is also the discovery time).
        
    Returns
    -------
    lst_peak : list
        2-D list containing data from only the observational period with the peak flux.
        
    '''
    lst_peak = []
    for row in lst:
        time_ij = float(row[1])
        if time_ij < peak_time + period and time_ij > peak_time - period:
            lst_peak.append(row)
    return lst_peak

def cut_outliers(lst, multiplier):
    '''
    Make a 2-D list containing only data that is below the cut off threshold.
    
    Parameters
    ----------
    lst : list 
        2-D list containing the light curve data.
    multiplier: float
        Determines at what value (multiplier * mad_std) to cut outlier data points.
        
    Returns
    -------
    lst_cut : list
        2-D list containing only data that is below the cut off threshold.
        
    '''
    lst_cut = []
    table = np.transpose(lst)
    if len(table) > 0:
        flux = table[4].astype(float)
        madstd = mad_std(flux)
        for row in lst:
            if float(row[4]) < multiplier * madstd:
                lst_cut.append(row)
    return lst_cut

def light_curve_event_data(file, period, multiplier, fltr):
    '''
    Make a 2-D array containing the time, flux, and flux uncertainties data only from the period containing the peak flux, with outliers cut, and in a single filter .
    
    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
    period: float
        Half the length of time from which you want to observe.
    multiplier: float
        Determines at what value (multiplier * mad_std) to cut outlier data points.
    fltr: string
        One of the elements from lst_filter (g, r, i, z).
        
    Returns
    -------
    lc_event_data : 2-D array
        2-D array containing the reduced light curve data which will be used to extract parameters for our model.
        
    '''
    data = s2c(file)
    peak_time = float(data[7][1])
    event = peak_event(data[17:-1], period, peak_time)
    table = np.transpose(filter_func
                         (cut_outliers
                          (event, multiplier), fltr))
    if len(table) > 0:
        time = table[1].astype(float) - peak_time
        flux = table[4].astype(float)
        flux_unc = table[5].astype(float)
        lc_event_data = np.array([time, flux, flux_unc])
    return lc_event_data
            
def flux_model(t, A, B, gamma, t_0, tau_rise, tau_fall):
    '''
    Calculate the flux given amplitude, plateau slope, plateau duration, start time, rise time, and fall time using theano.switch. Parameters.type = TensorType(float64, scalar).
    
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
        
    '''
    t_1 = t_0 + gamma
    flux_model = T.switch((t < t_1),
                 ((A + B * (t - t_0)) / 
                  (1. + np.exp((t - t_0) / -tau_rise))),
                 ((A + B * (gamma)) * np.exp((t - gamma - t_0) / -tau_fall) / 
                  (1. + np.exp((t - t_0) / -tau_rise))))
    return flux_model


# In[3]:


def SNe_LC_MCMC(file, fltr, iterations, tuning, walkers):
    '''
    Run a Metropolis Hastings MCMC for a file in a single filter with a certain number iterations, burn in (tuning), and walkers. The period and multiplier are 180 and 20, respectively. 
    
    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
    fltr: string
        One of the elements from lst_filter (g, r, i, z).
    iterations : float
        The number of iterations after tuning.
    tuning : float
        The number of iterations used for tuning.
    walkers : float
        The nnumber of cores and walkers used.
        
    Returns
    -------
    trace : MultiTrace
        The trace has a shape (len(varnames), walkers, iterations) and contains every iteration for each walker for all parameters.
        
    '''
    obs = light_curve_event_data(file, 180, 20, fltr)
    
    with pm.Model() as SNe_LC_model_final:
    
        obs_time = obs[0]
        obs_flux = obs[1]
        obs_unc = obs[2]

        A = 10 ** pm.Uniform('Log(Amplitude)', lower = 0, upper = 6)
        B = np.tan(pm.Uniform('Arctan(Plateau Slope)', lower = -1.56, upper = 0))
        gamma = 10 ** pm.Uniform('Log(Plateau Duration)', lower = -3, upper = 3)
        t_0 = pm.Uniform('Start Time', lower = -50, upper = 50)
        tau_rise = 10 ** pm.Uniform('Log(Rise Time)', lower = -3, upper = 3)
        tau_fall = 10 ** pm.Uniform('Log(Fall Time)', lower = -3, upper = 3)
        sigma = np.sqrt(pm.HalfNormal('sigma', sigma = 1) ** 2 + obs_unc ** 2)
        parameters = [A, B, gamma, t_0, tau_rise, tau_fall]
    
        exp_flux = flux_model(obs_time, *parameters)
    
        flux_post = pm.Normal('Flux_Posterior', mu = exp_flux, sigma = sigma, observed = obs_flux)
        trace = pm.sample(iterations, tune = tuning, cores = walkers, chains = walkers, step = pm.Metropolis())
    
    return trace


# In[4]:
#CREATE A DICTIONARY CONTAINING THE TRACE OF A LIGHT CURVE, WITH 10000 ITERATIONS PER WALKER AND 25 WALKERS PER PARAMETER

file = sys.argv[1]
dict_trace = {}
for var in varname:
    dict_trace[var] = []
for fltr in lst_filter:
    trace = SNe_LC_MCMC(file, fltr, 10000, 25000, 25)
    for var in varname:
        dict_trace[var].append(np.array(trace.get_values(var, combine = False)))
        
np.savez_compressed('/n/home01/fdauphin/NPZ_Files_BIG/' + os.path.basename(file).replace('.snana.dat', '.npz'), **dict_trace)
#CHANGE THE PATH TO SAVE TO A DIFFERENT DIRECTORY
#END
