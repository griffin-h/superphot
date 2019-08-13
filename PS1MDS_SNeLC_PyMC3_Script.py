#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import theano
import numpy as np
import pymc3 as pm
import theano.tensor as T
from astropy.stats import mad_std
lst_filter = ['g', 'r', 'i', 'z']
dict_filter = {'g' : 0, 'r' : 1, 'i' : 2, 'z' : 3}
varname = ['Log(Amplitude)', 'Tan(Plateau Angle)', 'Log(Plateau Duration)', 
           'Start Time', 'Log(Rise Time)', 'Log(Fall Time)']


# In[2]:


def s2c(file_num):
    #file = '/data/reu/fdauphin/PS1_MDS/PS1_Final/PS1_PS1MD_PSc' + file_num + '.snana.dat'
    file = file_num
    with open(file) as infile:
        lst = []
        for line in infile:
            lst.append(line.split())
    return lst

def filter_func(lst, flt):
    lst_new = []
    for row in lst:
        if row[2] == flt:
            lst_new.append(row)
    return lst_new

def peak_event(data, period, peak_time):
    lst = []
    for i in data:
        time_ij = float(i[1])
        if time_ij < peak_time + period and time_ij > peak_time - period:
            lst.append(i)
    return lst

def cut_outliers(data, multiplier):
    lst = []
    table = np.transpose(data)
    if len(table) > 0:
        flux = table[4].astype(float)
        madstd = mad_std(flux)
        for i in data:
            if float(i[4]) < multiplier * madstd:
                lst.append(i)
    return lst

def light_curve_event_data(file_num, period, multiplier, fltr):
    data = s2c(file_num)
    peak_time = float(data[7][1])
    event = peak_event(data[17:-1], period, peak_time)
    for i in range (len(lst_filter)):
        table = np.transpose(filter_func
                             (cut_outliers
                              (event, multiplier), lst_filter[i]))
        if len(table) > 0:
            time = table[1].astype(float) - peak_time
            flux = table[4].astype(float)
            flux_unc = table[5].astype(float)
            if lst_filter[i] == fltr:
                return np.array([time, flux, flux_unc])
            
def Func_switch(t, A, B, gamma, t_0, tau_rise, tau_fall):
    t_1 = t_0 + gamma
    F = T.switch((t < t_1),
                 ((A + B * (t - t_0)) / 
                  (1. + np.exp((t - t_0) / -tau_rise))),
                 ((A + B * (gamma)) * np.exp((t - gamma - t_0) / -tau_fall) / 
                  (1. + np.exp((t - t_0) / -tau_rise))))
    return F


# In[3]:


def SNe_LC_MCMC(file_num, fltr, iterations, tuning, walkers):

    obs = light_curve_event_data(file_num, 180, 20, fltr)
    
    with pm.Model() as SNe_LC_model_final:
    
        obs_time = obs[0]
        obs_flux = obs[1]
        obs_unc = obs[2]

        A = 10 ** pm.Uniform('Log(Amplitude)', lower = 0, upper = 6)
        B = np.tan(pm.Uniform('Tan(Plateau Angle)', lower = -1.56, upper = 0))
        gamma = 10 ** pm.Uniform('Log(Plateau Duration)', lower = -3, upper = 3)
        t_0 = pm.Uniform('Start Time', lower = -50, upper = 50)
        tau_rise = 10 ** pm.Uniform('Log(Rise Time)', lower = -3, upper = 3)
        tau_fall = 10 ** pm.Uniform('Log(Fall Time)', lower = -3, upper = 3)
        sigma = np.sqrt(pm.HalfNormal('sigma', sigma = 1) ** 2 + obs_unc ** 2)
        parameters = [A, B, gamma, t_0, tau_rise, tau_fall]
    
        exp_flux = Func_switch(obs_time, *parameters)
    
        flux_post = pm.Normal('Flux_Post', mu = exp_flux, sigma = sigma, observed = obs_flux)
        trace = pm.sample(iterations, tune = tuning, cores = walkers, chains = walkers, step = pm.Metropolis())
    
    return trace


# In[79]:


file_num = sys.argv[1]
#file_num = '000190'
dict_trace = {}
for fltr in lst_filter:
    for var in varname:
        dict_trace[varname] = SNe_LC_MCMC(file_num, fltr, 5000, 10000, 20)


# In[118]:


#dict_param = [{var: np.array(dict_trace[fltr].get_values(var, combine=False)) 
 #                     for var in varname}
  #            for fltr in lst_filter]
new_dict = {}
for var in varname:
    array_3d = []
    for fltr in lst_filter:
        array_3d.append(dict_trace[fltr].get_values(var, combine = False))
    new_dict[var] = np.dstack(array_3d)
np.savez_compressed('/data/reu/fdauphin/' + file_num[:19] + '.npz', **new_dict)


# In[ ]:




