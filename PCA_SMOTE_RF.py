#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import sys
import theano
import extinction
import numpy as np
import pymc3 as pm
import theano.tensor as T
from astropy.stats import mad_std
from sklearn.ensemble import RandomForestClassifier
from astropy.cosmology import Planck15
cosmo_P = Planck15
from sklearn import decomposition
import matplotlib.pyplot as plt

lst_filter = ['g', 'r', 'i', 'z']
dict_filter = {'g' : 0, 'r' : 1, 'i' : 2, 'z' : 3}
varname = ['Log(Amplitude)', 'Tan(Plateau Angle)', 'Log(Plateau Duration)',
              'Start Time', 'Log(Rise Time)', 'Log(Fall Time)']
dict_sne = {'SNIa' : 0, 'SNIbc' : 1, 'SLSNe' : 2, 'SNII' : 3, 'SNIIn' : 4}


# In[30]:


def s2c(file_num):
    file = '/data/reu/fdauphin/PS1_MDS/PS1_Final/PS1_PS1MD_PSc' + file_num + '.snana.dat'
    #file = file_num
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

def fltr_2_lmbd(flt):
    if flt == 'g':
          x = 4866.
    elif flt == 'r':
          x =  6215.
    elif flt == 'i':
          x = 7545.
    elif flt == 'z':
          x = 9633.
    return np.array([x])

def three_places(x):
    for i in range (len(x)):
          if x[i] == '.':
                index = i
    part_1 = x[:index]
    part_2 = x[index:]
    while len(part_2) != 4:
          part_2 += '0'
    return part_1 + part_2

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

                
def Func(t, A, B, gamma, t_0, tau_rise, tau_fall):
    F = np.piecewise(t, [t < t_0 + gamma, t >= t_0 + gamma],
                             [lambda t: ((A + B * (t - t_0)) /
                                                (1. + np.exp((t - t_0) / -tau_rise))),
                                lambda t: ((A + B * (gamma)) * np.exp((t - gamma - t_0) / -tau_fall) /
                                                (1. + np.exp((t - t_0) / -tau_rise)))])
    return F

def transform(lst):
    parameters = lst
    a = 10 ** parameters[0]
    b = np.tan(parameters[1])
    g = 10 ** parameters[2]
    tr = 10 ** parameters[4]
    tf = 10 ** parameters[5]
    trans = np.array([a, b, g, parameters[3], tr, tf])
    return trans

def absolute_magnitude(file_num, fltr):
    data = s2c(file_num)
    peak_time = float(data[7][1])
    z = float(data[6][1])
    A_v = float(data[5][1]) * 3.1
    table = np.transpose(filter_func
                                    (cut_outliers
                                     (peak_event
                                      (data[17:-1], 180, peak_time), 20), fltr))
    lst_flux = table[4]
    max_flux = max(lst_flux.astype(float))
    index = np.where(lst_flux == three_places(str(max_flux)))[0][0]
    min_m = float(table[6][index])
    m_std = float(table[7][index])
    A = extinction.ccm89(fltr_2_lmbd(fltr), A_v, 3.1)[0]
    mu = cosmo_P.distmod(z).value
    k = 2.5 * np.log10(1 + z)
    M = min_m - mu - A + 32.5 + k
    M_norm = np.mean(np.random.normal(M, m_std, 100))
    #return M
    return M_norm

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

def run():
    file_num = '000190'
    dict_trace = {}
    dict_multi_trace = {}
    for var in varname:
          dict_trace[var] = []
    for fltr in lst_filter:
        trace = SNe_LC_MCMC(file_num, fltr, 5000, 10000, 25)
        dict_multi_trace[fltr] = trace
        for var in varname:
            dict_trace[var].append(np.array(trace.get_values(var, combine = False)))

from sklearn.metrics import confusion_matrix
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import LeaveOneOut,train_test_split
loo = LeaveOneOut()
import itertools

def plot_confusion_matrix(cm, classes,
                                     normalize=False,
                                     title='Confusion matrix',
                                     cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From tutorial: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    plt.axis('equal')
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('/n/home01/fdauphin/plot.png')

def mean_lc(file_num, rand_num):
    time = np.arange(-50,150)
    file = np.load('/data/reu/fdauphin/NPZ_Files_CLASS_12/PS1_PS1MD_PSc'+file_num+'.npz')
    lst = []
    for i in varname:
        lst.append(file[i])
    lst = np.array(lst)
    lst = np.transpose(lst, axes = [2,1,3,0])
    lst = np.concatenate(lst, axis = 1)
    lst_mean_lc = []
    for i in range (len(lst)):
        lc_sum = 0
        for j in range (rand_num):
            index = np.random.randint(len(lst[i]))
            lc = Func(time, *transform(lst[i][index]))
            lc_sum += lc
        lc_mean = lc_sum / rand_num
        lst_mean_lc.append(lc_mean)
    return lst_mean_lc


# In[3]:


cls_x = open('/data/reu/fdauphin/lst_class.txt')
cls = cls_x.readlines()
cls_x.close()
lst_class = []
for i in cls:
    lst_class.append(i[:-1])
ps1_x = open('/data/reu/fdauphin/lst_PS1.txt')
ps1 = ps1_x.readlines()
ps1_x.close()
lst_PS1 = []
for i in ps1:
    lst_PS1.append(i[:-1])
conf_x = open('/data/reu/fdauphin/PS1_MDS/ps1confirmed_only_sne_without_outlier.txt')
conf = conf_x.readlines()[1:]
conf_x.close()
lst_conf = []
for i in conf:
    lst_conf.append(i.split())

lst_coswo = []
for i in dict_sne:
    lst_ij = []
    for j in lst_conf:
          if i == j[1]:
                lst_ij.append([j[0][3:], dict_sne[i]])
    lst_coswo.append(lst_ij)

lst_exists = []
lst_not_exists = []
for i in lst_class:
    try:
          file = np.load('/data/reu/fdauphin/NPZ_Files_BIG_CLASS/PS1_PS1MD_PSc'+i+'.npz')
          lst_exists.append(i)
    except FileNotFoundError as e:
          lst_not_exists.append(i)
lst_exists = np.array(lst_exists)
lst_not_exists = np.array(lst_not_exists)

lst_class_final = []
lst_nan = []
for i in lst_exists:
    M = absolute_magnitude(i, 'z')
    if str(M) != 'nan':
          lst_class_final.append(i)
    else:
          lst_nan.append(i)
lst_class_final = np.array(lst_class_final)
lst_nan = np.array(lst_nan)

lst_err = np.concatenate((lst_not_exists, lst_nan))

lst_index_err = []
for i in lst_err:
    lst_index_err.append(np.where(np.array(lst_class) == i)[0][0])


# In[4]:


train_class = []
for i in lst_conf:
    train_class.append(dict_sne[i[1]])
train_class = np.array(train_class)

lst_class_id = []
for i in range (len(lst_class)):
    if i not in lst_index_err:
          lst_class_id.append(train_class[i])
lst_class_id = np.array(lst_class_id)


# In[5]:


def produce_lc(file_num, rand_num):
    time = np.arange(-50,150)
    file = np.load('/data/reu/fdauphin/NPZ_Files_CLASS_12/PS1_PS1MD_PSc'+file_num+'.npz')
    lst = []
    for i in varname:
        lst.append(file[i])
    lst = np.array(lst)
    lst = np.transpose(lst, axes = [2,1,3,0])
    lst = np.concatenate(lst, axis = 1)
    lst_rand_lc = []
    for i in range (len(lst)):
        lst_rand_filter = []
        for j in range (rand_num):
            index = np.random.randint(len(lst[i]))
            lc = Func(time, *transform(lst[i][index]))
            lst_rand_filter.append(lc)
        lst_rand_lc.append(lst_rand_filter)
    return lst_rand_lc


# In[7]:


copies = 10


# In[31]:


#id with 100 copies of each object
lst_class_id_super = []
for i in lst_class_id:
    lst_class_id_super.append(np.linspace(i, i, copies))
lst_class_id_super = np.array(lst_class_id_super)

lst_class_id_super = np.concatenate(lst_class_id_super, axis = 0)


# In[32]:


lst_class_id_super.shape


# In[33]:


#mean light curves with 1 copies of each object
#lst_class_lc = []
#for i in lst_class_final:
 #   lst_class_lc.append(mean_lc(i, 1))
#lst_class_lc = np.array(lst_class_lc)
#lst_class_lc = np.transpose(lst_class_lc, axes = [1,0,2])


# In[34]:


#light curve numbers with 100 copies of each object
lst_class_final_super = []
for i in lst_class_final:
    lst_class_final_super.append([i] * copies)
lst_class_final_super = np.array(lst_class_final_super)

lst_class_final_super = np.concatenate(lst_class_final_super, axis = 0)


# In[35]:


lst_class_final_super.shape


# In[36]:


#light curve numbers with 100 copies of each object
lst_class_lc_super = []
for i in lst_class_final:
    lst_class_lc_super.append(produce_lc(i, copies))
lst_class_lc_super = np.array(lst_class_lc_super)

lst_class_lc_super = np.concatenate(lst_class_lc_super, axis = 1)


# In[37]:


lst_class_lc_super.shape


# In[38]:


lst_pca_smart = []
lst_M = []
pca = decomposition.PCA(n_components = 5, whiten = True)
for i in range (len(lst_class_lc_super)):
    PCA = pca.fit(lst_class_lc_super[i])
    lst_clc_trans = pca.transform(lst_class_lc_super[i])
    lst_pca_smart.append(lst_clc_trans)
lst_pca_smart = np.array(lst_pca_smart)


# In[39]:


lst_pca_smart.shape


# In[51]:


lst_M = []
for i in lst_filter:
    lst_M_ij = []
    for j in lst_class_final:
        lst_ij = []
        for k in range (copies):
            #M = absolute_magnitude(j, i)
            M = [absolute_magnitude(j, i)]
            lst_ij.append(M)
        lst_M_ij.append(lst_ij)
    lst_M.append(lst_M_ij)
lst_M = np.array(lst_M)

lst_M = np.transpose(lst_M, axes = [1, 0, 2, 3])
lst_M_smart = np.concatenate(lst_M, axis = 1)
lst_pca = np.append(lst_M_smart, lst_pca_smart, axis = 2)
lst_pca = np.concatenate(lst_pca, axis = 1)


# In[52]:


lst_M_smart.shape


# In[53]:


lst_pca.shape


# In[57]:


p = 10


# In[58]:


from sklearn.model_selection import LeavePOut
lpo = LeavePOut(p)

def pca_smote_rf(lst_pca, lst_class_id_super, size, n_est, depth, max_feat):

    class_pred = np.zeros(len(lst_class_id_super))

    for train_index, test_index in lpo.split(lst_pca):
        lst_pca_train, lst_pca_test = lst_pca[train_index], lst_pca[test_index]
        class_train, class_test = lst_class_id_super[train_index], lst_class_id_super[test_index]

        sampler = SMOTE(random_state=0)

        lst_pca_res, class_res = sampler.fit_resample(lst_pca_train, class_train)

        lst_pca_r_train, lst_pca_r_test, class_r_train, class_r_test =             train_test_split(lst_pca_res, class_res, test_size = size, random_state = 42)

        clf = RandomForestClassifier(n_estimators = n_est, max_depth = depth, random_state = 42,
                                            class_weight = 'balanced', criterion = 'entropy', max_features = max_feat)

        fit = clf.fit(lst_pca_r_train, class_r_train)
        class_pred[test_index] = clf.predict(lst_pca_test)

    #  Before we plot, we will make labels for both classes
    cat_names = ['SNIa', 'SNIbc', 'SLSNe', 'SNII', 'SNIIn']

    cnf_matrix = confusion_matrix(lst_class_id_super,class_pred)
    plot_confusion_matrix(cnf_matrix, classes=cat_names, normalize=True)

    return clf


# In[ ]:


pca_smote_rf(lst_pca, lst_class_id_super, 0.33, 25, None, None)


# In[ ]:




