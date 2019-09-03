#!/usr/bin/env python
# coding: utf-8

# FREDERICK DAUPHIN
# DEPARTMENT OF PHYSICS, CARNEGIE MELLON UNIVERSITY
# ADVISOR: DR. GRIFFIN HOSSEINZADEH
# MENOTR: PROF. EDO BERGER
# CENTER FOR ASTROPHYSICS HARVARD AND SMITHSONIAN
# REU 2019 INTERN PROGRAM
# LAST MODIFIED: 08/27/19
# CHECK IF FUNCTIONS WORK, ADD TRAIN VARIABLE TO ABSOLUTE MAGNITUDE TO CHOOSE Z, COMMENT LOOPS, DOCUMENT CLASSIFICATION
# FUNCTION

# IMPORTS
# SYS USES THE FILE NAME ON THE TERMINAL COMMAND LINE TO RUN THE SCRIPT
# NUMPY CONTAINS MATHEMATICAL FUNCTIONS, INCLUDING MATRIX MANIPULATION FUNCTIONS
# PYMC3 IS A MARKOV CHAIN MONTE CARLO PROCESS LIBRARY THAT WE USED TO EXTRACT PARAMETERS FROM OUR MODEL
# THEANO/.TENSOR IS A COMPONENT OF PYMC3 AND IS NEEDED TO EXPLICITLY CODE OUR MODEL
# MAD_STD IS THE STANDARD DEVIATION ESTIMATED USING MEDIAN ABSOLUTE DEVIATION, WHICH WE USED TO CUT OUTLIER FLUX POINTS
# WARNING DOESN'T PRINT WARNINGS
# EXTINCTION IS NEEDED TO CALCULATE ABSOLUTE MAGNITUDE
# MATPLOTLIB IS NEEDED TO MAKE GRAPHS
# PLANCK 15 IS THE MOST UP TO DATE COSMOLOGY
# SKLEARN IS A MACHINE LEARNING LIBRARY IN PYTHON THAT WE USE FOR PRINCIPAL COMPONENT ANALYSIS, SYNTHETIC MINORITY
# OVERSAMPLING TECHNIQUE, RANDOM FORESTS, AND KFOLDING METHODS.
# LST_FILTER IS A LIST OF THE FOUR FILTERS USED FOR OUR LIGHT CURVES
# VARNAME IS A DICTIONARY CONTAINING ALL OF THE VARIABLE NAMES FOR OUR PARAMETERS IN PYMC3
# Tan(Plateau Angle) IS THE VARIABLE NAME USED FOR DATA EXTRACTED DURING THE SUMMER. ANY NEW BATCHES POST AUGUST 2019
# MUST USE Arctan(Plateau Slope) AS A VARNAME. I AM NOT CHANGING IT HERE FOR MY CONVENIENCE
# DICT_SNE IS A DICTIONARY OF SNE TYPES ASSOCIATED WITH AN ARBITRARY CLASSIFICATION NUMBER, WHICH SCIKIT USES


import warnings
import extinction
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import Planck15 as cosmo_P
from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import itertools
from util import s2c, filter_func, cut_outliers, peak_event

warnings.filterwarnings('ignore')

varname = ['Log(Amplitude)', 'Tan(Plateau Angle)', 'Log(Plateau Duration)',
           'Start Time', 'Log(Rise Time)', 'Log(Fall Time)']
classes = ['SLSNe', 'SNII', 'SNIIn', 'SNIa', 'SNIbc']
effective_wavelengths = [4866., 6215., 7545., 9633.]  # g, r, i, z


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap='Blues'):
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


def Func(t, A, B, gamma, t_0, tau_rise, tau_fall):
    """
    Calculate the flux given amplitude, plateau slope, plateau duration, start time, rise time, and fall time using
    numpy.

    Parameters
    ----------
    t : 1-D numpy array
        Time.
    A : float
        Amplitude of the light curve.
    B : float
        Slope of the plateau after the light curve peaks.
    gamma : float
        The duration of the plateau after the light curve peaks.
    t_0 : float
        Start time, which is very close to when the actaul light curve peak flux occurs.
    tau_rise : float
        Exponential rise time to peak.
    tau_fall : float
        Exponential decay time after the plateau ends.

    Returns
    -------
    flux : numpy array
        1-D array of the predicted flux from the given model.

    """
    flux = np.piecewise(t, [t < t_0 + gamma, t >= t_0 + gamma],
                        [lambda t: ((A + B * (t - t_0)) /
                                    (1. + np.exp((t - t_0) / -tau_rise))),
                         lambda t: ((A + B * gamma) * np.exp((t - gamma - t_0) / -tau_fall) /
                                    (1. + np.exp((t - t_0) / -tau_rise)))])
    return flux


def transform(lst):
    """
    Convert the parameters from the flux model into their original units.
    [Amplitude] = counts
    [Plateau Slope] = counts / day
    [Plateau Duration] = days
    [Start Time] = days
    [Rise Time] = days
    [Fall Time] = days

    Parameters
    ----------
    lst : list
        1-D list containing the flux model parameters for a light curve in the order shown above.

    Returns
    -------
    trans : numpy array
        1-D array with the converted flux parameters.

    """
    parameters = lst
    a = 10 ** parameters[0]
    b = np.tan(parameters[1])
    g = 10 ** parameters[2]
    tr = 10 ** parameters[4]
    tf = 10 ** parameters[5]
    trans = np.array([a, b, g, parameters[3], tr, tf])
    return trans


def produce_lc(file, file_npz, rand_num):
    """
    Make a 2-D list containing a random number of light curves created using the parameters from the npz file with shape
    (number of filters, rand_num).

    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
    file_npz : path (.npz file)
        File extention containing the MCMC run for the light curve.
    rand_num : int
        The number of light curves randomly extracted from the MCMC run.

    Returns
    -------
    lst_rand_lc : list
        2-D list containing a random number of light curves for each filter.

    """
    time = np.arange(-50, 150)
    file_load = np.load(file_npz)
    z_index = np.where(np.transpose(new_ps1z)[0] == file)[0][0]
    z = new_ps1z[z_index][1]
    # z = float(s2c(file)[6][1])
    A_v = float(s2c(file)[5][1]) * 3.1
    lst = []
    for i in varname:
        lst.append(file_load[i])
    lst = np.array(lst)
    lst = np.transpose(lst, axes=[2, 1, 3, 0])
    lst = np.concatenate(lst, axis=1)
    lst_rand_lc = []
    for params, wl_eff in zip(lst, effective_wavelengths):
        lst_rand_filter = []
        for j in range(rand_num):
            A = extinction.ccm89(wl_eff, A_v, 3.1)[0]
            index = np.random.randint(len(params))
            lc = Func(time, *transform(params[index]))
            lum_lc = (4 * np.pi * lc * 10 ** (A / 2.5) *
                      cosmo_P.luminosity_distance(z).value ** 2)
            lst_rand_filter.append(lum_lc)
        lst_rand_lc.append(lst_rand_filter)
    return lst_rand_lc


def mean_lc(file, file_npz, rand_num):
    """
    Make a 2-D list containing the mean light curves of each filter created using the parameters from the npz file.

    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
    file_npz : path (.npz file)
        File extention containing the MCMC run for the light curve.
    rand_num : int
        The number of light curves randomly extracted from the MCMC run.

    Returns
    -------
    lst_mean_lc : list
        2-D list containing the light curve means separated by filter.

    """
    time = np.arange(-50, 150)
    file_load = np.load(file_npz)
    z = float(s2c(file)[6][1])
    A_v = float(s2c(file)[5][1]) * 3.1
    lst = []
    for i in varname:
        lst.append(file_load[i])
    lst = np.array(lst)
    lst = np.transpose(lst, axes=[2, 1, 3, 0])
    lst = np.concatenate(lst, axis=1)
    lst_mean_lc = []
    for params, wl_eff in zip(lst, effective_wavelengths):
        lc_sum = 0
        for j in range(rand_num):
            A = extinction.ccm89(wl_eff, A_v, 3.1)[0]
            index = np.random.randint(len(params))
            lc = Func(time, *transform(params[index]))
            lum_lc = (4 * np.pi * lc * 10 ** (A / 2.5) *
                      cosmo_P.luminosity_distance(z) ** 2)
            lc_sum += lum_lc
        lc_mean = lc_sum / rand_num
        lst_mean_lc.append(lc_mean)
    return lst_mean_lc


def absolute_magnitude(file, fltr, norm=True):
    """
    Calculate the absolute magnitude for a light curve in a given filter.

    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
    fltr: int
        Integer 0-3, corresponding to the filters g, r, i, z.
    norm : boolean
        If True, it will randomly choose an absolute magnitude with the mean and standard deviation equal to the
        absolute magnitude and apparent magnitude standard deviation. This randomness is used for producing copies of
        light curves to make sure not all the copies have the same absolute magnitude. If False, it will choose the
        actual absolute magnitude.

    Returns
    -------
    M, M_norm : float
        The absolute magnitude of the light curve.

    """
    data = s2c(file)
    peak_time = float(data[7][1])
    z_index = np.where(np.transpose(new_ps1z)[0] == file)[0][0]
    z = new_ps1z[z_index][1]
    # z = float(data[6][1])
    A_v = float(data[5][1]) * 3.1
    table = np.transpose(filter_func(cut_outliers(peak_event(data[17:-1], 180, peak_time), 20), fltr))
    lst_flux = table[4]
    index = np.argmax(lst_flux.astype(float))
    min_m = float(table[6][index])
    m_std = float(table[7][index])
    A = extinction.ccm89(effective_wavelengths[fltr], A_v, 3.1)[0]
    mu = cosmo_P.distmod(z).value
    k = 2.5 * np.log10(1 + z)
    M = min_m - mu - A + 32.5 + k
    if norm:
        M_norm = np.mean(np.random.normal(M, m_std, 100))
        return M_norm
    return M


# EVERYTHING FOR THE THIRD CELL WAS PICK AND CHOOSE WHICH TRANSIENTS I WANTED TO USE IN MY SAMPLE. IF YOU HAVE ALL OF
# THE FILES AND PATHS FOR THE LIGHT CURVES AND THEIR MCMC PARAMETERS, YOU CAN LOOP THROUGH USING THOSE DIRECTLY. FOR
# REFERENCE:
# NEW_PS1Z HAS THE CORRECT REDSHIFTS FOR OUR OBJECTS
# LST_CLASS CONTAINS ALL THE PATHS OF SPECTROSCOPICALLY CLASSIFIED TRANSIENTS
# LST_PS1 CONTAINS THE PATHS OF ALL OF THE TRANSIENTS, CLASSIFIED AND UN CLASSIFIED
# PS1CONFIRMED_ONLY_SNE_WITHOUT_OUTLIER IS LST_CLASS WITH EACH PATH CORRESPONDING TO A CLASSIFICATION
# ~/PS1_PS1MD_PSc000000.snana.dat SNIa
# LST_CLASS_FINAL CONTAINS ALL THE PATHS WE USED IN OUR SAMPLE
# LST_CLASS_ID CONTAINS ALL THE CLASSIFICATION NUMBERS (SEE DICT_SNE) FOR EACH PATH, IN INDEX ORDER


new_ps1 = open('/data/reu/fdauphin/new_ps1z.dat').readlines()[1:]
new_ps1z = []
for i in new_ps1:
    row = i.split()
    new_ps1z.append([row[0][3:], float(row[1])])

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
for classid in range(5):
    lst_ij = []
    for j in lst_conf:
        if i == j[1]:
            lst_ij.append([j[0][3:], classid])
    lst_coswo.append(lst_ij)

lst_exists = []
lst_not_exists = []
for i in lst_class:
    if i != '380108':
        try:
            file = np.load('/data/reu/fdauphin/NPZ_Files_BIG_CLASS/PS1_PS1MD_PSc' + i + '.npz')
            lst_exists.append(i)
        except FileNotFoundError as e:
            lst_not_exists.append(i)
lst_exists = np.array(lst_exists)
lst_not_exists = np.array(lst_not_exists)

lst_class_final = []
lst_nan = []
for i in lst_exists:
    M = absolute_magnitude(i, 3, norm=False)
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

train_class = []
for _, classname in lst_conf:
    train_class.append(classes.index(classname))
train_class = np.array(train_class)

lst_class_id = []
for i in range(len(lst_class)):
    if i not in lst_index_err:
        lst_class_id.append(train_class[i])
lst_class_id = np.array(lst_class_id)

# CREATING COPIES OF THE CLASSIFICATION NUMBERS IN INDEX ORDER. PRINTING OUT THE SHAPES HELPS TO MAKE SURE EVERYTHING IS
# LOOPING CORRECTLY.
# [0, 1] --> [0, 0, 0, 0, 1, 1, 1, 1]

copies = 4

lst_class_id_super = []
for i in lst_class_id:
    lst_class_id_super.append(np.linspace(i, i, copies))
lst_class_id_super = np.array(lst_class_id_super)
lst_class_id_super = np.concatenate(lst_class_id_super, axis=0)

# CREATING COPIES OF THE PATH NAMES IN INDEX ORDER
# ['000000', '000001'] --> ['000000', '000000', '000000', '000000', '000001', '000001', '000001', '000001']

lst_class_final_super = []
for i in lst_class_final:
    lst_class_final_super.append([i] * copies)
lst_class_final_super = np.array(lst_class_final_super)
lst_class_final_super = np.concatenate(lst_class_final_super, axis=0)

# CREATING COPIES OF THE LIGHT CURVES IN INDEX ORDER. I DID NOT EXPLICITLY USE THE NPZ_FILE VARIABLE BECAUSE I JUST
# LOOPED USING IT'S ID NUMBER. YOU WILL NEED TO MAKE A "LST_NPZ" LIST WITH ALL THE NPZ FILES IN INDEX ORDER.

lst_class_lc_super = []
for filename in lst_class_final:
    lst_class_lc_super.append(produce_lc(filename, lst_npz_classified[i], copies))
lst_class_lc_super = np.array(lst_class_lc_super)
lst_class_lc_super = np.concatenate(lst_class_lc_super, axis=1)

# CONVERTING LIGHT CURVES INTO WHITEN PRINCIPAL COMPONENTS IN EACH FILTER.

lst_pca_smart = []
lst_M = []
pca = decomposition.PCA(n_components=5, whiten=True)
for model_lc in lst_class_lc_super:
    PCA = pca.fit(model_lc)
    lst_clc_trans = pca.transform(model_lc)
    lst_pca_smart.append(lst_clc_trans)
lst_pca_smart = np.array(lst_pca_smart)

# FOR EACH LIGHT CURVE, CONCATENATE ALL THE PRINCIPAL COMPONENTS FOR EACH FILTER INTO ONE ARRAY AND APPEND THE ABSOLUTE
# VALUES FROM EACH FILTER.

lst_M = []
for fltr in range(4):
    lst_M_ij = []
    for j in lst_class_final:
        lst_ij = []
        for k in range(copies):
            M = [absolute_magnitude(j, fltr)]
            lst_ij.append(M)
        lst_M_ij.append(lst_ij)
    lst_M.append(lst_M_ij)
lst_M = np.array(lst_M)

lst_M = np.transpose(lst_M, axes=[1, 0, 2, 3])
lst_M_smart = np.concatenate(lst_M, axis=1)
lst_pca = np.append(lst_M_smart, lst_pca_smart, axis=2)
lst_pca = np.concatenate(lst_pca, axis=1)

print(lst_M_smart.shape, lst_pca.shape)

folds = int(len(lst_class_id_super) / copies)
kf = KFold(folds)


def pca_smote_rf(lst_pca, lst_class_id_super, size, n_est, depth=None, max_feat=None):
    """
    Make a random forest classifier using synthetic minority oversampling technique and kfolding. A lot of the
    documentation is taken directly from SciKit Learn page and should be referenced for further questions.

    Parameters
    ----------
    lst_pca : numpy array
        2-D array of the principal components for each transient.
    lst_class_id_super: numpy array
        The classification numbers of each transient and their copies.
    size: float
        Should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
    n_est: float
        The number of trees in the forest.
    depth : int or None
        The maxiumum depth of a tree. If None, the tree will have all pure leaves.
    max_feat : int or None
        The maximum number of used before making a split. If None, max_feat = n_features.

    Returns
    -------
    clf : RandomForestClassifier
        A random forest classifier trained from the classified transients. It will print which fold it is on and a
        confusion matrix to assess how well the forest performed.

    """

    class_pred = np.zeros(len(lst_class_id_super))
    cnt = 0

    for train_index, test_index in kf.split(lst_pca):
        print(cnt)
        lst_pca_train, lst_pca_test = lst_pca[train_index], lst_pca[test_index]
        class_train, class_test = lst_class_id_super[train_index], lst_class_id_super[test_index]

        sampler = SMOTE(random_state=0)

        lst_pca_res, class_res = sampler.fit_resample(lst_pca_train, class_train)

        lst_pca_r_train, lst_pca_r_test, class_r_train, class_r_test = train_test_split(
            lst_pca_res, class_res, test_size=size, random_state=42)

        clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, class_weight='balanced',
                                     criterion='entropy', max_features=max_feat)

        clf.fit(lst_pca_r_train, class_r_train)
        class_pred[test_index] = clf.predict(lst_pca_test)

        cnt += 1

    cat_names = ['SLSNe', 'SNII', 'SNIIn', 'SNIa', 'SNIbc']

    cnf_matrix = confusion_matrix(lst_class_id_super, class_pred)
    plot_confusion_matrix(cnf_matrix, classes=cat_names, normalize=True)

    return clf


# MAKE RANDOM FOREST WITH SIZE .33 AND N_EST 100

clf = pca_smote_rf(lst_pca, lst_class_id_super, 0.33, 100)

# MAKE A LIST OF UNCLASSIFIED TRANSIENTS WITH COMPLETED MCMC RUNS

lst_PS1_final = []
for i in lst_PS1:
    if i != '530540' and i != '000518' and i != '420082':
        try:
            file = np.load('/data/reu/fdauphin/NPZ_Files_BIG/PS1_PS1MD_PSc' + i + '.npz')
            lst_PS1_final.append(i)
        except FileNotFoundError as e:
            print(i, 'not found')

# RECREATE LIGHT CURVES FOR THE UNCLASSIFIED TRANSIENTS USING THEIR MCMC PARAMETERS. WILL NEED TO MAKE A LIST OF PATHS
# FOR THE NPZ FILES OF UNCLASSIFIED TRANSIENTS TO LOOP THROUGH CORRECTLY.

lst_photo_lc = []
for filename in lst_PS1_final:
    lst_photo_lc.append(mean_lc(filename, lst_npz_unclassified[i], 100))
lst_photo_lc = np.array(lst_photo_lc)

lst_photo_lc = np.concatenate(lst_photo_lc, axis=1)

# CONVERT THE LIGHT CURVES OF INCLASSIFIED TRANSIENTS TO PRINCIPAL COMPONENTS AND CLASSIFY THEM.

lst_pca_unclassified = []
pca = decomposition.PCA(n_components=5, whiten=True)
for filename in lst_photo_lc:
    PCA = pca.fit(filename)
    lst_plc_trans = pca.transform(filename)
    lst_pca_unclassified.append(lst_plc_trans)
lst_pca_unclassified = np.array(lst_pca_unclassified)

clf.predict(lst_pca_unclassified)
