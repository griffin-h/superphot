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


import extinction
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import os
import argparse
from astropy.table import Table, join
from astropy.cosmology import Planck15 as cosmo_P
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import itertools
from util import read_snana, light_curve_event_data
from fit_model import setup_model

classes = ['SLSNe', 'SNII', 'SNIIn', 'SNIa', 'SNIbc']
effective_wavelengths = np.array([4866., 6215., 7545., 9633.])  # g, r, i, z


def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap='Blues'):
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


def flux_model(t, A, B, gamma, t_0, tau_rise, tau_fall):
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


def load_trace(file, trace_path='.'):
    """
    Read the stored PyMC3 traces into a 3-D array with shape (nfilters, nsteps, nparams).

    Parameters
    ----------
    file : str
        Filename of the original SNANA data file.
    trace_path : str, optional
        Directory where the PyMC3 trace data is stored. Default: current directory.

    Returns
    -------
    lst : numpy.array
        PyMC3 trace stored as 3-D array with shape (nfilters, nsteps, nparams).
    """
    tracefile = os.path.join(trace_path, os.path.basename(file).replace('.snana.dat', '_{}'))
    lst = []
    t = light_curve_event_data(file)
    for fltr in 'griz':
        obs = t[t['FLT'] == fltr]
        model, varnames = setup_model(obs)
        trace = pm.load_trace(tracefile.format(fltr), model)
        trace_values = np.transpose([trace.get_values(var) for var in varnames])
        lst.append(trace_values)
    lst = np.array(lst)
    return lst


def produce_lc(row, rand_num, trace_path='.'):
    """
    Make a 2-D list containing a random number of light curves created using the parameters from the npz file with shape
    (number of filters, rand_num).

    Parameters
    ----------
    row : astropy.table.row.Row
        Astropy table row for a given transient, containing columns 'filename', 'A_V', and 'redshift'/'hostz'
    rand_num : int
        The number of light curves randomly extracted from the MCMC run.
    trace_path : str, optional
        Directory where the PyMC3 trace data is stored. Default: current directory.

    Returns
    -------
    lst_rand_lc : list
        2-D list containing a random number of light curves for each filter.

    """
    time = np.arange(-50, 150)
    if 'redshift' in row.colnames and not np.ma.is_masked(row['redshift']):
        z = row['redshift']
    else:
        z = row['hostz']
    try:
        lst = load_trace(row['filename'], trace_path=trace_path)
    except ValueError:
        lst = np.tile(np.nan, (rand_num, 4, len(time)))
    lst_rand_filter = []
    for j in range(rand_num):
        lst_rand_lc = []
        A_coeffs = extinction.ccm89(effective_wavelengths, row['A_V'], 3.1)
        for params, A in zip(lst, A_coeffs):
            index = np.random.randint(len(params))
            lc = flux_model(time, *transform(params[index]))
            lum_lc = (4 * np.pi * lc * 10 ** (A / 2.5) *
                      cosmo_P.luminosity_distance(z).value ** 2)
            lst_rand_lc.append(lum_lc)
        lst_rand_filter.append(lst_rand_lc)
    return np.array(lst_rand_filter)


def absolute_magnitude(file, z=None):
    """
    Calculate the peak absolute magnitudes for a light curve in each filter.

    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
    z : float, optional
        Redshift of the transient. Default: use the redshift in the SNANA file.

    Returns
    -------
    M : float
        The absolute magnitude of the light curve.

    """
    min_m = []
    t = light_curve_event_data(file)
    for fltr in 'griz':
        obs = t[t['FLT'] == fltr]
        if len(obs):
            min_m.append(obs['MAG'].min())
        else:
            min_m.append(np.nan)
    min_m = np.array(min_m)
    if not z:
        z = t.meta['REDSHIFT']
    A_v = t.meta['A_V']
    A = extinction.ccm89(effective_wavelengths, A_v, 3.1)
    mu = cosmo_P.distmod(z).value
    k = 2.5 * np.log10(1 + z)
    M = min_m - mu - A + 32.5 + k
    return M


def get_principal_components(light_curves, n_components=5, whiten=True):
    """
    Run a principal component analysis on a list of light curves and return a list of their 5 principal components.

    Parameters
    ----------
    light_curves : array-like
        A list of evenly-sampled model light curves.
    n_components : int, optional
        The number of principal components to calculate. Default: 5.
    whiten : bool
        Whiten the input data before calculating the principal components. Default: True.

    Returns
    -------
    principal_components : array-like
        A list of the principal components for each of the input light curves.
    """
    principal_components = []
    pca = PCA(n_components, whiten=whiten)
    for lc_filter in np.moveaxis(light_curves, 1, 0):
        pca.fit(lc_filter)
        princ_comp = pca.transform(lc_filter)
        principal_components.append(princ_comp)
    principal_components = np.array(principal_components)
    principal_components = np.moveaxis(principal_components, 0, 1)
    return principal_components


def pca_smote_rf(lst_pca, lst_class_id_super, size, n_est, folds, depth=None, max_feat=None):
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
    folds : int
        Number of splits for k-fold cross-validation.
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

    class_pred = np.empty_like(lst_class_id_super, int)
    kf = KFold(folds)
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, class_weight='balanced',
                                 criterion='entropy', max_features=max_feat)
    sampler = SMOTE(random_state=0)

    for train_index, test_index in kf.split(lst_pca):
        lst_pca_train, lst_pca_test = lst_pca[train_index], lst_pca[test_index]
        class_train, class_test = lst_class_id_super[train_index], lst_class_id_super[test_index]

        lst_pca_res, class_res = sampler.fit_resample(lst_pca_train, class_train)

        lst_pca_r_train, lst_pca_r_test, class_r_train, class_r_test = train_test_split(lst_pca_res, class_res,
                                                                                        test_size=size, random_state=42)

        clf.fit(lst_pca_r_train, class_r_train)
        class_pred[test_index] = clf.predict(lst_pca_test)

    cnf_matrix = confusion_matrix(lst_class_id_super, class_pred)
    plot_confusion_matrix(cnf_matrix, normalize=True)
    return clf


def extract_features(t, ndraws, trace_path='.'):
    """
    Extract features for a table of model light curves: the peak absolute magnitudes and principal components of the
    light curves in each filter.

    Parameters
    ----------
    t : astropy.table.Table
        Table containing the 'filename' and 'redshift' of each transient to be classified.
    ndraws : int
        Number of random draws from the MCMC posterior.
    trace_path : str, optional
        Directory where the PyMC3 trace data is stored. Default: current directory.

    Returns
    -------
    features : numpy.ndarray
        2-D array of 24 features corresponding to each draw from the posterior. Shape = (len(t) * ndraws, 24).
    """
    peakmags = np.concatenate([np.tile(absolute_magnitude(row['filename'], z=row['redshift']), (ndraws, 1))
                               for row in t])
    models = np.concatenate([produce_lc(row['filename'], ndraws, z=row['redshift'], trace_path=trace_path)
                             for row in t])
    good = ~np.isnan(peakmags).any(axis=1) & ~np.isnan(models).any(axis=2).any(axis=1)
    pcs = get_principal_components(models[good])
    features = np.dstack([peakmags[good], pcs]).reshape(-1, 24)
    return features, good


def meta_table(filenames):
    t_meta = Table(names=['id', 'A_V', 'hostz'], dtype=['S9', float, float], masked=True)
    for filename in filenames:
        t = read_snana(filename)
        t_meta.add_row([t.meta['SNID'], t.meta['A_V'], t.meta['REDSHIFT']])
    t_meta['filename'] = filenames
    t_meta['hostz'].mask = t_meta['hostz'] < 0.
    t_meta['A_V'].format = '%.5f'
    t_meta['hostz'].format = '%.4f'
    return t_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', type=str, help='Input SNANA files')
    parser.add_argument('--ndraws', type=int, default=4, help='Number of draws from the LC posterior for training set')
    parser.add_argument('--trace-path', type=str, default='.', help='Directory where the PyMC3 trace data is stored')
    args = parser.parse_args()

    lst_input = meta_table(args.filenames)
    new_ps1z = Table.read('new_ps1z.dat', format='ascii')  # redshifts of 521 classified SNe
    lst_conf = Table.read('ps1confirmed_only_sne_without_outlier.txt', format='ascii')  # classifications of 499 SNe
    bad_lcs = Table.read('bad_lcs.dat', names=['idnum', 'flag0', 'flag1'], format='ascii', fill_values=('-', '0'))
    bad_lcs['id'] = ['PSc{:0>6d}'.format(idnum) for idnum in bad_lcs['idnum']]  # 1227 VAR, AGN, QSO transients
    bad_lcs.remove_column('idnum')
    bad_lcs_2 = np.loadtxt('bad_lcs_2.dat', dtype=str, usecols=[0, -1])  # 526 transients with bad host galaxy spectra
    bad_lcs_2 = Table([['PSc' + idnum for idnum in bad_lcs_2[:, 0]], bad_lcs_2[:, 1]], names=['id', 'flag2'])

    lst_final = join(lst_input, new_ps1z, join_type='left')
    lst_final = join(lst_final, lst_conf, join_type='left')
    lst_final = join(lst_final, bad_lcs, join_type='left')
    lst_final = join(lst_final, bad_lcs_2, join_type='left')

    lst_final = lst_final[lst_final['flag0'].mask & lst_final['flag1'].mask & lst_final['flag2'].mask
                          & ~lst_final['hostz'].mask]
    lst_train = lst_final[~lst_final['type'].mask]

    features_train, good_train = extract_features(lst_train, args.ndraws, trace_path=args.trace_path)
    classid_train = np.repeat([classes.index(t) for t in lst_train['type']], args.ndraws)[good_train]
    folds = good_train.sum() // args.ndraws
    clf = pca_smote_rf(features_train, classid_train, size=0.33, n_est=100, folds=folds)

    features_test, good_test = extract_features(lst_final, args.ndraws)
    classid_test = clf.predict(features_test)
    good_final = good_test.reshape(-1, args.ndraws).all(axis=1)
    classid_final = classid_test.reshape(-1, args.ndraws)
    for i, classname in enumerate(classes):
        lst_final[classname] = -1
        lst_final[classname][good_final] = (classid_final == i).sum(axis=1) / args.ndraws
        lst_final[classname].mask = ~good_final
    lst_final[['id', 'redshift', 'type'] + classes].write('results.txt', format='ascii.fixed_width')
