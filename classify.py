#!/usr/bin/env python

# FREDERICK DAUPHIN
# DEPARTMENT OF PHYSICS, CARNEGIE MELLON UNIVERSITY
# ADVISOR: DR. GRIFFIN HOSSEINZADEH
# MENTOR: PROF. EDO BERGER
# CENTER FOR ASTROPHYSICS | HARVARD & SMITHSONIAN
# REU 2019 INTERN PROGRAM

import extinction
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
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
from fit_model import setup_model, flux_model

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


def load_trace(file, trace_path='.', version='2'):
    """
    Read the stored PyMC3 traces into a 3-D array with shape (nfilters, nsteps, nparams).

    Parameters
    ----------
    file : str
        Filename of the original SNANA data file.
    trace_path : str, optional
        Directory where the PyMC3 trace data is stored. Default: current directory.
    version : str, optional
        Version of the fit to use, where "version" is the character in the filename before the filter. Default: '2'.

    Returns
    -------
    lst : numpy.array
        PyMC3 trace stored as 3-D array with shape (nfilters, nsteps, nparams).
    """
    tracefile = os.path.join(trace_path, os.path.basename(file).replace('.snana.dat', '_{}{}'))
    lst = []
    t = light_curve_event_data(file)
    for fltr in 'griz':
        obs = t[t['FLT'] == fltr]
        model, varnames = setup_model(obs)
        trace = pm.load_trace(tracefile.format(version, fltr), model)
        trace_values = np.transpose([trace.get_values(var) for var in varnames])
        lst.append(trace_values)
    lst = np.array(lst)
    return lst


def produce_lc(trace, tmin=-50., tmax=150.):
    """
    Load the stored PyMC3 traces and produce model light curves from the parameters.

    Parameters
    ----------
    trace : numpy.array
        PyMC3 trace stored as 3-D array with shape (nfilters, nsteps, nparams).
    tmin : float, optional
        Minimum phase (in days, with respect to PEAKMJD) to calculate the model. Default: -50.
    tmax : float, optional
        Maximum phase (in days, with respect to PEAKMJD) to calculate the model. Default: 150.

    Returns
    -------
    time : numpy.array
        Range of times over which the model was calculated.
    lc : numpy.array
        Model light curves. Shape = (len(trace) * nwalkers, nfilters, len(time)).
    """
    time = np.arange(tmin, tmax)
    tt.config.compute_test_value = 'ignore'
    parameters = tt.dmatrices(6)
    flux = flux_model(time[:, np.newaxis, np.newaxis], *parameters)
    param_values = {param: values for param, values in zip(parameters, trace.T)}
    lc = flux.eval(param_values)
    lc = np.moveaxis(lc, 0, 2)
    return time, lc


def luminosity_model(row, rand_num, trace_path='.', trace_version='2'):
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
    trace_version : str, optional
        Version of the trace to use, i.e., the character before the filter in the filename. Default: '2'.

    Returns
    -------
    lst_rand_lc : list
        2-D list containing a random number of light curves for each filter.

    """
    try:
        trace = load_trace(row['filename'], trace_path=trace_path, version=trace_version)
    except ValueError:
        return np.tile(np.nan, (rand_num, 4, 200))
    i_rand = np.random.randint(trace.shape[1], size=rand_num)
    _, lc = produce_lc(trace[:, i_rand])
    if 'redshift' in row.colnames and not np.ma.is_masked(row['redshift']):
        z = row['redshift']
    else:
        z = row['hostz']
    A_coeffs = extinction.ccm89(effective_wavelengths, row['A_V'], 3.1)
    lc *= 10. ** (A_coeffs[:, np.newaxis] / 2.5) * cosmo_P.luminosity_distance(z).value ** 2. * (1. + z)
    return lc


def absolute_magnitude(row):
    """
    Calculate the peak absolute magnitudes for a light curve in each filter.

    Parameters
    ----------
    row : astropy.table.row.Row
        Astropy table row for a given transient, containing columns 'filename', 'A_V', and 'redshift'/'hostz'

    Returns
    -------
    M : numpy.array
        The peak absolute magnitudes of the light curve.

    """
    min_m = []
    t = light_curve_event_data(row['filename'])
    for fltr in 'griz':
        obs = t[t['FLT'] == fltr]
        if len(obs):
            min_m.append(obs['MAG'].min())
        else:
            min_m.append(np.nan)
    min_m = np.array(min_m)
    if 'redshift' in row.colnames and not np.ma.is_masked(row['redshift']):
        z = row['redshift']
    else:
        z = row['hostz']
    A = extinction.ccm89(effective_wavelengths, row['A_V'], 3.1)
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
    peakmags = np.concatenate([np.tile(absolute_magnitude(row), (ndraws, 1)) for row in t])
    models = np.concatenate([luminosity_model(row, ndraws, trace_path=trace_path) for row in t])
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
