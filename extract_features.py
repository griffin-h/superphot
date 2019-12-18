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
import os
import argparse
import logging
from astropy.table import Table, join
from astropy.cosmology import Planck15 as cosmo_P
from sklearn.decomposition import PCA
from util import read_snana, light_curve_event_data, filter_colors
from fit_model import setup_model, produce_lc, sample_posterior

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
effective_wavelengths = np.array([4866., 6215., 7545., 9633.])  # g, r, i, z


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
        tracefile_filter = tracefile.format(version, fltr)
        if not os.path.exists(tracefile_filter):
            raise FileNotFoundError(f"[Errno 2] No such file or directory: '{tracefile_filter}'")
        trace = pm.load_trace(tracefile_filter, model)
        trace_values = np.transpose([trace.get_values(var) for var in varnames])
        lst.append(trace_values)
    lst = np.array(lst)
    return lst


def flux_to_luminosity(row):
    """
    Return the flux-to-luminosity conversion factor for the transient in a given row of a data table.

    Parameters
    ----------
    row : astropy.table.row.Row
        Astropy table row for a given transient, containing columns 'A_V', and 'redshift'/'hostz'.

    Returns
    -------
    flux2lum : numpy.ndarray
        Array of flux-to-luminosity conversion factors for the filters g, r, i, and z.
    """
    if 'redshift' in row.colnames and not np.ma.is_masked(row['redshift']):
        z = row['redshift']
    elif 'hostz' in row.colnames and not np.ma.is_masked(row['hostz']):
        z = row['hostz']
    else:
        z = np.nan
    A_coeffs = extinction.ccm89(effective_wavelengths, row['A_V'], 3.1)
    flux2lum = 10. ** (A_coeffs / 2.5) * cosmo_P.luminosity_distance(z).to('dapc').value ** 2. * (1. + z)
    return flux2lum


def get_principal_components(light_curves, light_curves_fit=None, n_components=6, whiten=True):
    """
    Run a principal component analysis on a list of light curves and return a list of their principal components.

    Parameters
    ----------
    light_curves : array-like
        A list of evenly-sampled model light curves.
    light_curves_fit : array-like, optional
        A list of model light curves to be used for fitting the PCA. Default: fit and transform the same light curves.
    n_components : int, optional
        The number of principal components to calculate. Default: 6.
    whiten : bool
        Whiten the input data before calculating the principal components. Default: True.

    Returns
    -------
    principal_components : array-like
        A list of the principal components for each of the input light curves.
    """
    if light_curves_fit is None:
        light_curves_fit = light_curves

    pcas = []
    reconstructed = []
    coefficients = []

    for lc_filter, lc_filter_fit in zip(light_curves, light_curves_fit):
        pca = PCA(n_components, whiten=whiten)
        pcas.append(pca)

        coeffs_fit = pca.fit_transform(lc_filter_fit)
        reconst = pca.inverse_transform(coeffs_fit)
        reconstructed.append(reconst)

        coeffs = pca.transform(lc_filter)
        coefficients.append(coeffs)

    coefficients = np.array(coefficients)
    reconstructed = np.array(reconstructed)
    plot_principal_components(pcas)

    return coefficients, reconstructed


def plot_principal_components(pcas):
    nrows = int(pcas[0].n_components ** 0.5)
    ncols = int(np.ceil(pcas[0].n_components / nrows))
    fig, axes = plt.subplots(nrows, ncols, sharex=True)
    for pca, fltr in zip(pcas, 'griz'):
        for pc, ax in zip(pca.components_, axes.flatten()):
            ax.plot(pc, color=filter_colors[fltr], label=fltr)
    axes[0, 0].legend()
    fig.tight_layout()
    fig.savefig('principal_components.pdf')


def extract_features(t, stored_models, ndraws=10, zero_point=27.5, use_pca=True):
    """
    Extract features for a table of model light curves: the peak absolute magnitudes and principal components of the
    light curves in each filter.

    Parameters
    ----------
    t : astropy.table.Table
        Table containing the 'filename' and 'redshift' of each transient to be classified.
    stored_models : str
        If a directory, look in this directory for PyMC3 trace data and sample the posterior to produce model LCs.
        If a Numpy file, read the parameters from this file.
    ndraws : int, optional
        Number of random draws from the MCMC posterior. Default: 10. Ignored if models are read fron Numpy file.
    zero_point : float, optional
        Zero point to be used for calculating the peak absolute magnitudes. Default: 27.5 mag.
    use_pca : bool, optional
        Use the peak absolute magnitudes and principal components of the light curve as the features (default).
        Otherwise, use the model parameters directly.

    Returns
    -------
    t_good : astropy.table.Table
        Slice of the input table with a 'features' column added. Rows with any bad features are excluded.
    """
    if os.path.isdir(stored_models):
        stored = {}
    else:
        stored = np.load(stored_models)
        ndraws = stored.get('ndraws', ndraws)

    if 'params' in stored:
        params = stored['params']
        logging.info(f'parameters read from {stored_models}')
    else:
        params = []
        bad_rows = []
        for i, filename in enumerate(t['filename']):
            try:
                trace = load_trace(filename, trace_path=stored_models)
            except FileNotFoundError:
                bad_rows.append(i)
                continue
            params.append(sample_posterior(trace, ndraws))
        params = np.hstack(params)
        t.remove_rows(bad_rows)
        t.write('data_table.txt', format='ascii.fixed_width')  # after removing rows that have not been fit
        np.savez_compressed('params.npz', params=params, ndraws=ndraws)
        logging.info(f'posteriors sampled from {stored_models}, saved to data_table.txt & params.npz')

    flux2lum = np.concatenate([np.tile(flux_to_luminosity(row), (ndraws, 1)) for row in t]).T
    params[:, :, 0] *= flux2lum
    if use_pca:
        time = np.linspace(0., 300., 1000)
        models = produce_lc(time, params, align_to_t0=True)
        good = np.isfinite(models).all(axis=(0, 2))
        peakmags = zero_point - 2.5 * np.log10(models[:, good].max(axis=2))
        logging.info('peak magnitudes extracted')
        coefficients, reconstructed = get_principal_components(models[:, good])
        logging.info('PCA finished')
        features = np.dstack([peakmags, coefficients])
    else:
        good = np.isfinite(params).all(axis=(0, 2))
        features = params[:, good]
    i_good, = np.where(good.reshape(-1, ndraws).all(axis=1))
    t_good = t[np.repeat(i_good, ndraws)]
    t_good['features'] = np.hstack(features)
    return t_good


def meta_table(filenames):
    t_meta = Table(names=['id', 'A_V', 'hostz'], dtype=['S9', float, float], masked=True)
    for filename in filenames:
        t = read_snana(filename)
        t_meta.add_row([t.meta['SNID'], t.meta['A_V'], t.meta['REDSHIFT']])
    t_meta['filename'] = filenames
    t_meta['hostz'].mask = t_meta['hostz'] <= 0.
    t_meta['A_V'].format = '%.5f'
    t_meta['hostz'].format = '%.4f'
    return t_meta


def save_test_data(test_table):
    save_table = test_table[['id', 'A_V', 'hostz', 'filename', 'redshift', 'err', 'type', 'flag0', 'flag1', 'flag2']]
    save_table.write('test_data.txt', format='ascii.fixed_width', overwrite=True)
    np.savez_compressed('test_data.npz', features=test_table['features'])
    logging.info('test data saved to test_data.txt and test_data.npz')


def compile_data_table(filenames):
    t_input = meta_table(filenames)
    new_ps1z = Table.read('new_ps1z.dat', format='ascii')  # redshifts of 524 classified transients
    t_conf = Table.read('ps1confirmed_only_sne.txt', format='ascii')  # classifications of 513 SNe
    bad_lcs = Table.read('bad_lcs.dat', names=['idnum', 'flag0', 'flag1'], format='ascii', fill_values=('-', '0'))
    bad_lcs['id'] = ['PSc{:0>6d}'.format(idnum) for idnum in bad_lcs['idnum']]  # 1227 VAR, AGN, QSO transients
    bad_lcs.remove_column('idnum')
    bad_lcs_2 = np.loadtxt('bad_lcs_2.dat', dtype=str, usecols=[0, -1])  # 526 transients with bad host galaxy spectra
    bad_lcs_2 = Table([['PSc' + idnum for idnum in bad_lcs_2[:, 0]], bad_lcs_2[:, 1]], names=['id', 'flag2'])

    t_final = join(t_input, new_ps1z, join_type='left')
    t_final = join(t_final, t_conf, join_type='left')
    t_final = join(t_final, bad_lcs, join_type='left')
    t_final = join(t_final, bad_lcs_2, join_type='left')

    t_final = t_final[~t_final['hostz'].mask | ~t_final['redshift'].mask]
    return t_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_table', type=str, help='List of input SNANA files, or input data table')
    parser.add_argument('stored_models', help='Directory where the PyMC3 trace data is stored, '
                                              'or Numpy file containing stored model parameters/LCs')
    parser.add_argument('--ndraws', type=int, default=10, help='Number of draws from the LC posterior for training set')
    parser.add_argument('--use-params', action='store_false', dest='use_pca', help='Use model parameters as features')
    args = parser.parse_args()

    logging.info('started extract_features.py')
    data_table = Table.read(args.input_table, format='ascii.fixed_width')
    if 'id' not in data_table.colnames:
        data_table = compile_data_table(data_table['filename'])
    test_data = extract_features(data_table, args.stored_models, args.ndraws, use_pca=args.use_pca)
    save_test_data(test_data)
    logging.info('finished extract_features.py')
