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
import logging
from astropy.table import Table, join
from astropy.cosmology import Planck15 as cosmo_P
from sklearn.decomposition import PCA
from util import read_snana, light_curve_event_data
from fit_model import setup_model, flux_model

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


def sample_lcs(filename, rand_num, trace_path='.', trace_version='2'):
    """
    Produce a random number of light curves created using the parameters from the stored MCMC traces.

    Parameters
    ----------
    filename : str
        Filename of the original SNANA data file.
    rand_num : int
        The number of light curves randomly extracted from the MCMC run.
    trace_path : str, optional
        Directory where the PyMC3 trace data is stored. Default: current directory.
    trace_version : str, optional
        Version of the trace to use, i.e., the character before the filter in the filename. Default: '2'.

    Returns
    -------
    lc : numpy.ndarray, shape=(rand_num, 4, 200)
        3-D array containing a random number of light curves for each filter.

    """
    try:
        trace = load_trace(filename, trace_path=trace_path, version=trace_version)
    except ValueError:
        return np.tile(np.nan, (rand_num, 4, 200))
    i_rand = np.random.randint(trace.shape[1], size=rand_num)
    _, lc = produce_lc(trace[:, i_rand])
    return lc


def flux_to_luminosity(row):
    """
    Return the flux-to-luminosity conversion factor for the transient in a given row of a data table.

    Parameters
    ----------
    row : astropy.table.row.Row
        Astropy table row for a given transient, containing columns 'A_V', and 'redshift'/'hostz'.

    Returns
    -------
    flux2lum : numpy.ndarray, shape=(4, 1)
        Array of flux-to-luminosity conversion factors for the filters g, r, i, and z.
    """
    if 'redshift' in row.colnames and not np.ma.is_masked(row['redshift']):
        z = row['redshift']
    else:
        z = row['hostz']
    A_coeffs = extinction.ccm89(effective_wavelengths, row['A_V'], 3.1)
    flux2lum = 10. ** (A_coeffs[:, np.newaxis] / 2.5) * cosmo_P.luminosity_distance(z).value ** 2. * (1. + z)
    return flux2lum


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


def extract_features(t, ndraws, trace_path='.', use_stored=False):
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
    use_stored : bool, optional
        Use the peak magnitudes and model LCs stored in model_lcs.npz instead of calculating new ones.

    Returns
    -------
    t_good : astropy.table.Table
        Slice of the input table with a 'features' column added. Rows with any bad features are excluded.
    """
    if use_stored:
        stored = np.load('model_lcs.npz')
        peakmags = stored['peakmags']
        models = stored['models']
    else:
        peakmags = np.concatenate([np.tile(absolute_magnitude(row), (ndraws, 1)) for row in t])
        logging.info('peak magnitudes extracted')
        models = np.concatenate([sample_lcs(row['filename'], ndraws, trace_path) * flux_to_luminosity(row)
                                 for row in t])
        logging.info('model LCs produced')
        np.savez_compressed('model_lcs.npz', peakmags=peakmags, models=models)
    good = np.isfinite(peakmags).all(axis=1) & np.isfinite(models).all(axis=(1, 2))
    pcs = get_principal_components(models[good])
    logging.info('PCA finished')
    i_good, = np.where(good.reshape(-1, ndraws).all(axis=1))
    t_good = t[np.repeat(i_good, ndraws)]
    t_good['features'] = np.dstack([peakmags[good], pcs]).reshape(-1, 24)
    return t_good


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


def plot_final_fit(data, trace_path='.'):
    row = data[0]
    flux_to_lum = flux_to_luminosity(row)
    t = light_curve_event_data(row['filename'])

    if 'models' in data.colnames:
        lc1 = lc2 = data['models']
    else:
        lc1 = sample_lcs(row, len(data), trace_path, '1')
        lc2 = sample_lcs(row, len(data), trace_path, '2')
    lc1 = np.moveaxis(lc1, 0, -1)
    lc2 = np.moveaxis(lc2, 0, -1)

    colors = ['#00CCFF', '#FF7D00', '#90002C', '#000000']
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    time = np.arange(-50., 150.)
    for ax, fltr, color, lc_filt1, lc_filt2, flux2lum in zip(axes.flatten(), 'griz', colors, lc1, lc2, flux_to_lum):
        obs = t[t['FLT'] == fltr]
        ax.errorbar(obs['PHASE'], obs['FLUXCAL'], obs['FLUXCALERR'], fmt='o', color=color)
        ax.text(0.95, 0.95, fltr, transform=ax.transAxes, ha='right', va='top')
        if lc1 is not lc2:
            ax.plot(time, lc_filt1, color=color, ls=':', alpha=0.2)
        ax.plot(time, lc_filt2, color=color, alpha=0.2)

    # autoscale y-axis without errorbars
    ymin = min(t['FLUXCAL'].min(), lc2.min())
    ymax = max(t['FLUXCAL'].max(), lc2.max())
    height = ymax - ymin
    axes[0, 0].set_ylim(ymin - 0.08 * height, ymax + 0.08 * height)

    for i in range(2):
        axes[1, i].set_xlabel('Phase')
        axes[i, 0].set_ylabel('Flux')
    if np.ma.is_masked(row['type']):
        title = row['id']
    else:
        title = f'{row["id"]} = {row["type"]} ($z={row["redshift"]:.3f}$)'
    fig.text(0.5, 0.95, title, ha='center', va='bottom', size='large')
    plt.tight_layout(w_pad=0, h_pad=0, rect=(0, 0, 1, 0.95))
    return fig


def save_test_data(test_table):
    save_table = test_table[['id', 'A_V', 'hostz', 'filename', 'redshift', 'err', 'type']]
    save_table.write('test_data.txt', format='ascii.fixed_width', overwrite=True)
    np.savez_compressed('test_data.npz', features=test_table['features'])
    logging.info('test data saved to test_data.txt and test_data.npz')


def compile_data_table(filenames):
    t_input = meta_table(filenames)
    new_ps1z = Table.read('new_ps1z.dat', format='ascii')  # redshifts of 521 classified SNe
    t_conf = Table.read('ps1confirmed_only_sne_without_outlier.txt', format='ascii')  # classifications of 499 SNe
    bad_lcs = Table.read('bad_lcs.dat', names=['idnum', 'flag0', 'flag1'], format='ascii', fill_values=('-', '0'))
    bad_lcs['id'] = ['PSc{:0>6d}'.format(idnum) for idnum in bad_lcs['idnum']]  # 1227 VAR, AGN, QSO transients
    bad_lcs.remove_column('idnum')
    bad_lcs_2 = np.loadtxt('bad_lcs_2.dat', dtype=str, usecols=[0, -1])  # 526 transients with bad host galaxy spectra
    bad_lcs_2 = Table([['PSc' + idnum for idnum in bad_lcs_2[:, 0]], bad_lcs_2[:, 1]], names=['id', 'flag2'])

    t_final = join(t_input, new_ps1z, join_type='left')
    t_final = join(t_final, t_conf, join_type='left')
    t_final = join(t_final, bad_lcs, join_type='left')
    t_final = join(t_final, bad_lcs_2, join_type='left')

    t_final = t_final[t_final['flag0'].mask & t_final['flag1'].mask & t_final['flag2'].mask & ~t_final['hostz'].mask]
    return t_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', type=str, help='Input SNANA files')
    parser.add_argument('--ndraws', type=int, default=4, help='Number of draws from the LC posterior for training set')
    parser.add_argument('--trace-path', type=str, default='.', help='Directory where the PyMC3 trace data is stored')
    parser.add_argument('--use-stored-models', action='store_true', help='Use stored model LCs in model_lcs.npz')
    args = parser.parse_args()

    logging.info('started extract_features.py')
    data_table = compile_data_table(args.filenames)
    test_data = extract_features(data_table, args.ndraws, trace_path=args.trace_path, use_stored=args.use_stored_models)
    save_test_data(test_data)
    logging.info('finished extract_features.py')
