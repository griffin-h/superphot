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


def produce_lc(time, trace, align_to_t0=False):
    """
    Load the stored PyMC3 traces and produce model light curves from the parameters.

    Parameters
    ----------
    time : numpy.array
        Range of times (in days, with respect to PEAKMJD) over which the model should be calculated.
    trace : numpy.array
        PyMC3 trace stored as 3-D array with shape (nfilters, nsteps, nparams).
    align_to_t0 : bool, optional
        Interpret `time` as days with respect to t_0 instead of PEAKMJD.

    Returns
    -------
    lc : numpy.array
        Model light curves. Shape = (len(trace) * nwalkers, nfilters, len(time)).
    """
    tt.config.compute_test_value = 'ignore'
    mytensor = tt.TensorType('float64', (False, False, True))
    parameters = [mytensor() for _ in range(6)]
    flux = flux_model(time, *parameters)
    param_values = {param: values[:, :, np.newaxis] for param, values in zip(parameters, np.moveaxis(trace, 2, 0))}
    if align_to_t0:
        param_values[parameters[3]] = 0.
    lc = flux.eval(param_values)
    return lc


def sample_posterior(filename, rand_num, trace_path='.', trace_version='2'):
    """
    Randomly sample the parameters from the stored MCMC traces.

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
    trace_rand : numpy.ndarray, shape=(4, rand_num, 6)
        3-D array containing a random sampling of parameters for each filter.

    """
    try:
        trace = load_trace(filename, trace_path=trace_path, version=trace_version)
        i_rand = np.random.randint(trace.shape[1], size=rand_num)
        trace_rand = trace[:, i_rand]
    except ValueError:
        trace_rand = np.tile(np.nan, (4, rand_num, 6))
    return trace_rand


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


def get_principal_components(light_curves, use_for_fitting=slice(None), n_components=6, whiten=True):
    """
    Run a principal component analysis on a list of light curves and return a list of their principal components.

    Parameters
    ----------
    light_curves : array-like
        A list of evenly-sampled model light curves.
    use_for_fitting : array-like or slice, optional
        A boolean or index array into light_curves to indicate which samples should be used to fit the PCA.
        Default: use all for fitting.
    n_components : int, optional
        The number of principal components to calculate. Default: 6.
    whiten : bool
        Whiten the input data before calculating the principal components. Default: True.

    Returns
    -------
    principal_components : array-like
        A list of the principal components for each of the input light curves.
    """
    principal_components = []
    pca = PCA(n_components, whiten=whiten)
    for lc_filter in light_curves:
        pca.fit(lc_filter[use_for_fitting])
        princ_comp = pca.transform(lc_filter)
        principal_components.append(princ_comp)
    principal_components = np.array(principal_components)
    return principal_components


def extract_features(t, stored_models, ndraws=10, zero_point=27.5):
    """
    Extract features for a table of model light curves: the peak absolute magnitudes and principal components of the
    light curves in each filter.

    Parameters
    ----------
    t : astropy.table.Table
        Table containing the 'filename' and 'redshift' of each transient to be classified.
    stored_models : str
        If a directory, look in this directory for PyMC3 trace data and sample the posterior to produce model LCs.
        If a Numpy file, read the model LCs (or alternatively, the parameters) from this file.
    ndraws : int, optional
        Number of random draws from the MCMC posterior. Default: 10. Ignored if models are read fron Numpy file.
    zero_point : float, optional
        Zero point to be used for calculating the peak absolute magnitudes. Default: 27.5 mag.

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

    if 'flux' in stored:
        flux = stored['flux']
        logging.info(f'model LCs read from {stored_models}')
    else:
        if 'params' in stored:
            params = stored['params']
            logging.info(f'parameters read from {stored_models}')
        else:
            params = np.hstack([sample_posterior(filename, ndraws, stored_models) for filename in t['filename']])
            logging.info(f'posteriors sampled from {stored_models}')
        time = np.arange(-50., 180.)
        flux = produce_lc(time, params, align_to_t0=True)
        np.savez_compressed('model_lcs.npz', time=time, flux=flux, params=params, ndraws=ndraws)
        logging.info('model LCs produced, saved to model_lcs.npz')

    flux2lum = np.concatenate([np.tile(flux_to_luminosity(row), (ndraws, 1)) for row in t]).T
    models = flux * flux2lum[:, :, np.newaxis]
    good = np.isfinite(models).all(axis=(0, 2))
    peakmags = zero_point - 2.5 * np.log10(models[:, good].max(axis=2))
    logging.info('peak magnitudes extracted')
    pcs = get_principal_components(models[:, good])
    logging.info('PCA finished')
    i_good, = np.where(good.reshape(-1, ndraws).all(axis=1))
    t_good = t[np.repeat(i_good, ndraws)]
    t_good['features'] = np.hstack(np.dstack([peakmags, pcs]))
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


def plot_final_fit(time, data, trace_path='.'):
    row = data[0]
    t = light_curve_event_data(row['filename'])

    if 'models' in data.colnames:
        lc1 = lc2 = np.moveaxis(data['models'], 0, 1)
    else:
        params1 = sample_posterior(row['filename'], len(data), trace_path, '1')
        params2 = sample_posterior(row['filename'], len(data), trace_path, '2')
        time, lc1 = produce_lc(time, params1)
        time, lc2 = produce_lc(time, params2)

    colors = ['#00CCFF', '#FF7D00', '#90002C', '#000000']
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
    for ax, fltr, color, lc_filt1, lc_filt2 in zip(axes.flatten(), 'griz', colors, lc1, lc2):
        obs = t[t['FLT'] == fltr]
        ax.errorbar(obs['PHASE'], obs['FLUXCAL'], obs['FLUXCALERR'], fmt='o', color=color)
        ax.text(0.95, 0.95, fltr, transform=ax.transAxes, ha='right', va='top')
        if lc1 is not lc2:
            ax.plot(time, lc_filt1.T, color=color, ls=':', alpha=0.2)
        ax.plot(time, lc_filt2.T, color=color, alpha=0.2)

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
    save_table = test_table[['id', 'A_V', 'hostz', 'filename', 'redshift', 'err', 'type', 'flag0', 'flag1', 'flag2']]
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

    t_final = t_final[~t_final['hostz'].mask | ~t_final['redshift'].mask]
    return t_final


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filenames', nargs='+', type=str, help='Input SNANA files')
    parser.add_argument('stored_models', help='Directory where the PyMC3 trace data is stored, '
                                              'or Numpy file containing stored model parameters/LCs')
    parser.add_argument('--ndraws', type=int, default=10, help='Number of draws from the LC posterior for training set')
    args = parser.parse_args()

    logging.info('started extract_features.py')
    data_table = compile_data_table(args.filenames)
    test_data = extract_features(data_table, args.stored_models, args.ndraws)
    save_test_data(test_data)
    logging.info('finished extract_features.py')
