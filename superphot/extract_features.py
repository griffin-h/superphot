#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pymc3 as pm
import os
import argparse
import logging
from astropy.table import Table, hstack
from astropy.cosmology import Planck15 as cosmo
from sklearn.decomposition import PCA
from tqdm import trange
from .util import read_light_curve, select_event_data, filter_colors, meta_columns, select_labeled_events
from .fit_model import setup_model1, produce_lc, sample_posterior
import pickle

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def load_trace(file, trace_path='.', version='2'):
    """
    Read the stored PyMC3 traces into a 3-D array with shape (nsteps, nfilters, nparams).

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
    trace_values : numpy.array
        PyMC3 trace stored as 3-D array with shape (nsteps, nfilters, nparams).
    """
    basename = os.path.basename(file)
    tracefile = os.path.join(trace_path, basename.replace('.snana.dat', '_{}{}'))
    trace_values = []
    t_full = read_light_curve(file)
    t = select_event_data(t_full)
    max_flux = t['FLUXCAL'].max()
    missing_filters = []
    for fltr in 'griz':
        tracefile_filter = tracefile.format(version, fltr)
        if os.path.exists(tracefile_filter):
            obs = t[t['FLT'] == fltr]
            model, varnames = setup_model1(obs, max_flux)
            trace = pm.load_trace(tracefile_filter, model)
            trace_values.append([trace.get_values(var) for var in varnames])
        else:
            logging.warning(f"No such file or directory: '{tracefile_filter}'")
            missing_filters.append(fltr)
    if len(missing_filters) == 4:
        raise FileNotFoundError(f"No traces found for {basename}")
    for fltr in missing_filters:
        trace_values.insert('griz'.index(fltr), np.mean(trace_values, axis=0))
    trace_values = np.moveaxis(trace_values, 2, 0)
    return trace_values


def flux_to_luminosity(row, R_V=3.1):
    """
    Return the flux-to-luminosity conversion factor for the transient in a given row of a data table.

    Parameters
    ----------
    row : astropy.table.row.Row
        Astropy table row for a given transient, containing columns 'MWEBV' and 'redshift'.
    R_V : float
        Ratio of total to selective extinction, i.e., A_V = row['MWEBV'] * R_V. Default: 3.1

    Returns
    -------
    flux2lum : numpy.ndarray
        Array of flux-to-luminosity conversion factors for the filters g, r, i, and z.
    """
    A_coeffs = row['MWEBV'] * R_V * np.array([1.16269427, 0.87191851, 0.66551667, 0.42906714])  # g, r, i, z
    dist = cosmo.luminosity_distance(row['redshift']).to('dapc').value
    flux2lum = 10. ** (A_coeffs / 2.5) * dist ** 2. * (1. + row['redshift'])
    return flux2lum


def get_principal_components(light_curves, light_curves_fit=None, n_components=6, whiten=True, stored_pcas=None):
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
    whiten : bool, optional
        Whiten the input data before calculating the principal components. Default: True.
    stored_pcas : str, optional
        Path to pickled PCA objects. Default: create and fit new PCA objects.

    Returns
    -------
    principal_components : array-like
        A list of the principal components for each of the input light curves.
    """
    filt_range = range(light_curves.shape[1])
    if light_curves_fit is None:
        light_curves_fit = light_curves

    if stored_pcas is None:
        pcas = [PCA(n_components, whiten=whiten) for _ in filt_range]
    else:
        with open(stored_pcas, 'rb') as f:
            pcas = pickle.load(f)
    reconstructed = np.empty_like(light_curves_fit)
    coefficients = np.empty(light_curves.shape[:-1] + (n_components,))

    for i in filt_range:
        if stored_pcas is None:
            pcas[i].fit(light_curves_fit[:, i])
        coeffs_fit = pcas[i].transform(light_curves_fit[:, i])
        reconstructed[:, i] = pcas[i].inverse_transform(coeffs_fit)
        coefficients[:, i] = pcas[i].transform(light_curves[:, i])

    if stored_pcas is None:
        with open('pca.pickle', 'wb') as f:
            pickle.dump(pcas, f)

    return coefficients, reconstructed, pcas


def plot_histograms(data_table, colname, class_kwd='type', varnames=(), rownames='griz', no_autoscale=(), saveto=None):
    """
    Plot a grid of histograms of the column `colname` of `data_table`, grouped by the column `groupby`.

    Parameters
    ----------
    data_table : astropy.table.Table
        Data table containing the columns `colname` and `groupby` for each supernova.
    colname : str
        Column name of `data_table` to plot (e.g., 'params' or 'features').
    class_kwd : str, optional
        Column name of `data_table` to group by before plotting (e.g., 'type' or 'prediction'). Default: 'type'.
    varnames : iterable, optional
        Parameter names to list on the x-axes of the plot. Default: no labels.
    rownames : iterable, optional
        Labels for the leftmost y-axes. Default: 'g', 'r', 'i', 'z'.
    no_autoscale : tuple or list, optional
        Class names not to use in calculating the axis limits. Default: include all.
    saveto : str, optional
        Filename to which to save the plot. Default: display the plot instead of saving it.
    """
    _, nrows, ncols = data_table[colname].shape
    if class_kwd:
        data_table = select_labeled_events(data_table, key=class_kwd).group_by(class_kwd)
        data_table.groups.keys['patch'] = None
    else:
        data_table = data_table.group_by(np.ones(len(data_table)))
    ngroups = len(data_table.groups)
    fig, axarr = plt.subplots(nrows, ncols, sharex='col')
    for j in range(ncols):
        xlims = []
        for i in range(nrows):
            ylims = []
            for k in range(ngroups):
                histdata = data_table.groups[k][colname][:, i, j]
                histrange = np.percentile(histdata, (5., 95.))
                n, b, p = axarr[i, j].hist(histdata, range=histrange, density=True, histtype='step')
                if class_kwd:
                    data_table.groups.keys['patch'][k] = p[0]
                if not class_kwd or data_table.groups.keys[class_kwd][k] not in no_autoscale:
                    xlims.append(b)
                    ylims.append(n)
            axarr[i, j].set_ylim(0., 1.05 * np.max(ylims))
            axarr[i, j].set_yticks([])
        xmin, xmax = np.percentile(xlims, (0., 100.))
        axarr[-1, j].set_xlim(1.05 * xmin - 0.05 * xmax, 1.05 * xmax - 0.05 * xmin)
        axarr[-1, j].xaxis.set_major_locator(plt.MaxNLocator(2))
    if class_kwd:
        fig.legend(data_table.groups.keys['patch'], data_table.groups.keys[class_kwd], loc='upper center', ncol=ngroups,
                   title={'type': 'Spectroscopic Class', 'prediction': 'Photometric Class'}.get(class_kwd, class_kwd))
    for ax, var in zip(axarr[-1], varnames):
        ax.set_xlabel(var, size='small')
        ax.tick_params(labelsize='small')
        if 'mag' in var.lower():
            ax.invert_xaxis()
    for ax, filt in zip(axarr[:, 0], rownames):
        ax.set_ylabel(filt, rotation=0, va='center')
    fig.tight_layout(h_pad=0., w_pad=0., rect=(0., 0., 1., 0.9 if class_kwd else 1.))
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
    plt.close(fig)


def plot_principal_components(pcas, time=None):
    """
    Plot the principal components being used to extract features from the model light curves. The plot will be saved to
    principal_components.pdf.

    Parameters
    ----------
    pcas : list
        List of the PCA objects for each filter, after fitting.
    time : array-like, optional
        Times (x-values) to plot the principal components against.
    """
    nrows = int(pcas[0].n_components ** 0.5)
    ncols = int(np.ceil(pcas[0].n_components / nrows))
    fig, axes = plt.subplots(nrows, ncols, sharex=True)
    if time is None:
        time = np.arange(pcas[0].n_features_)
    else:
        for ax in axes[-1]:
            ax.set_xlabel('Phase')
    lines = []
    for pca, fltr in zip(pcas, 'griz'):
        for pc, ax in zip(pca.components_, axes.flatten()):
            p = ax.plot(time, pc, color=filter_colors[fltr], label=fltr)
        lines += p
    fig.legend(lines, 'griz', ncol=4, loc='upper center')
    fig.tight_layout(h_pad=0., w_pad=0., rect=(0., 0., 1., 0.95))
    fig.savefig('principal_components.pdf')


def plot_pca_reconstruction(models, reconstructed, coefficients=None):
    """
    Plot comparisons between the model light curves and the light curves reconstructed from the PCA for each transient.
    These are saved as a multi-page PDF called pdf_reconstruction.pdf.

    Parameters
    ----------
    models : array-like
        A 3-D array of model light curves with shape (ntransients, nfilters, ntimes)
    reconstructed : array-like
        A 3-D array of reconstructed light curves with shape (ntransients, nfilters, ntimes)
    coefficients : array-like, optional
        A 3-D array of the principal component coefficients with shape (ntransients, nfilters, ncomponents). If given,
        the coefficients will be printed at the top right of each plot.
    """
    with PdfPages('pca_reconstruction.pdf') as pdf:
        ax = plt.axes()
        for i in trange(models.shape[1], desc='PCA reconstruction'):
            for j, fltr in enumerate('griz'):
                c = filter_colors[fltr]
                ax.plot(models[j, i], color=c)
                ax.plot(reconstructed[j, i], ls=':', color=c)
            if coefficients is not None:
                with np.printoptions(precision=2):
                    ax.text(0.99, 0.99, str(coefficients[:, i]), va='top', ha='right', transform=ax.transAxes)
            pdf.savefig()
            ax.clear()


def extract_features(t, stored_models, ndraws=10, zero_point=27.5, use_pca=True, reconstruct=False, stored_pcas=None):
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
    reconstruct : bool, optional
        Plot and save the reconstructed light curves to pca_reconstruction.pdf (slow). Default: False.
    stored_pcas : str, optional
        Path to pickled PCA objects. Default: create and fit new PCA objects.

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
                logging.info(f'loaded trace from {filename}')
            except FileNotFoundError as e:
                bad_rows.append(i)
                logging.error(e)
                continue
            if ndraws:
                params.append(sample_posterior(trace, ndraws))
            else:  # ndraws == 0 means take the average
                params.append(trace.mean(axis=0)[np.newaxis])
                ndraws = 1
        params = np.vstack(params)
        t.remove_rows(bad_rows)  # excluding rows that have not been fit
        np.savez_compressed('params.npz', params=params, ndraws=ndraws)
        logging.info(f'posteriors sampled from {stored_models}, saved to params.npz')

    t = t[np.repeat(range(len(t)), ndraws)]
    t.meta['ndraws'] = ndraws
    t['params'] = params
    t.meta['paramnames'] = ['Amplitude', 'Plateau Slope', 'Plateau Duration', 'Start Time', 'Rise Time', 'Fall Time']
    params[:, :, 0] *= np.vstack([flux_to_luminosity(row) for row in t])
    if use_pca:
        time = np.linspace(0., 300., 1000)
        models = produce_lc(time, params, align_to_t0=True)
        t_good, good_models = select_good_events(t, models)
        peakmags = zero_point - 2.5 * np.log10(good_models.max(axis=2))
        logging.info('peak magnitudes extracted')
        if t_good.has_masked_values and 'type' in t_good.colnames:
            models_to_fit = good_models[~t_good.mask['type']]
        else:
            models_to_fit = good_models
        coefficients, reconstructed, pcas = get_principal_components(good_models, models_to_fit,
                                                                     stored_pcas=stored_pcas)
        plot_principal_components(pcas, time)
        logging.info('PCA finished')
        if reconstruct:
            plot_pca_reconstruction(good_models, reconstructed, coefficients)
        features = np.dstack([peakmags, coefficients])
        t_good.meta['featnames'] = ['Peak Abs. Mag.'] + [f'PC{i:d} Proj.' for i in range(1, 7)]
    else:
        params[:, :, 0] = zero_point - 2.5 * np.log10(params[:, :, 0])  # convert amplitude to magnitude
        t_good, features = select_good_events(t, params[:, :, [0, 1, 2, 4, 5]])  # remove start time from features
        t_good.meta['featnames'] = ['Amplitude (mag)', 'Plateau Slope', 'Plateau Duration', 'Rise Time', 'Fall Time']
    t_good['features'] = features
    return t_good


def select_good_events(t, data):
    """
    Select only events with finite data for all draws. Returns the table and data for only these events.

    Parameters
    ----------
    t : astropy.table.Table
        Original data table. Must have `t.meta['ndraws']` to indicate now many draws it contains for each event.
    data : array-like, shape=(nfilt, len(t), ...)
        Numpy array containing the data upon which finiteness will be judged.

    Returns
    -------
    t_good : astropy.table.Table
        Data table containing only the good events.
    good_data : array-like
        Numpy array containing only the data for good events.
    """
    finite_values = np.isfinite(data)
    finite_draws = finite_values.all(axis=(1, 2))
    events_with_finite_draws = finite_draws.reshape(-1, t.meta['ndraws'])
    finite_events = events_with_finite_draws.all(axis=1)
    draws_from_finite_events = np.repeat(finite_events, t.meta['ndraws'])
    t_good = t[draws_from_finite_events]
    good_data = data[draws_from_finite_events]
    return t_good, good_data


def compile_data_table(filename):
    t_input = Table.read(filename, format='ascii')
    required_cols = ['MWEBV', 'redshift']
    missing_cols = [col for col in required_cols if col not in t_input.colnames]
    if missing_cols:
        t_meta = Table(names=required_cols, dtype=[float, float])
        for lc_file in t_input['filename']:
            t = read_light_curve(lc_file)
            t_meta.add_row([t.meta[col.upper()] for col in required_cols])
        t_final = hstack([t_input, t_meta[missing_cols]])
    else:
        t_final = t_input
    good_redshift = t_final['redshift'] > 0.
    if not good_redshift.all():
        logging.warning('excluding files with redshifts <= 0')
        t_final[~good_redshift].pprint(max_lines=-1)
    t_final = t_final[good_redshift]
    t_final['MWEBV'].format = '%.4f'
    t_final['redshift'].format = '%.4f'
    return t_final


def save_data(t, basename):
    t.sort('filename')
    save_table = t[[col for col in meta_columns if col in t.colnames]][::t.meta['ndraws']]
    save_table.write(f'{basename}.txt', format='ascii.fixed_width_two_line', overwrite=True)
    np.savez_compressed(f'{basename}.npz', params=t['params'], features=t['features'], ndraws=t.meta['ndraws'],
                        paramnames=t.meta['paramnames'], featnames=t.meta['featnames'])
    logging.info(f'data saved to {basename}.txt and {basename}.npz')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_table', type=str, help='List of input SNANA files, or input data table')
    parser.add_argument('stored_models', help='Directory where the PyMC3 trace data is stored, '
                                              'or Numpy file containing stored model parameters/LCs')
    parser.add_argument('--ndraws', type=int, default=10, help='Number of draws from the LC posterior for training set.'
                                                               ' Set to 0 to use the mean of the LC parameters.')
    parser.add_argument('--pcas', help='Path to pickled PCA objects. Default: create and fit new PCA objects.')
    parser.add_argument('--use-params', action='store_false', dest='use_pca', help='Use model parameters as features')
    parser.add_argument('--reconstruct', action='store_true',
                        help='Plot and save the reconstructed light curves to pca_reconstruction.pdf (slow)')
    parser.add_argument('--output', default='test_data',
                        help='Filename (without extension) to save the test data and features')
    args = parser.parse_args()

    logging.info('started extract_features.py')
    data_table = compile_data_table(args.input_table)
    test_data = extract_features(data_table, args.stored_models, args.ndraws, use_pca=args.use_pca,
                                 reconstruct=args.reconstruct, stored_pcas=args.pcas)
    save_data(test_data, args.output)
    if 'type' in test_data.colnames:
        plot_histograms(test_data, 'params', varnames=test_data.meta['paramnames'],
                        saveto=args.output + '_parameters.pdf')
        plot_histograms(test_data, 'features', varnames=test_data.meta['featnames'],
                        no_autoscale=['SLSN', 'SNIIn'] if args.use_pca else [], saveto=args.output + '_features.pdf')
    logging.info('finished extract_features.py')
