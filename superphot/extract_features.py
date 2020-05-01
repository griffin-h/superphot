#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
import os
import argparse
import logging
from astropy.table import Table, hstack
from astropy.cosmology import Planck15 as cosmo
from sklearn.decomposition import PCA
from tqdm import trange
from .util import read_light_curve, filter_colors, meta_columns, select_labeled_events
from .util import has_labeled_events, subplots_layout
from .fit_model import produce_lc
import pickle

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# using WavelengthMean from the SVO Filter Profile Service http://svo2.cab.inta-csic.es/theory/fps/
R_FILTERS = {'g': 3.57585511, 'r': 2.54033913, 'i': 1.88284171, 'z': 1.49033933, 'y': 1.24431944,  # Pan-STARRS filters
             'U': 4.78442941, 'B': 4.05870021, 'V': 3.02182672, 'R': 2.34507832, 'I': 1.69396924}  # Bessell filters
PARAMNAMES = ['Amplitude', 'Plateau Slope', 'Plateau Duration', 'Start Time', 'Rise Time', 'Fall Time']


def load_trace(file, filters, trace_path='.', version='2'):
    """
    Read the stored PyMC3 traces into a 3-D array with shape (nsteps, nfilters, nparams).

    Assumes the traces are stored in a directory `basename_vf`, where `basename` is `file` up to the first '.', `v` is
    `version`, and `f` is an element of `filters`.

    Parameters
    ----------
    file : str
        Filename of the original SNANA data file.
    filters : iterable
        Filters for which to load traces. If one or more filters are not found, the posteriors of the remaining filters
        will be combined and used in place of the missing ones.
    trace_path : str, optional
        Directory where the PyMC3 trace data is stored. Default: current directory.
    version : str, optional
        Version of the fit to use, where "version" is the character in the filename before the filter. Default: '2'.

    Returns
    -------
    trace_values : numpy.array
        PyMC3 trace stored as 3-D array with shape (nsteps, nfilters, nparams).
    """
    basename = os.path.basename(file).split('.')[0]
    tracefile = os.path.join(trace_path, basename) + '_{}{}'
    trace_values = []
    missing_filters = []
    for fltr in filters:
        tracefile_filter = tracefile.format(version, fltr)
        if os.path.exists(tracefile_filter):
            trace = []
            for chain in glob(os.path.join(tracefile_filter, '*/samples.npz')):
                chain_dict = np.load(chain)
                trace.append([chain_dict[var] for var in PARAMNAMES])
            trace_values.append(np.hstack(trace))
        else:
            logging.warning(f"No such file or directory: '{tracefile_filter}'")
            missing_filters.append(fltr)
    if len(missing_filters) == len(filters):
        raise FileNotFoundError(f"No traces found for {basename}")
    for fltr in missing_filters:
        trace_values.insert(filters.index(fltr), np.mean(trace_values, axis=0))
    trace_values = np.moveaxis(trace_values, 2, 0)
    return trace_values


def flux_to_luminosity(row, R_filter):
    """
    Return the flux-to-luminosity conversion factor for the transient in a given row of a data table.

    Parameters
    ----------
    row : astropy.table.row.Row
        Astropy table row for a given transient, containing columns 'MWEBV' and 'redshift'.
    R_filter : list
        Ratios of A_filter to `row['MWEBV']` for each of the filters used. This determines the length of the output.

    Returns
    -------
    flux2lum : numpy.ndarray
        Array of flux-to-luminosity conversion factors for each filter.
    """
    A_coeffs = row['MWEBV'] * np.array(R_filter)
    dist = cosmo.luminosity_distance(row['redshift']).to('dapc').value
    flux2lum = 10. ** (A_coeffs / 2.5) * 4. * np.pi * dist ** 2. * (1. + row['redshift'])
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


def plot_histograms(data_table, colname, class_kwd='type', varnames=(), rownames=(), no_autoscale=(), saveto=None):
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
        Labels for the leftmost y-axes.
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
    fig, axarr = plt.subplots(nrows, ncols, sharex='col', squeeze=False)
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


def plot_principal_components(pcas, time=None, filters=None, saveto='principal_components.pdf'):
    """
    Plot the principal components being used to extract features from the model light curves.

    Parameters
    ----------
    pcas : list
        List of the PCA objects for each filter, after fitting.
    time : array-like, optional
        Times (x-values) to plot the principal components against.
    filters : iterable, optional
        Names of the filters corresponding to the PCA objects. Only used for coloring and labeling the lines.
    saveto : str, optional
        Filename to which to save the plot. Default: principal_components.pdf.
    """
    nrows, ncols = subplots_layout(pcas[0].n_components)
    fig, axes = plt.subplots(nrows, ncols, sharex=True, squeeze=False)
    if time is None:
        time = np.arange(pcas[0].n_features_)
    else:
        for ax in axes[-1]:
            ax.set_xlabel('Phase')
    lines = []
    if filters is None:
        filters = [f'Filter {i+1:d}' for i in range(len(pcas))]
    for pca, fltr in zip(pcas, filters):
        for pc, ax in zip(pca.components_, axes.flat):
            p = ax.plot(time, pc, color=filter_colors.get(fltr), label=fltr)
        lines += p
    fig.legend(lines, filters, ncol=len(filters), loc='upper center')
    fig.tight_layout(h_pad=0., w_pad=0., rect=(0., 0., 1., 0.95))
    fig.savefig(saveto)


def plot_pca_reconstruction(models, reconstructed, coefficients=None, filters=None, saveto='pca_reconstruction.pdf'):
    """
    Plot comparisons between the model light curves and the light curves reconstructed from the PCA for each transient.
    These are saved as a multipage PDF.

    Parameters
    ----------
    models : array-like
        A 3-D array of model light curves with shape (ntransients, nfilters, ntimes)
    reconstructed : array-like
        A 3-D array of reconstructed light curves with shape (ntransients, nfilters, ntimes)
    coefficients : array-like, optional
        A 3-D array of the principal component coefficients with shape (ntransients, nfilters, ncomponents). If given,
        the coefficients will be printed at the top right of each plot.
    filters : iterable, optional
        Names of the filters corresponding to the PCA objects. Only used for coloring the lines.
    saveto : str, optional
        Filename for the output file. Default: pca_reconstruction.pdf.
    """
    if filters is None:
        filters = [f'Filter {i+1:d}' for i in range(models.shape[1])]
    with PdfPages(saveto) as pdf:
        ax = plt.axes()
        for i in trange(models.shape[0], desc='PCA reconstruction'):
            for j in range(models.shape[1]):
                c = filter_colors.get(filters[j])
                ax.plot(models[i, j], color=c)
                ax.plot(reconstructed[i, j], ls=':', color=c)
            if coefficients is not None:
                with np.printoptions(precision=2):
                    ax.text(0.99, 0.99, str(coefficients[i]), va='top', ha='right', transform=ax.transAxes)
            pdf.savefig()
            ax.clear()


def extract_features(t, stored_models, filters, R_filters=None, ndraws=10, zero_point=27.5, use_pca=True,
                     stored_pcas=None, save_pca_to=None, save_reconstruction_to=None, random_state=None):
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
    filters : iterable
        Filters for which to extract features. If `stored_models` is a directory, these should be the last characters
        of the subdirectories in which the traces are stored. Ignored if models are read from a Numpy file.
    R_filters : dict, optional
        Ratios of total to selective extinction for `filters`. This package includes the values for common filters
        (see `superphot.extract_features.R_FILTERS`). Use this argument to override those default values or to include
        additional filters.
    ndraws : int, optional
        Number of random draws from the MCMC posterior. Default: 10. Ignored if models are read from a Numpy file.
    zero_point : float, optional
        Zero point to be used for calculating the peak absolute magnitudes. Default: 27.5 mag.
    use_pca : bool, optional
        Use the peak absolute magnitudes and principal components of the light curve as the features (default).
        Otherwise, use the model parameters directly.
    stored_pcas : str, optional
        Path to pickled PCA objects. Default: create and fit new PCA objects.
    save_pca_to : str, optional
        Plot and save the principal components to this file. Default: skip this step.
    save_reconstruction_to : str, optional
        Plot and save the reconstructed light curves to this file (slow). Default: skip this step.
    random_state : int, optional
        Seed for the random number generator, which is used to sample the posterior. Use for reproducibility.

    Returns
    -------
    t_good : astropy.table.Table
        Slice of the input table with a 'features' column added. Rows with any bad features are excluded.
    """
    if os.path.isdir(stored_models):
        stored = {}
    else:
        stored = np.load(stored_models)
        filters = stored.get('filters', filters)
        ndraws = stored.get('ndraws', ndraws)

    R_filter = []
    for fltr in filters:
        if R_filters is not None and fltr in R_filters:
            R_filter.append(R_filter[fltr])
        elif fltr in R_FILTERS:
            R_filter.append(R_FILTERS[fltr])
        else:
            raise ValueError(f'Unrecognized filter {fltr}. Please specify the extinction correction using `R_filters`.')

    if 'params' in stored:
        params = stored['params']
        logging.info(f'parameters read from {stored_models}')
    else:
        params = []
        bad_rows = []
        for i, filename in enumerate(t['filename']):
            try:
                trace = load_trace(filename, filters, trace_path=stored_models)
                logging.info(f'loaded trace from {filename}')
            except FileNotFoundError as e:
                bad_rows.append(i)
                logging.error(e)
                continue
            if ndraws:
                rng = np.random.default_rng(random_state)
                params.append(rng.choice(trace, ndraws))
            else:  # ndraws == 0 means take the average
                params.append(trace.mean(axis=0)[np.newaxis])
                ndraws = 1
        params = np.vstack(params)
        t.remove_rows(bad_rows)  # excluding rows that have not been fit
        np.savez_compressed('params.npz', params=params, filters=list(filters), ndraws=ndraws)
        logging.info(f'posteriors sampled from {stored_models}, saved to params.npz')

    t = t[np.repeat(range(len(t)), ndraws)]
    t.meta['filters'] = list(filters)
    t.meta['ndraws'] = ndraws
    t['params'] = params
    params[:, :, 0] *= np.vstack([flux_to_luminosity(row, R_filter) for row in t])
    if use_pca:
        time = np.linspace(0., 300., 1000)
        models = produce_lc(time, params, align_to_t0=True)
        t_good, good_models = select_good_events(t, models)
        peakmags = zero_point - 2.5 * np.log10(good_models.max(axis=2))
        logging.info('peak magnitudes extracted')
        if has_labeled_events(t_good):
            models_to_fit = good_models[~t_good.mask['type']]
        else:
            models_to_fit = good_models
        coefficients, reconstructed, pcas = get_principal_components(good_models, models_to_fit,
                                                                     stored_pcas=stored_pcas)
        if save_pca_to is not None:
            plot_principal_components(pcas, time, filters, save_pca_to)
        logging.info('PCA finished')
        if save_reconstruction_to is not None:
            plot_pca_reconstruction(models_to_fit, reconstructed, coefficients, filters, save_reconstruction_to)
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
    np.savez_compressed(f'{basename}.npz', params=t['params'], features=t['features'], filters=t.meta['filters'],
                        ndraws=t.meta['ndraws'], paramnames=PARAMNAMES, featnames=t.meta['featnames'])
    logging.info(f'data saved to {basename}.txt and {basename}.npz')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_table', type=str, help='List of input SNANA files, or input data table')
    parser.add_argument('stored_models', help='Directory where the PyMC3 trace data is stored, '
                                              'or Numpy file containing stored model parameters/LCs')
    parser.add_argument('--filters', type=str, default='griz', help='Filters from which to extract features')
    parser.add_argument('--ndraws', type=int, default=10, help='Number of draws from the LC posterior for training set.'
                                                               ' Set to 0 to use the mean of the LC parameters.')
    parser.add_argument('--pcas', help='Path to pickled PCA objects. Default: create and fit new PCA objects.')
    parser.add_argument('--use-params', action='store_false', dest='use_pca', help='Use model parameters as features')
    parser.add_argument('--reconstruct', action='store_true',
                        help='Plot and save the reconstructed light curves to {output}_reconstruction.pdf (slow)')
    parser.add_argument('--random-state', type=int, help='Seed for the random number generator (for reproducibility).')
    parser.add_argument('--output', default='test_data',
                        help='Filename (without extension) to save the test data and features')
    args = parser.parse_args()

    logging.info('started extract_features.py')
    data_table = compile_data_table(args.input_table)
    if args.reconstruct:
        save_reconstruction_to = args.output + '_reconstruction.pdf'
    else:
        save_reconstruction_to = None
    test_data = extract_features(data_table, args.stored_models, args.filters, ndraws=args.ndraws, use_pca=args.use_pca,
                                 stored_pcas=args.pcas, save_pca_to=args.output + '_pca.pdf',
                                 save_reconstruction_to=save_reconstruction_to, random_state=args.random_state)
    save_data(test_data, args.output)
    if has_labeled_events(test_data):
        plot_histograms(test_data, 'params', varnames=PARAMNAMES, rownames=args.filters,
                        saveto=args.output + '_parameters.pdf')
        plot_histograms(test_data, 'features', varnames=test_data.meta['featnames'], rownames=args.filters,
                        no_autoscale=['SLSN', 'SNIIn'] if args.use_pca else [], saveto=args.output + '_features.pdf')
    logging.info('finished extract_features.py')
