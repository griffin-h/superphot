import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
import os
import argparse
import logging
from astropy.table import Table, hstack, join
from astropy.cosmology import Planck15 as cosmo
from sklearn.decomposition import PCA
from tqdm import trange
from .util import filter_colors, meta_columns, load_data, plot_histograms, subplots_layout
from .fit import read_light_curve, produce_lc
import pickle
from scipy.stats import spearmanr

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# using WavelengthMean from the SVO Filter Profile Service http://svo2.cab.inta-csic.es/theory/fps/
R_FILTERS = {'g': 3.57585511, 'r': 2.54033913, 'i': 1.88284171, 'z': 1.49033933, 'y': 1.24431944,  # Pan-STARRS filters
             'U': 4.78442941, 'B': 4.05870021, 'V': 3.02182672, 'R': 2.34507832, 'I': 1.69396924}  # Bessell filters
PARAMNAMES = ['Amplitude', 'Plateau Slope', 'Plateau Duration', 'Start Time', 'Rise Time', 'Fall Time']


def load_trace(tracefile, filters):
    """
    Read the stored PyMC3 traces into a 3-D array with shape (nsteps, nfilters, nparams).

    Parameters
    ----------
    tracefile : str
        Directory where the traces are stored. Should contain an asterisk (*) to be replaced by elements of `filters`.
    filters : iterable
        Filters for which to load traces. If one or more filters are not found, the posteriors of the remaining filters
        will be combined and used in place of the missing ones.

    Returns
    -------
    trace_values : numpy.array
        PyMC3 trace stored as 3-D array with shape (nsteps, nfilters, nparams).
    """
    trace_values = []
    missing_filters = []
    for fltr in filters:
        tracefile_filter = tracefile.replace('*', fltr)
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
        raise FileNotFoundError(f"No traces found for {tracefile}")
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


def plot_feature_correlation(data_table, saveto=None):
    """
    Plot a matrix of the Spearman rank correlation coefficients between each pair of features.

    Parameters
    ----------
    data_table : astropy.table.Table
        Astropy table containing a 'features' column. Must also have 'featnames' and 'filters' in `data_table.meta`.
    saveto : str, optional
        Filename to which to save the plot. Default: show instead of saving.
    """
    X = data_table['features'].reshape(len(data_table), -1, order='F')
    featnames = data_table.meta['featnames']
    filters = data_table.meta['filters']
    nfeats = len(featnames)
    nfilt = len(filters)
    corr = spearmanr(X).correlation
    fig, ax = plt.subplots(1, 1, figsize=(6., 5.))
    cmap = ax.imshow(np.abs(corr), vmin=0., vmax=1.)
    lines = np.arange(1., nfeats) * nfilt - 0.5
    ax.vlines(lines, *ax.get_ylim(), lw=1)
    ax.hlines(lines, *ax.get_xlim(), lw=1)
    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Spearman Rank Correlation Coefficient $|\\rho|$')
    ticks = np.arange(nfeats * nfilt)
    ticklabels = np.tile(filters, nfeats)
    ax.set_xticks([])
    ax.set_xticks(ticks, minor=True)
    ax.set_xticklabels(ticklabels, size='small', minor=True, va='center', rotation='vertical')
    ax.set_yticks([])
    ax.set_yticks(ticks, minor=True)
    ax.set_yticklabels(ticklabels, size='small', minor=True, ha='center')
    for i, featname in enumerate(data_table.meta['featnames']):
        pos = (i + 0.5) * nfilt - 0.5
        ax.text(-0.05, pos, featname, ha='right', va='center', transform=ax.get_yaxis_transform())
        ax.text(pos, -0.05, featname, ha='center', va='top', transform=ax.get_xaxis_transform(), rotation='vertical')
    fig.tight_layout()
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
    plt.close(fig)


def compile_parameters(t, stored_models, filters, ndraws=10, random_state=None):
    """
    Read the saved PyMC3 traces and compile an array of fit parameters for each transient. Save to a Numpy file.

    Parameters
    ----------
    t : astropy.table.Table
        Table containing the 'filename' of each transient to be classified.
    stored_models : str
        Look in this directory for PyMC3 trace data and sample the posterior to produce model LCs.
    filters : iterable
        Filters for which to compile parameters. These should be the last characters of the subdirectories in which the
        traces are stored.
    ndraws : int, optional
        Number of random draws from the MCMC posterior. Default: 10.
    random_state : int, optional
        Seed for the random number generator, which is used to sample the posterior. Use for reproducibility.
    """
    params = []
    median_params = []
    bad_rows = []
    for i, filename in enumerate(t['filename']):
        try:
            basename = os.path.basename(filename).split('.')[0]
            tracefile = os.path.join(stored_models, basename) + '_2*'
            trace = load_trace(tracefile, filters)
            logging.info(f'loaded trace from {tracefile}')
        except FileNotFoundError as e:
            bad_rows.append(i)
            logging.error(e)
            continue
        rng = np.random.default_rng(random_state)
        params.append(rng.choice(trace, ndraws))
        median_params.append(np.median(trace, axis=0))
    params = np.vstack(params)
    median_params = np.stack(median_params)
    if bad_rows:
        t[bad_rows].write('failed.txt', format='ascii.fixed_width_two_line', overwrite=True)
    t.remove_rows(bad_rows)
    t['median_params'] = median_params
    t = t[np.repeat(range(len(t)), ndraws)]
    t.meta['filters'] = list(filters)
    t.meta['ndraws'] = ndraws
    t.meta['paramnames'] = PARAMNAMES
    t['params'] = params
    return t


def extract_features(t, zero_point=27.5, use_median=False, use_pca=True, stored_pcas=None, save_pca_to=None,
                     save_reconstruction_to=None):
    """
    Extract features for a table of model light curves: the peak absolute magnitudes and principal components of the
    light curves in each filter.

    Parameters
    ----------
    t : astropy.table.Table
        Table containing the 'params'/'median_params', 'redshift', and 'MWEBV' of each transient to be classified.
    zero_point : float, optional
        Zero point to be used for calculating the peak absolute magnitudes. Default: 27.5 mag.
    use_median : bool, optional
        Use the median parameters to produce the light curves instead of the multiple draws from the posterior.
    use_pca : bool, optional
        Use the peak absolute magnitudes and principal components of the light curve as the features (default).
        Otherwise, use the model parameters directly.
    stored_pcas : str, optional
        Path to pickled PCA objects. Default: create and fit new PCA objects.
    save_pca_to : str, optional
        Plot and save the principal components to this file. Default: skip this step.
    save_reconstruction_to : str, optional
        Plot and save the reconstructed light curves to this file (slow). Default: skip this step.

    Returns
    -------
    t_good : astropy.table.Table
        Slice of the input table with a 'features' column added. Rows with any bad features are excluded.
    """
    R_filter = []
    for fltr in t.meta['filters']:
        if fltr in R_FILTERS:
            R_filter.append(R_FILTERS[fltr])
        else:
            raise ValueError(f'Unrecognized filter {fltr}. Please add the extinction correction to `R_FILTERS`.')

    if use_median:
        t = t[::t.meta['ndraws']]
        t.meta['ndraws'] = 1
        t['params'] = t['median_params']
    params = t['params'].data
    params[:, :, 0] *= np.vstack([flux_to_luminosity(row, R_filter) for row in t])
    if use_pca:
        time = np.linspace(0., 300., 1000)
        models = produce_lc(time, params, align_to_t0=True)
        t_good, good_models = select_good_events(t, models)
        peakmags = zero_point - 2.5 * np.log10(good_models.max(axis=2))
        logging.info('peak magnitudes extracted')
        models_to_fit = good_models[~t_good['type'].mask]
        coefficients, reconstructed, pcas = get_principal_components(good_models, models_to_fit,
                                                                     stored_pcas=stored_pcas)
        if save_pca_to is not None:
            plot_principal_components(pcas, time, t.meta['filters'], save_pca_to)
        logging.info('PCA finished')
        if save_reconstruction_to is not None:
            plot_pca_reconstruction(models_to_fit, reconstructed, coefficients, t.meta['filters'],
                                    save_reconstruction_to)
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
    if 'type' in t_input.colnames:
        t_input['type'] = np.ma.array(t_input['type'])
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
    save_dict = t.meta.copy()
    for col in set(t.colnames) - set(meta_columns):
        save_dict[col] = t[col]
    np.savez_compressed(f'{basename}.npz', **save_dict)
    logging.info(f'data saved to {basename}.txt and {basename}.npz')


def _compile_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_table', type=str, help='List of input light curve files')
    parser.add_argument('stored_models', help='Directory where the PyMC3 trace data is stored')
    parser.add_argument('--filters', type=str, default='griz', help='Filters from which to extract features')
    parser.add_argument('--ndraws', type=int, default=10, help='Number of draws from the LC posterior for test set.')
    parser.add_argument('--random-state', type=int, help='Seed for the random number generator (for reproducibility).')
    parser.add_argument('--output', default='params', help='Filename (without extension) to save the parameters')
    args = parser.parse_args()

    data_table = compile_data_table(args.input_table)
    data_table = compile_parameters(data_table, args.stored_models, args.filters, args.ndraws, args.random_state)
    save_data(data_table, args.output)
    plot_data = data_table[~data_table['type'].mask]
    if plot_data:
        plot_histograms(plot_data, 'params', var_kwd='paramnames', row_kwd='filters',
                        saveto=args.output + '_parameters.pdf')


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_table', type=str, help='List of input light curve files, or input data table')
    parser.add_argument('--param-table', type=str, help='Data table corresponding to the Numpy parameter file '
                                                        '(if different than input_table)')
    parser.add_argument('--params', type=str, help='Numpy file containing stored model parameters')
    parser.add_argument('--use-median', action='store_true', help='Use median parameters instead of multiple draws')
    parser.add_argument('--pcas', help='Path to pickled PCA objects. Default: create and fit new PCA objects.')
    parser.add_argument('--use-params', action='store_false', dest='use_pca', help='Use model parameters as features')
    parser.add_argument('--reconstruct', action='store_true',
                        help='Plot and save the reconstructed light curves to {output}_reconstruction.pdf (slow)')
    parser.add_argument('--output', default='test_data', help='Filename (without extension) to save the features')
    args = parser.parse_args()

    logging.info('started feature extraction')
    if args.param_table is None:
        data_table = load_data(args.input_table, args.params)
    else:
        input_table = Table.read(args.input_table, format='ascii')
        data_table = load_data(args.param_table, args.params)
        data_table = join(input_table[['filename']], data_table)
    test_data = extract_features(data_table, use_median=args.use_median, use_pca=args.use_pca, stored_pcas=args.pcas,
                                 save_pca_to=args.output + '_pca.pdf',
                                 save_reconstruction_to=args.output+'_reconstruction.pdf' if args.reconstruct else None)
    save_data(test_data, args.output)
    plot_data = test_data[~test_data['type'].mask]
    if plot_data:
        plot_histograms(plot_data, 'features', var_kwd='featnames', row_kwd='filters',
                        no_autoscale=['SLSN', 'SNIIn'] if args.use_pca else [], saveto=args.output + '_features.pdf')
    plot_feature_correlation(test_data, saveto=args.output + '_correlation.pdf')
    logging.info('finished feature extraction')
