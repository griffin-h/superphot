import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from glob import glob
import os
import re
import argparse
import logging
from astropy.table import Table, hstack, join
from astropy.cosmology import Planck15 as cosmo
from sklearn.decomposition import PCA
from tqdm import trange
from .util import filter_colors, meta_columns, load_data, plot_histograms, subplots_layout
from .fit import read_light_curve, produce_lc, PARAMNAMES
import pickle
from scipy.stats import spearmanr

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# using WavelengthMean from the SVO Filter Profile Service http://svo2.cab.inta-csic.es/theory/fps/
R_FILTERS = {'g': 3.57585511, 'r': 2.54033913, 'i': 1.88284171, 'z': 1.49033933, 'y': 1.24431944,  # Pan-STARRS filters
             'U': 4.78442941, 'B': 4.05870021, 'V': 3.02182672, 'R': 2.34507832, 'I': 1.69396924}  # Bessell filters


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

    Luminosities are per steradian (i.e., the factor of 4π is not included) for easy conversion to absolute magnitudes.

    .. math:: \\frac{L_ν}{4π F_ν}  = \\frac{D_L^2 10^{0.4 E(B-V) R}}{1+z}

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
    flux2lum = 10. ** (A_coeffs / 2.5) * dist ** 2. / (1. + row['redshift'])
    return flux2lum


def get_principal_components(light_curves, n_components=6, whiten=True):
    """
    Run a principal component analysis on a set of light curves for each filter.

    Parameters
    ----------
    light_curves : array-like
        An array of model light curves to be used for fitting the PCA.
    n_components : int, optional
        The number of principal components to calculate. Default: 6.
    whiten : bool, optional
        Whiten the input data before calculating the principal components. Default: True.

    Returns
    -------
    pcas : list
        A list of the PCA objects for each filter.
    """
    pcas = []
    for i in range(light_curves.shape[1]):
        pca = PCA(n_components, whiten=whiten)
        pca.fit(light_curves[:, i])
        pcas.append(pca)
    return pcas


def project_onto_principal_components(light_curves, pcas):
    """
    Project a set of light curves onto their principal components for each filter.

    Parameters
    ----------
    light_curves : array-like
        An array of model light curves to be projected onto the principal components.
    pcas : list
        A list of the PCA objects for each filter.

    Returns
    -------
    coefficients : numpy.ndarray
        An array of the coefficients on the principal components.
    reconstructed : numpy.ndarray
        An reconstruction of the light curves from their principal components.
    """
    coefficients = np.empty(light_curves.shape[:-1] + (pcas[0].n_components_,))
    reconstructed = np.empty_like(light_curves)
    for i, pca in enumerate(pcas):
        coefficients[:, i] = pca.transform(light_curves[:, i])
        reconstructed[:, i] = pca.inverse_transform(coefficients[:, i])
    explained_variance = coefficients.var(axis=0) * [pca.explained_variance_ if pca.whiten else
                                                     np.ones_like(pca.explained_variance_) for pca in pcas]
    explained_variance_ratio = explained_variance.sum(axis=-1) / light_curves.var(axis=0).sum(axis=-1)
    logging.info(f'PCA explained variance ratios: {explained_variance_ratio}')
    return coefficients, reconstructed


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
            ax.set_xlabel('Phase (d)')
    lines = []
    if filters is None:
        filters = [f'Filter {i+1:d}' for i in range(len(pcas))]
    for pca, fltr in zip(pcas, filters):
        for pc, ax in zip(pca.components_, axes.flat):
            p = ax.plot(time, pc, color=filter_colors.get(fltr), label=fltr)
        lines += p
    fig.legend(lines, filters, ncol=len(filters), loc='upper center', title='Principal Components')
    fig.tight_layout(h_pad=0., w_pad=0., rect=(0., 0., 1., 0.9))
    fig.savefig(saveto)
    plt.close(fig)


def plot_pca_reconstruction(models, reconstructed, time=None, coefficients=None, filters=None, titles=None,
                            saveto='pca_reconstruction.pdf'):
    """
    Plot comparisons between the model light curves and the light curves reconstructed from the PCA for each transient.
    These are saved as a multipage PDF.

    Parameters
    ----------
    models : array-like
        A 3-D array of model light curves with shape (ntransients, nfilters, ntimes)
    reconstructed : array-like
        A 3-D array of reconstructed light curves with shape (ntransients, nfilters, ntimes)
    time : array-like, optional
        A 1-D array of times that correspond to the last axis of `models`. Default: x-axis will run from 0 to ntimes.
    coefficients : array-like, optional
        A 3-D array of the principal component coefficients with shape (ntransients, nfilters, ncomponents). If given,
        the coefficients will be printed at the top right of each plot.
    filters : iterable, optional
        Names of the filters corresponding to the PCA objects. Only used for coloring the lines.
    titles : iterable, optional
        Titles for each plot.
    saveto : str, optional
        Filename for the output file. Default: pca_reconstruction.pdf.
    """
    if time is None:
        time = np.arange(models.shape[-1])
        xlabel = None
    else:
        xlabel = 'Phase (d)'
    if coefficients is None:
        legend_title = None
    else:
        legend_title = 'Principal Component Projection'
    if filters is None:
        filters = [f'Filter {i+1:d}' for i in range(models.shape[1])]
    with PdfPages(saveto) as pdf:
        fig, ax = plt.subplots()
        for i in trange(models.shape[0], desc='PCA reconstruction'):
            for j in range(models.shape[1]):
                c = filter_colors.get(filters[j])
                if coefficients is None:
                    label = filters[j]
                else:
                    with np.printoptions(precision=2, suppress=True, floatmode='fixed'):
                        label = f'{filters[j]} = {coefficients[i, j]}'
                ax.plot(time, models[i, j], color=c)
                ax.plot(time, reconstructed[i, j], ls=':', color=c, label=label)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Luminosity')
            ax.set_title(titles[i])
            ax.legend(title=legend_title)
            fig.tight_layout()
            pdf.savefig(fig)
            ax.clear()
        plt.close(fig)


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


def compile_parameters(stored_models, filters, ndraws=10, random_state=None):
    """
    Read the saved PyMC3 traces and compile an array of fit parameters for each transient. Save to a Numpy file.

    Parameters
    ----------
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
    basenames = set()
    for fn in os.listdir(stored_models):
        match = re.search('(\\w+)_2\\w+', fn)
        if match is not None:
            basenames.add(match.groups()[0])
    t = Table([sorted(basenames)], names=['filename'])
    for i, basename in enumerate(t['filename']):
        try:
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
        if stored_pcas is None:
            pcas = get_principal_components(good_models[~t_good['type'].mask])
            with open('pca.pickle', 'wb') as f:
                pickle.dump(pcas, f)
        else:
            with open(stored_pcas, 'rb') as f:
                pcas = pickle.load(f)
        coefficients, reconstructed = project_onto_principal_components(good_models, pcas)
        if save_pca_to is not None:
            plot_principal_components(pcas, time, t.meta['filters'], save_pca_to)
        logging.info('PCA finished')
        if save_reconstruction_to is not None:
            plot_pca_reconstruction(good_models, reconstructed, time, coefficients, t.meta['filters'],
                                    t_good['filename'], save_reconstruction_to)
        features = np.dstack([peakmags, coefficients])
        t_good.meta['featnames'] = ['Peak Abs. Mag.'] + [f'PC{i:d} Proj.' for i in range(1, 7)]
    else:
        params[:, :, 0] = zero_point - 2.5 * np.log10(params[:, :, 0])  # convert amplitude to magnitude
        t_good, features = select_good_events(t, params[:, :, [0, 1, 2, 4, 5]])  # remove reference epoch from features
        t_good.meta['featnames'] = ['Amplitude (mag)'] + [PARAMNAMES[i] for i in [1, 2, 4, 5]]
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
    return t_final


def save_data(t, basename):
    t.sort('filename')
    save_table = t[[col for col in t.colnames if col in meta_columns]][::t.meta['ndraws']]
    if 'MWEBV' in save_table.colnames:
        save_table['MWEBV'].format = '%.4f'
    if 'redshift' in save_table.colnames:
        save_table['redshift'].format = '%.4f'
    save_table.write(f'{basename}.txt', format='ascii.fixed_width_two_line', overwrite=True)
    save_dict = t.meta.copy()
    for col in set(t.colnames) - set(meta_columns):
        save_dict[col] = t[col]
    np.savez_compressed(f'{basename}.npz', **save_dict)
    logging.info(f'data saved to {basename}.txt and {basename}.npz')


def _compile_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('stored_models', help='Directory where the PyMC3 trace data is stored')
    parser.add_argument('--filters', type=str, default='griz', help='Filters from which to extract features')
    parser.add_argument('--ndraws', type=int, default=10, help='Number of draws from the LC posterior for test set.')
    parser.add_argument('--random-state', type=int, help='Seed for the random number generator (for reproducibility).')
    parser.add_argument('--output', default='params', help='Filename (without extension) to save the parameters')
    args = parser.parse_args()

    data_table = compile_parameters(args.stored_models, args.filters, args.ndraws, args.random_state)
    np.savez_compressed(args.output, **data_table, **data_table.meta)


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_table', help='Filename containing metadata (redshift, MWEBV) for the light curves.')
    parser.add_argument('param_table', help='Filename of the Numpy archive containing the parameters.')
    parser.add_argument('--use-median', action='store_true', help='Use median parameters instead of multiple draws')
    parser.add_argument('--pcas', help='Path to pickled PCA objects. Default: create and fit new PCA objects.')
    parser.add_argument('--use-params', action='store_false', dest='use_pca', help='Use model parameters as features')
    parser.add_argument('--reconstruct', action='store_true',
                        help='Plot and save the reconstructed light curves to {output}_reconstruction.pdf (slow)')
    parser.add_argument('--output', default='test_data', help='Filename (without extension) to save the features')
    args = parser.parse_args()

    logging.info('started feature extraction')
    data_table = load_data(args.input_table, args.param_table)
    test_data = extract_features(data_table, use_median=args.use_median, use_pca=args.use_pca, stored_pcas=args.pcas,
                                 save_pca_to=args.output + '_pca.pdf',
                                 save_reconstruction_to=args.output+'_reconstruction.pdf' if args.reconstruct else None)
    save_data(test_data, args.output)
    if 'type' in data_table.colnames and not data_table['type'].mask.all():
        plot_data = test_data[~test_data['type'].mask]
        plot_histograms(plot_data, 'features', var_kwd='featnames', row_kwd='filters',
                        saveto=args.output + '_features.pdf')
    plot_feature_correlation(test_data, saveto=args.output + '_correlation.pdf')
    logging.info('finished feature extraction')
