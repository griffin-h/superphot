import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from .classify import _validate_args, validate_classifier, calc_metrics
from .util import subplots_layout
from astropy.table import Table, vstack, join
from argparse import ArgumentParser
import os
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
from itertools import product
import json
from multiprocessing import Pool


def titlecase(x):
    """Capitalize the first letter of each word in a string (where words are separated by whitespace)."""
    words = x.split()
    cap_words = [word[0].upper() + word[1:] for word in words]
    title = ' '.join(cap_words)
    return title


def plot_hyperparameters_3d(t, ccols, xcol, ycol, zcol, cmap=None, cmin=None, cmax=None, figtitle=''):
    """
    Plot 3D scatter plots of the metrics against the hyperparameters.

    Parameters
    ----------
    t : astropy.table.Table
        Table of results from `test_hyperparameters`.
    ccols : list
        List of columns to plot as metrics.
    xcol, ycol, zcol : str
        Columns to plot on the x-, y-, and z-axes of the scatter plots.
    cmap : str, optional
        Name of the colormap to use to color the values in `ccols`.
    cmin, cmax : float, optional
        Data limits corresponding to the minimum and maximum colors in `cmap`.
    """
    nrows, ncols = subplots_layout(len(ccols))
    gs = plt.GridSpec(nrows + 1, ncols, height_ratios=(8,) * nrows + (1,))
    fig = plt.figure(figsize=(11., 8.5))
    for (i, j), ccol in zip(product(range(nrows), range(ncols)), ccols):
        ax = fig.add_subplot(gs[i, j], projection='3d')
        ax.minorticks_off()
        ax.set_zticks([], minor=True)
        logz = np.log(t[zcol])  # 3D axes do not support log scaling
        cbar = ax.scatter(t[xcol], t[ycol], logz, marker='o', c=t[ccol], cmap=cmap, vmin=cmin, vmax=cmax, alpha=1.)
        vmin, vmax = np.percentile(t[ccol], [0., 100.])
        ax.set_xlabel(xcol.split('__')[1])
        ax.set_ylabel(ycol.split('__')[1])
        ax.set_zlabel(zcol.split('__')[1])
        ax.set_xticks(np.unique(t[xcol]))
        ax.set_yticks(np.unique(t[ycol]))
        ax.set_zticks(np.unique(logz))
        ax.set_zticklabels(np.unique(t[zcol]))
        title = titlecase(ccol.replace('_', ' '))
        if len(title) > 12:
            title = title[:-7] + '.'
        ax.set_title(f'${vmin:.2f}$ < {title} < ${vmax:.2f}$', size='medium')
    cax = fig.add_subplot(gs[-1, :])
    cax = fig.colorbar(cbar, cax, orientation='horizontal')
    if cmin is None or cmax is None:
        cax.set_ticks(cbar.get_clim())
        cax.set_ticklabels(['min', 'max'])
    fig.text(0.5, 0.99, figtitle, va='top', ha='center')
    fig.tight_layout(pad=3.)
    return fig


def plot_hyperparameters_with_diff(t, dcol=None, xcol=None, ycol=None, zcol=None, saveto=None, **criteria):
    """
    Plot the metrics for one value of `dcol` and the difference in the metrics for the second value of `dcol`.

    Parameters
    ----------
    t : astropy.table.Table
        Table of results from `test_hyperparameters`.
    dcol : str, optional
        Column to plot as a difference. Default: alphabetically first column starting with 'classifier'
    xcol, ycol, zcol : str, optional
        Columns to plot on the x-, y-, and z-axes of the scatter plots. Default: alphabetically 2nd-4th columns
        starting wtih 'classifier'.
    saveto : str, optional
        Save the plot to this filename. If None, the plot is displayed and not saved.
    """
    for key, val in criteria.items():
        t = t[t[key] == val]
        t.remove_column(key)
    kcols = sorted({col for col in t.colnames if col.startswith('classifier')} - {dcol, xcol, ycol, zcol})[::-1]
    if dcol is None:
        dcol = kcols.pop()
    if xcol is None:
        xcol = kcols.pop()
    if ycol is None:
        ycol = kcols.pop()
    if zcol is None:
        zcol = kcols.pop()
    t = t.group_by(dcol)
    if len(t.groups) != 2:
        raise ValueError('`dcol` must have exactly two possible values')
    t.sort([dcol, xcol, ycol, zcol])
    t0, t1 = t.groups
    ccols = sorted(set(t.colnames) - {dcol, xcol, ycol, zcol})
    j = join(t0, t1, keys=[xcol, ycol, zcol])
    for ccol in ccols:
        j[ccol] = j[ccol + '_1'] - j[ccol + '_2']
    title1 = ', '.join([f'{key} = {val}' for key, val in criteria.items()] + [f'{dcol} = {t.groups.keys[dcol][0]}'])
    fig1 = plot_hyperparameters_3d(t0, ccols, xcol, ycol, zcol, figtitle=title1)
    title2 = f'{dcol} = {t.groups.keys[dcol][0]} → {t.groups.keys[dcol][1]}'
    fig2 = plot_hyperparameters_3d(j, ccols, xcol, ycol, zcol, cmap='coolwarm_r', cmin=-0.2, cmax=0.2, figtitle=title2)
    if saveto is None:
        plt.show()
    else:
        with PdfPages(saveto) as pdf:
            pdf.savefig(fig1)
            pdf.savefig(fig2)
    plt.close(fig1)
    plt.close(fig2)


def _plot_hyperparameters_from_file():
    parser = ArgumentParser()
    parser.add_argument('results', help='Table of results from superphot-optimize.')
    parser.add_argument('--saveto', help='Filename to which to save the plot.')
    args = parser.parse_args()

    t = Table.read(args.results, format='ascii')
    dcol, xcol, ycol, zcol = sorted([col for col in t.colnames if col.startswith('classifier__')])
    plot_hyperparameters_with_diff(t, dcol, xcol, ycol, zcol, saveto=args.saveto)


def _main():
    parser = ArgumentParser()
    parser.add_argument('param_dist', help='JSON-encoded parameter grid/distribution to test.')
    parser.add_argument('pipeline', help='Filename of the pickled pipeline object.')
    parser.add_argument('test_data', help='Filename of the metadata table for the test set.')
    parser.add_argument('--train-data', help='Filename of the metadata table for the training set.'
                                             'By default, use all classified supernovae in the test data.')
    parser.add_argument('-i', '--n-iter', type=int, help='Number of hyperparameter combinations to try. Default: all.')
    parser.add_argument('-j', '--n-jobs', type=int, help='Number of parallel processes to use. Default: 1.')
    parser.add_argument('--saveto', default='hyperparameters.txt', help='Filename to which to write the results.')
    args = parser.parse_args()
    pipeline, train_data, validation_data = _validate_args(args)

    with open(args.param_dist, 'r') as f:
        param_distributions = json.load(f)

    def test_hyperparams(param_set):
        try:
            pipeline.set_params(classifier__n_jobs=1, **param_set)
            results = validate_classifier(pipeline, train_data, validation_data)
            param_set = calc_metrics(results, param_set, save=True)
        except Exception as e:
            logging.error(f'Problem testing {param_set}:\n{e}')
        return param_set

    if args.n_iter is None:
        ps = ParameterGrid(param_distributions)
    else:
        ps = ParameterSampler(param_distributions, n_iter=args.n_iter)
    logging.info(f'Testing {len(ps):d} combinations...')

    if args.n_jobs is None:
        rows = [test_hyperparams(param_set) for param_set in ps]
    else:
        with Pool(args.n_jobs) as p:
            rows = p.map(test_hyperparams, ps)

    tfinal = Table(rows)
    if os.path.exists(args.saveto):
        logging.warning(f'Appending results to {args.saveto}')
        tprev = Table.read(args.saveto, format='ascii')
        tfinal = vstack([tprev, tfinal])
    else:
        logging.info(f'Writing results to {args.saveto}')
    tfinal.write(args.saveto, format='ascii.fixed_width_two_line', overwrite=True)

    plot_hyperparameters_with_diff(tfinal, saveto=args.saveto.replace('.txt', '.pdf'))
