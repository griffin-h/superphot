import numpy as np
import pickle
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from .classify import load_data, validate_classifier, aggregate_probabilities
from .util import select_labeled_events, subplots_layout
from astropy.table import Table, vstack
from argparse import ArgumentParser
import os
import logging
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.markers import MarkerStyle
from itertools import product
import json
from multiprocessing import Pool
from functools import partial


def test_hyperparameters(param_set, pipeline, train_data, test_data):
    """
    Validates the pipeline for a set of hyperparameters.

    Measures F1 score and accuracy, as well as completeness and purity for each class.

    Parameters
    ----------
    param_set : dict
        A dictionary containing keywords that match the parameters of `pipeline` and values to which to set them.
    pipeline : sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
        The full classification pipeline, including rescaling, resampling, and classification.
    train_data : astropy.table.Table
        Astropy table containing the training data. Must have a 'features' column and a 'type' column.
    test_data : astropy.table.Table
        Astropy table containing the test data. Must have a 'features' column.
    """
    param_names = sorted(param_set.keys())
    pipeline.set_params(classifier__n_jobs=1, **param_set)
    validation_data = select_labeled_events(test_data)
    validation_data['probabilities'] = validate_classifier(pipeline, train_data, validation_data)
    results = aggregate_probabilities(validation_data)
    predicted_types = pipeline.classes_[np.argmax(results['probabilities'], axis=1)]
    param_set['accuracy'] = accuracy_score(results['type'], predicted_types)
    param_set['f1_score'] = f1_score(results['type'], predicted_types, average='macro')
    cnf_matrix = confusion_matrix(results['type'], predicted_types)
    completeness = np.diag(cnf_matrix) / cnf_matrix.sum(axis=1)
    purity = np.diag(cnf_matrix) / cnf_matrix.sum(axis=0)
    for sntype, complete, pure in zip(pipeline.classes_, completeness, purity):
        param_set[sntype + '_completeness'] = complete
        param_set[sntype + '_purity'] = pure
    filename = '_'.join([str(param_set[key]) for key in param_names]) + '.json'
    with open(filename, 'w') as f:
        json.dump(param_set, f)
    return param_set


def plot_hyperparameters_3d(t, ccols=None, xcol='classifier__max_depth', ycol='classifier__max_features',
                            zcol='classifier__n_estimators', mcol='classifier__criterion', saveto=None):
    """
    Plot 3D scatter plots of the metrics against the hyperparameters.

    Parameters
    ----------
    t : astropy.table.Table
        Table of results from `test_hyperparameters`.
    ccols : list, optional
        List of columns to plot as metrics. Default: all columns other than `xcol`, `ycol`, `zcol`, and `mcol`.
    xcol, ycol, zcol : str, optional
        Columns to plot on the x-, y-, and z-axes of the scatter plots.
    mcol : str, optional
        Column to plot as the marker shape in the scatter plots.
    saveto : str, optional
        Save the plot to this filename. If None, the plot is displayed and not saved.
    """
    t = t.group_by(mcol)
    t.groups.keys['marker'] = None
    if ccols is None:
        ccols = [col for col in t.colnames if col not in [xcol, ycol, zcol, mcol]]
    nrows, ncols = subplots_layout(len(ccols))
    fig = plt.figure(figsize=(12., 10.))
    gs = plt.GridSpec(nrows + 1, ncols, height_ratios=(8,) * nrows + (1,))
    for (i, j), ccol in zip(product(range(nrows), range(ncols)), ccols):
        ax = fig.add_subplot(gs[i, j], projection='3d')
        ax.minorticks_off()
        ax.set_zticks([], minor=True)
        for g, k, m in zip(t.groups, t.groups.keys, MarkerStyle.filled_markers):
            k['marker'] = m
            cmap = ax.scatter(g[xcol], g[ycol], g[zcol], marker=m, c=g[ccol], label=k[mcol], vmin=0., vmax=1.)
        ax.set_xlabel(xcol.split('__')[1])
        ax.set_ylabel(ycol.split('__')[1])
        ax.set_zlabel(zcol.split('__')[1])
        ax.set_title(ccol)
    fig.legend([plt.Line2D([], [], marker=m, color='k', ls='none') for m in t.groups.keys['marker']],
               t.groups.keys[mcol], title=mcol.split('__')[1], loc='upper center', ncol=ncols)
    cax = fig.add_subplot(gs[-1, :])
    fig.colorbar(cmap, cax, orientation='horizontal')
    fig.tight_layout(rect=[0.05, 0., 0.95, 0.95], h_pad=3., w_pad=5.)
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)


def main():
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

    with open(args.param_dist, 'r') as f:
        param_distributions = json.load(f)

    with open(args.pipeline, 'rb') as f:
        pipeline = pickle.load(f)

    test_data = load_data(args.test_data)
    if args.train_data is None:
        train_data = test_data
    else:
        train_data = load_data(args.train_data)
    train_data = select_labeled_events(train_data)

    test_hyperparams = partial(test_hyperparameters, pipeline=pipeline, test_data=test_data, train_data=train_data)

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
        logging.warning('Appending results to', args.saveto)
        tprev = Table.read(args.saveto, format='ascii')
        tfinal = vstack([tprev, tfinal])
    else:
        logging.info('Writing results to', args.saveto)
    tfinal.write(args.saveto, format='ascii.fixed_width_two_line', overwrite=True)
    plot_hyperparameters_3d(tfinal, saveto=args.saveto.replace('.txt', '.pdf'))
