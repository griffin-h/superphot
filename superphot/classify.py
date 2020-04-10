#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import logging
from astropy.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    from imblearn.over_sampling.base import BaseOverSampler
    from imblearn.utils._docstring import Substitution, _random_state_docstring
    from imblearn.over_sampling import SMOTE
from .util import meta_columns, select_labeled_events
import itertools
from tqdm import tqdm
from argparse import ArgumentParser
from superphot.extract_features import plot_histograms

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def plot_confusion_matrix(confusion_matrix, classes, ndraws=0, cmap='Blues', purity=False, title=None,
                          xlabel='Photometric Classification', ylabel='Spectroscopic Classification', ax=None):
    """
    Plot a confusion matrix with each cell labeled by its fraction and absolute number.

    Based on tutorial: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    Parameters
    ----------
    confusion_matrix : array-like
        The confusion matrix as a square array of integers.
    classes : list
        List of class labels for the axes of the confusion matrix.
    ndraws : int, optional
        If `ndraws > 0`, divide each cell in the matrix by `ndraws` before plotting and note this in the title.
    cmap : str, optional
        Name of a Matplotlib colormap to color the matrix.
    purity : bool, optional
        If False (default), aggregate by row (true label). If True, aggregate by column (predicted label).
    title : str, optional
        Text to go above the plot. Default: "Confusion Matrix (`N = confusion_matrix.sum()`)".
    xlabel, ylabel : str, optional
        Labels for the x- and y-axes. Default: "True Label" and "Predicted Label".
    ax : matplotlib.pyplot.axes, optional
        Axis on which to plot the confusion matrix. Default: new axis.
    """
    if ndraws:
        confusion_matrix = confusion_matrix / ndraws
    n_per_true_class = confusion_matrix.sum(axis=1)
    n_per_pred_class = confusion_matrix.sum(axis=0)
    if purity:
        cm = confusion_matrix / n_per_pred_class[np.newaxis, :]
        title_word = 'Purity'
    else:
        cm = confusion_matrix / n_per_true_class[:, np.newaxis]
        title_word = 'Completeness'
    if ax is None:
        ax = plt.axes()
    ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    if title is not None:
        ax.set_title(title)
    elif ndraws:
        ax.set_title('{} ($N={:.0f}\\times{:d}$)'.format(title_word, confusion_matrix.sum(), ndraws))
    else:
        ax.set_title('{} ($N={:d}$)'.format(title_word, confusion_matrix.sum()))
    nclasses = len(classes)
    ax.set_xticks(range(nclasses))
    ax.set_yticks(range(nclasses))
    ax.set_xticklabels(['{}\n({:.0f})'.format(label, n) for label, n in zip(classes, n_per_pred_class)])
    ax.set_yticklabels(['{}\n({:.0f})'.format(label, n) for label, n in zip(classes, n_per_true_class)])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.set_ylim(nclasses - 0.5, -0.5)

    thresh = cm.max() / 2.
    cell_label = '{:.2f}\n({:.1f})' if ndraws > 1 else '{:.2f}\n({:.0f})'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cell_label.format(cm[i, j], confusion_matrix[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring.replace('dict or callable', 'dict, callable or int'),
    random_state=_random_state_docstring)
class MultivariateGaussian(BaseOverSampler):
    """Class to perform over-sampling using a multivariate Gaussian (``numpy.random.multivariate_normal``).

    Parameters
    ----------
    {sampling_strategy}

        - When ``int``, it corresponds to the total number of samples in each
          class (including the real samples). Can be used to oversample even
          the majority class. If ``sampling_strategy`` is smaller than the
          existing size of a class, that class will not be oversampled and
          the classes may not be balanced.

    {random_state}
    """
    def __init__(self, sampling_strategy='all', random_state=None):
        self.random_state = random_state
        if isinstance(sampling_strategy, int):
            self.samples_per_class = sampling_strategy
            sampling_strategy = 'all'
        else:
            self.samples_per_class = None
        super().__init__(sampling_strategy=sampling_strategy)

    def _fit_resample(self, X, y):
        self.fit(X, y)

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            X_class = X[y == class_sample]
            if self.samples_per_class is not None:
                n_samples = self.samples_per_class - X_class.shape[0]
            if n_samples <= 0:
                continue

            mean = np.mean(X_class, axis=0)
            cov = np.cov(X_class, rowvar=False)
            rs = check_random_state(self.random_state)
            X_new = rs.multivariate_normal(mean, cov, n_samples)
            y_new = np.repeat(class_sample, n_samples)

            X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled


def fit_predict(clf, sampler, train_data, test_data, scaler=StandardScaler()):
    """
    Train a random forest classifier on `train_data` and use it to classify `test_data`. Balance the classes before
    training by oversampling.

    Parameters
    ----------
    clf : sklearn.emsemble.RandomForestClassifier
        A random forest classifier trained from the classified transients.
    sampler : imblearn.over_sampling.base.BaseOverSampler
        A resampler used to balance the training sample.
    train_data : astropy.table.Table
        Astropy table containing the training data. Must have a 'features' column and a 'type' column.
    test_data : astropy.table.Table
        Astropy table containing the test data. Must have a 'features' column.
    scaler : sklearn.preprocessing.StandardScaler
        Scikit-learn scaler to apply to the features before training and classification. Default: mean = 0, var = 1.

    Returns
    -------
    p_class : numpy.array
        Classification probabilities for each of the supernovae in `test_features`.
    """
    train_features = scaler.fit_transform(train_data['features'].reshape(len(train_data), -1))
    test_features = scaler.transform(test_data['features'].reshape(len(test_data), -1))
    features_resamp, labels_resamp = sampler.fit_resample(train_features, train_data['type'])
    clf.fit(features_resamp, labels_resamp)
    p_class = clf.predict_proba(test_features)
    return p_class


def mean_axis0(x, axis=0):
    """Equivalent to the numpy.mean function but with axis=0 by default."""
    return x.mean(axis=axis)


def aggregate_probabilities(table):
    """
    Average the classification probabilities for a given supernova across the multiple model light curves.

    Parameters
    ----------
    table : astropy.table.Table
        Astropy table containing the metadata for a supernova and the classification probabilities from
        `clf.predict_proba` (column name = 'probabilities')

    Returns
    -------
    results : astropy.table.Table
        Astropy table containing the supernova metadata and average classification probabilities for each supernova
    """
    table.keep_columns(meta_columns + ['probabilities'])
    grouped = table.filled().group_by(meta_columns)
    results = grouped.groups.aggregate(mean_axis0)
    return results


def validate_classifier(clf, sampler, train_data, test_data=None):
    """
    Validate the performance of a machine-learning classifier using leave-one-out cross-validation.

    Parameters
    ----------
    clf : sklearn.emsemble.RandomForestClassifier
        The classifier to validate.
    sampler : imblearn.over_sampling.SMOTE
        First resample the data using this sampler.
    train_data : astropy.table.Table
        Astropy table containing the training data. Must have a 'features' column and a 'type' column.
    test_data : astropy.table.Table, optional
        Astropy table containing the test data. Must have a 'features' column to which to apply the trained classifier.
        If None, use the training data itself for validation.

    Returns
    -------
    p_class : numpy.array
        Classification probabilities for each of the supernovae in `test_features`.
    """
    if test_data is None:
        test_data = train_data
    p_class = fit_predict(clf, sampler, train_data, test_data)
    for filename in tqdm(train_data['filename'], desc='Cross-validation'):
        train_index = train_data['filename'] != filename
        test_index = test_data['filename'] == filename
        p_class[test_index] = fit_predict(clf, sampler, train_data[train_index], test_data[test_index])
    return p_class


def make_confusion_matrix(results, classes=None, p_min=0., saveto=None, purity=False):
    """
    Given a data table with classification probabilities, calculate and plot the confusion matrix.

    Parameters
    ----------
    results : astropy.table.Table
        Astropy table containing the supernova metadata and classification probabilities (column name = 'probabilities')
    classes : array-like, optional
        Labels corresponding to the 'probabilities' column. If None, use the sorted entries in the 'type' column.
    p_min : float, optional
        Minimum confidence to be included in the confusion matrix. Default: include all samples.
    saveto : str, optional
        Save the plot to this filename. If None, the plot is displayed and not saved.
    purity : bool, optional
        If False (default), aggregate by row (true label). If True, aggregate by column (predicted label).
    """
    results = select_labeled_events(results)
    if classes is None:
        classes = np.unique(results['type'])
    predicted_types = classes[np.argmax(results['probabilities'], axis=1)]
    include = results['probabilities'].max(axis=1) > p_min
    cnf_matrix = confusion_matrix(results['type'][include], predicted_types[include])
    fig = plt.figure(figsize=(len(classes),) * 2)
    plot_confusion_matrix(cnf_matrix, classes, purity=purity)
    fig.tight_layout()
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)


def load_data(meta_file, data_file=None):
    """
    Read input from a text file (the metadata table) and a Numpy file (the features) and return as an Astropy table.

    Parameters
    ----------
    meta_file : str
        Filename of the input metadata table. Must in an ASCII format readable by Astropy.
    data_file : str, optional
        Filename where the features are saved. Must be in Numpy binary format. If None, replace the extension of
        `meta_file` with .npz.

    Returns
    -------
    data_table : astropy.table.Table
        Table containing the metadata along with a 'features' column.
    """
    if data_file is None:
        meta_file_parts = meta_file.split('.')
        meta_file_parts[-1] = 'npz'
        data_file = '.'.join(meta_file_parts)
    t = Table.read(meta_file, format='ascii', fill_values=('', ''))
    stored = np.load(data_file)
    data_table = t[np.repeat(np.arange(len(t)), stored['ndraws'])]
    for col in data_table.colnames:
        if data_table[col].dtype.type is np.str_:
            data_table[col].fill_value = ''
    for key in stored:
        if key in ['ndraws', 'paramnames', 'featnames']:
            data_table.meta[key] = stored[key]
        else:
            data_table[key] = stored[key]
    logging.info(f'data loaded from {meta_file} and {data_file}')
    return data_table


def load_results(filename):
    results = Table.read(filename, format='ascii')
    classes = [col for col in results.colnames if col not in meta_columns]
    results['probabilities'] = np.stack([results[sntype].data for sntype in classes]).T
    results.remove_columns(classes)
    return results


def write_results(test_data, classes, filename):
    """
    Write the classification results to a text file.

    Parameters
    ----------
    test_data : astropy.table.Table
        Astropy table containing the supernova metadata and the classification probabilities for each sample
        from `clf.predict_proba` (column name = 'probabilities').
    classes : list
        The labels that correspond to the columns in 'probabilities'
    filename : str
        Name of the output file
    """
    output = test_data[meta_columns]
    output['MWEBV'].format = '%.4f'
    output['redshift'].format = '%.4f'
    for i, classname in enumerate(classes):
        output[classname] = test_data['probabilities'][:, i]
        output[classname].format = '%.3f'
    output.write(filename, format='ascii.fixed_width_two_line', overwrite=True)
    logging.info(f'classification results saved to {filename}')


def plot_confusion_matrix_from_file():
    parser = ArgumentParser()
    parser.add_argument('filename', type=str, help='Filename containing the table of classification results.')
    parser.add_argument('--pmin', type=float, default=0.,
                        help='Minimum confidence to be included in the confusion matrix.')
    parser.add_argument('--saveto', type=str, help='If provided, save the confusion matrix to this file.')
    parser.add_argument('--purity', action='store_true', help='Aggregate by column instead of by row.')
    args = parser.parse_args()

    results = load_results(args.filename)
    make_confusion_matrix(results, p_min=args.pmin, saveto=args.saveto, purity=args.purity)


def main():
    parser = ArgumentParser()
    parser.add_argument('test_data', help='Filename of the metadata table for the test set.')
    parser.add_argument('--train-data', help='Filename of the metadata table for the training set.'
                                             'By default, use all classified supernovae in the test data.')
    parser.add_argument('--classifier', choices=['rf', 'svm', 'mlp'], default='rf', help='The classification algorithm '
                        'to use. Current choices are "rf" (random forest; default), "svm" (support vector machine), or '
                        '"mlp" (multilayer perceptron).')
    parser.add_argument('--sampler', choices=['mvg', 'smote'], default='mvg', help='The resampling algorithm to use. '
                        'Current choices are "mvg" (multivariate Gaussian; default) or "smote" (synthetic minority '
                        'oversampling technique).')
    parser.add_argument('--random-state', type=int, help='Seed for the random number generator (for reproducibility).')
    parser.add_argument('--pmin', type=float, default=0.,
                        help='Minimum confidence to be included in the confusion matrix.')
    args = parser.parse_args()

    logging.info('started classify.py')
    test_data = load_data(args.test_data)
    if args.train_data is None:
        train_data = test_data
    else:
        train_data = load_data(args.train_data)
    train_data = select_labeled_events(train_data)
    validation_data = select_labeled_events(test_data)

    if args.classifier == 'rf':
        clf = RandomForestClassifier(criterion='entropy', max_features=5, n_jobs=-1, random_state=args.random_state)
    elif args.classifier == 'svm':
        clf = SVC(C=1000, gamma=0.1, probability=True, random_state=args.random_state)
    elif args.classifier == 'mlp':
        clf = MLPClassifier(hidden_layer_sizes=(10, 5), alpha=1e-5, early_stopping=True, random_state=args.random_state)
    else:
        raise NotImplementedError(f'{args.classifier} is not a recognized classifier type')

    if args.sampler == 'mvg':
        sampler = MultivariateGaussian(sampling_strategy=1000, random_state=args.random_state)
    elif args.sampler == 'smote':
        sampler = SMOTE(random_state=args.random_state)
    else:
        raise NotImplementedError(f'{args.sampler} is not a recognized sampler type')

    test_data['probabilities'] = fit_predict(clf, sampler, train_data, test_data)
    test_data['prediction'] = clf.classes_[test_data['probabilities'].argmax(axis=1)]
    plot_histograms(test_data, 'params', 'prediction', varnames=test_data.meta['paramnames'],
                    saveto='phot_class_parameters.pdf')
    plot_histograms(test_data, 'features', 'prediction', varnames=test_data.meta['featnames'],
                    no_autoscale=['SLSN', 'SNIIn'], saveto='phot_class_features.pdf')

    results = aggregate_probabilities(test_data)
    write_results(results, clf.classes_, 'results.txt')

    validation_data['probabilities'] = validate_classifier(clf, sampler, train_data, validation_data)
    write_results(validation_data, clf.classes_, 'validation_full.txt')
    results_validate = aggregate_probabilities(validation_data)
    write_results(results_validate, clf.classes_, 'validation.txt')
    make_confusion_matrix(results_validate, clf.classes_, args.pmin, 'confusion_matrix.pdf')
    logging.info('validation complete')
    logging.info('finished classify.py')
