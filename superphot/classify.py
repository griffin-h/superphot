#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import logging
from astropy.table import Table
from sklearn.ensemble import RandomForestClassifier
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

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def plot_confusion_matrix(confusion_matrix, classes, ndraws=0, title=None, cmap='Blues', filename=None):
    """
    This function prints and plots the confusion matrix.
    From tutorial: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if ndraws:
        confusion_matrix = confusion_matrix / ndraws
    n_per_class = confusion_matrix.sum(axis=1)
    cm = confusion_matrix / n_per_class[:, np.newaxis]
    plt.figure(figsize=(6., 6.))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    if title is not None:
        plt.title(title)
    elif ndraws:
        plt.title('Confusion Matrix ($N={:.0f}\\times{:d}$)'.format(confusion_matrix.sum(), ndraws))
    else:
        plt.title('Confusion Matrix ($N={:d}$)'.format(confusion_matrix.sum()))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, ['{}\n({:.0f})'.format(label, n) for label, n in zip(classes, n_per_class)])
    plt.ylim(4.5, -0.5)

    thresh = cm.max() / 2.
    cell_label = '{:.2f}\n({:.1f})' if ndraws > 1 else '{:.2f}\n({:.0f})'
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cell_label.format(cm[i, j], confusion_matrix[i, j]),
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)


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


def fit_predict(clf, sampler, train_data, test_data):
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

    Returns
    -------
    p_class : numpy.array
        Classification probabilities for each of the supernovae in `test_features`.
    """
    features_resamp, labels_resamp = sampler.fit_resample(train_data['features'], train_data['type'])
    clf.fit(features_resamp, labels_resamp)
    p_class = clf.predict_proba(test_data['features'])
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


def make_confusion_matrix(results, classes=None, p_min=0., saveto=None):
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
    """
    results = select_labeled_events(results)
    if classes is None:
        classes = np.unique(results['type'])
    predicted_types = classes[np.argmax(results['probabilities'], axis=1)]
    include = results['probabilities'].max(axis=1) > p_min
    cnf_matrix = confusion_matrix(results['type'][include], predicted_types[include])
    plot_confusion_matrix(cnf_matrix, classes, filename=saveto)


def load_data(basename):
    t = Table.read(f'{basename}.txt', format='ascii', fill_values=('', ''))
    stored = np.load(f'{basename}.npz')
    data_table = t[np.repeat(np.arange(len(t)), stored['ndraws'])]
    for col in data_table.colnames:
        if data_table[col].dtype.type is np.str_:
            data_table[col].fill_value = ''
    data_table['features'] = stored['features']
    logging.info(f'data loaded from {basename}.txt and {basename}.npz')
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
    args = parser.parse_args()

    results = load_results(args.filename)
    make_confusion_matrix(results, p_min=args.pmin, saveto=args.saveto)


def main():
    parser = ArgumentParser()
    parser.add_argument('test_data', help='Filename (without extension) of the test data table and features.')
    parser.add_argument('--train-data', help='Filename (without extension) of the training data and features.'
                                             'By default, use all classified supernovae in the test data.')
    parser.add_argument('--estimators', type=int, default=100,
                        help='Number of estimators (trees) in the random forest classifier.')
    parser.add_argument('--criterion', default='entropy', choices=['gini', 'entropy'],
                        help='The function to measure the quality of a split in the random forest classifier.')
    parser.add_argument('--max-depth', type=int, help='Maximum depth of a tree in the random forest classifier. '
                        'By default, the tree will have all pure leaves.')
    parser.add_argument('--max-features', type=int, default=5,
                        help='Maximum number of features in the decision tree before making a split.')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of jobs to run in parallel for the classifier. '
                        'By default, use all available processors.')
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
    scaler = StandardScaler(copy=False)
    scaler.fit_transform(train_data['features'])
    scaler.transform(test_data['features'])
    validation_data = select_labeled_events(test_data)

    clf = RandomForestClassifier(n_estimators=args.estimators, criterion=args.criterion, max_depth=args.max_depth,
                                 max_features=args.max_features, n_jobs=args.jobs, random_state=args.random_state)
    if args.sampler == 'mvg':
        sampler = MultivariateGaussian(sampling_strategy=1000, random_state=args.random_state)
    elif args.sampler == 'smote':
        sampler = SMOTE(random_state=args.random_state)
    else:
        raise NotImplementedError(f'{args.sampler} is not a recognized sampler type')

    test_data['probabilities'] = fit_predict(clf, sampler, train_data, test_data)
    results = aggregate_probabilities(test_data)
    write_results(results, clf.classes_, 'results.txt')

    validation_data['probabilities'] = validate_classifier(clf, sampler, train_data, validation_data)
    write_results(validation_data, clf.classes_, 'validation_full.txt')
    results_validate = aggregate_probabilities(validation_data)
    write_results(results_validate, clf.classes_, 'validation.txt')
    make_confusion_matrix(results_validate, clf.classes_, args.pmin, 'confusion_matrix.pdf')
    logging.info('validation complete')
    logging.info('finished classify.py')
