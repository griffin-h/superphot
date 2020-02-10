#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import logging
from astropy.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from sklearn.utils import safe_indexing, check_random_state
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution, _docstring
from imblearn.over_sampling import SMOTE
from .util import get_VAV19
import itertools
from tqdm import tqdm
from argparse import ArgumentParser

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
t_conf = Table.read(get_VAV19('ps1confirmed_only_sne.txt'), format='ascii')
classes = sorted(set(t_conf['type']))


def plot_confusion_matrix(confusion_matrix, ndraws=1, title='Confusion Matrix ($N={:.0f}\\times{:d}$)', cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    From tutorial: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    confusion_matrix = confusion_matrix / ndraws
    n_per_class = confusion_matrix.sum(axis=1)
    cm = confusion_matrix / n_per_class[:, np.newaxis]
    plt.figure(figsize=(6., 6.))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    plt.title(title.format(confusion_matrix.sum(), ndraws))
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
    plt.savefig('confusion_matrix.pdf')


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring.replace('dict or callable', 'dict, callable or int'),
    random_state=_docstring._random_state_docstring)
class MultivariateGaussian(BaseOverSampler):
    """Class to perform over-sampling using a multivariate Gaussian (``numpy.random.multivariate_normal``).

    Parameters
    ----------
    {sampling_strategy}

        - When ```int``, it corresponds to the total number of samples in each
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
            target_class_indices = np.flatnonzero(y == class_sample)
            if self.samples_per_class is not None:
                n_samples = self.samples_per_class - len(target_class_indices)
            if n_samples <= 0:
                continue
            X_class = safe_indexing(X, target_class_indices)

            mean = np.mean(X_class, axis=0)
            cov = np.cov(X_class, rowvar=False)
            rs = check_random_state(self.random_state)
            X_new = rs.multivariate_normal(mean, cov, n_samples)
            y_new = np.repeat(class_sample, n_samples)

            X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled


def train_classifier(data, n_est=100, depth=None, max_feat=5, n_jobs=-1, sampler_type='mvg', random_state=None):
    """
    Initialize and train a random forest classifier. Balance the classes before training by oversampling.

    Parameters
    ----------
    data : astropy.table.Table
        Astropy table containing the training data. Must have a 'features' column and a 'label' (integers) column.
    n_est: int, optional
        The number of trees in the forest. Default: 100.
    depth : int, optional
        The maxiumum depth of a tree. If None, the tree will have all pure leaves.
    max_feat : int, optional
        The maximum number of features used before making a split. If None, use all features.
    n_jobs : int, optional
        The number of jobs to run in parallel for the classifier. If -1, use all available processors.
    sampler_type : str, optional
        The type of resampler to use. Current choices are 'mvg' (multivariate Gaussian; default) or 'smote' (synthetic
        minority oversampling technique).
    random_state : int, optional
        A seed for the random number generators used in the classifier and resampler.

    Returns
    -------
    clf : sklearn.emsemble.RandomForestClassifier
        A random forest classifier trained from the classified transients.
    sampler : imblearn.over_sampling.base.BaseOverSampler
        A resampler used to balance the training sample.
    """
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, class_weight='balanced',
                                 criterion='entropy', max_features=max_feat, n_jobs=n_jobs, random_state=random_state)
    if sampler_type == 'mvg':
        sampler = MultivariateGaussian(sampling_strategy=1000, random_state=random_state)
    elif sampler_type == 'smote':
        sampler = SMOTE(random_state=random_state)
    else:
        raise NotImplementedError(f'{sampler_type} is not a recognized sampler type')
    features_resamp, labels_resamp = sampler.fit_resample(data['features'], data['label'])
    clf.fit(features_resamp, labels_resamp)
    return clf, sampler


def validate_classifier(clf, sampler, data):
    """
    Validate the performance of a machine-learning classifier using leave-one-out cross-validation. The results are
    plotted as a confusion matrix, which is saved as a PDF.

    Parameters
    ----------
    clf : sklearn.emsemble.RandomForestClassifier
        The classifier to validate.
    sampler : imblearn.over_sampling.SMOTE
        First resample the data using this sampler.
    data : astropy.table.Table
        Astropy table containing the training data. Must have a 'features' column and a 'label' (integers) column.
    """
    kf = KFold(len(np.unique(data['id'])))
    p_class = np.empty((len(data), len(classes)))
    pbar = tqdm(desc='Cross-validation', total=kf.n_splits)
    for train_index, test_index in kf.split(data):
        features_resamp, labels_resamp = sampler.fit_resample(data['features'][train_index], data['label'][train_index])
        clf.fit(features_resamp, labels_resamp)
        p_class[test_index] = clf.predict_proba(data['features'][test_index])
        pbar.update()
    pbar.close()

    labels_test = np.argmax(p_class, axis=1)
    cnf_matrix = confusion_matrix(data['label'], labels_test)
    plot_confusion_matrix(cnf_matrix, len(data) // kf.n_splits)
    write_results(data, p_class, 'validation.txt', aggregate=False)
    return cnf_matrix


def load_test_data():
    test_table = Table.read('test_data.txt', format='ascii.fixed_width', fill_values=('', ''))
    test_table['features'] = np.load('test_data.npz')['features']
    logging.info('test data loaded from test_data.txt and test_data.npz')
    return test_table


def write_results(test_data, p_class, filename, aggregate=True):
    """
    Write the classification results to a text file.

    Parameters
    ----------
    test_data : astropy.table.Table
        Astropy table containing the supernova metadata: 'id', 'hostz', 'type', 'flag0', 'flag1', 'flag2'
    p_class : array-like, shape=(nsamples, nclasses)
        Array containing the classification probabilities for each sample (from `clf.predict_proba`)
    filename : str
        Name of the output file
    aggregate : bool, optional
        If True, average the classification probabilities for each unique supernova (if ndraws > 1)
    """
    meta_columns = ['id', 'hostz', 'type', 'flag0', 'flag1', 'flag2']
    output = test_data[meta_columns]
    for i, classname in enumerate(classes):
        output[classname] = p_class[:, i]
        output[classname].format = '%.3f'
    if aggregate:
        for col in ['type', 'flag0', 'flag1', 'flag2']:
            output[col].fill_value = ''
        grouped = output.filled().group_by(meta_columns)
        output = grouped.groups.aggregate(np.mean)
    output.write(filename, format='ascii.fixed_width', overwrite=True)
    logging.info(f'classification results saved to {filename}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--estimators', type=int, default=100,
                        help='Number of estimators (trees) in the random forest classifier.')
    parser.add_argument('--max-depth', type=int, help='Maximum depth of a tree in the random forest classifier. '
                        'By default, the tree will have all pure leaves.')
    parser.add_argument('--max-feat', type=int, default=5,
                        help='Maximum number of features in the decision tree before making a split.')
    parser.add_argument('--jobs', type=int, default=-1, help='Number of jobs to run in parallel for the classifier. '
                        'By default, use all available processors.')
    parser.add_argument('--sampler', choices=['mvg', 'smote'], default='mvg', help='The resampling algorithm to use. '
                        'Current choices are "mvg" (multivariate Gaussian; default) or "smote" (synthetic minority '
                        'oversampling technique).')
    parser.add_argument('--seed', type=int, help='Seed for the random number generator (use for reproducibility).')
    args = parser.parse_args()

    logging.info('started classify.py')
    test_data = load_test_data()
    test_data['features'] = scale(test_data['features'])
    train_data = test_data[~test_data['type'].mask]
    train_data['label'] = [classes.index(t) for t in train_data['type']]
    clf, sampler = train_classifier(train_data, n_est=args.estimators, depth=args.max_depth, max_feat=args.max_feat,
                                    n_jobs=args.jobs, sampler_type=args.sampler, random_state=args.seed)
    logging.info('classifier trained')

    p_class = clf.predict_proba(test_data['features'])
    write_results(test_data, p_class, 'results.txt')

    cnf_matrix = validate_classifier(clf, sampler, train_data)
    logging.info('validation complete')
    logging.info('finished classify.py')
