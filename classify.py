#!/usr/bin/env python

# FREDERICK DAUPHIN
# DEPARTMENT OF PHYSICS, CARNEGIE MELLON UNIVERSITY
# ADVISOR: DR. GRIFFIN HOSSEINZADEH
# MENTOR: PROF. EDO BERGER
# CENTER FOR ASTROPHYSICS | HARVARD & SMITHSONIAN
# REU 2019 INTERN PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import logging
from astropy.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import itertools

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
classes = ['SLSNe', 'SNII', 'SNIIn', 'SNIa', 'SNIbc']


def plot_confusion_matrix(cm, normalize=False, title='Confusion Matrix', cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From tutorial: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(6., 6.))
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal', origin='upper')
    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_norm.pdf' if normalize else 'confusion_matrix.pdf')


def train_classifier(data, n_est, depth=None, max_feat=None):
    """
    Make a random forest classifier using synthetic minority oversampling technique and kfolding. A lot of the
    documentation is taken directly from SciKit Learn page and should be referenced for further questions.

    Parameters
    ----------
    data : astropy.table.Table
        Astropy data table with a 'features' column and a 'type' column.
    n_est: int
        The number of trees in the forest.
    depth : int, optional
        The maxiumum depth of a tree. If None, the tree will have all pure leaves.
    max_feat : int, optional
        The maximum number of used before making a split. If None, use all features.

    Returns
    -------
    clf : RandomForestClassifier
        A random forest classifier trained from the classified transients.
    """
    labels = np.array([classes.index(t) for t in data['type']])
    labels_test = np.empty_like(labels)
    kf = KFold(len(np.unique(data['id'])))
    clf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, class_weight='balanced',
                                 criterion='entropy', max_features=max_feat)
    sampler = SMOTE()

    for i, (train_index, test_index) in enumerate(kf.split(data)):
        features_resamp, labels_resamp = sampler.fit_resample(data['features'][train_index], labels[train_index])
        clf.fit(features_resamp, labels_resamp)
        labels_test[test_index] = clf.predict(data['features'][test_index])
        logging.info(f'completed fold {i+1:d}/{kf.n_splits:d} of cross-validation')

    cnf_matrix = confusion_matrix(labels, labels_test)
    plot_confusion_matrix(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, normalize=True)
    return clf


def load_test_data():
    test_table = Table.read('test_data.txt', format='ascii.fixed_width')
    test_table['features'] = np.load('test_data.npz')['features']
    logging.info('test data loaded from test_data.txt and test_data.npz')
    return test_table


if __name__ == '__main__':
    logging.info('started classify.py')
    test_data = load_test_data()
    train_data = test_data[~test_data['type'].mask]
    clf = train_classifier(train_data, n_est=100)
    logging.info('classifier trained')

    classid_test = clf.predict(test_data['features'])
    meta_columns = ['id', 'hostz', 'type']
    test_data.keep_columns(meta_columns)
    test_data['type'].fill_value = ''
    for i, classname in enumerate(classes):
        test_data[classname] = classid_test == i
    grouped = test_data.filled().group_by(meta_columns)
    output = grouped.groups.aggregate(np.mean)
    output.write('results.txt', format='ascii.fixed_width')
    logging.info('finished classify.py')
