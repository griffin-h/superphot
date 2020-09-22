import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import logging
from astropy.table import Table, join
from astropy.io.ascii import masked
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.inspection import permutation_importance
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils._docstring import Substitution, _random_state_docstring
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import pickle
from .util import meta_columns, plot_histograms, filter_colors, load_data, CLASS_KEYWORDS
import itertools
from tqdm import tqdm
from argparse import ArgumentParser
import json

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


def plot_confusion_matrix(confusion_matrix, classes, cmap='Blues', purity=False, title='',
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
    cmap : str, optional
        Name of a Matplotlib colormap to color the matrix.
    purity : bool, optional
        If False (default), aggregate by row (spec. class). If True, aggregate by column (phot. class).
    title : str, optional
        Text to go above the plot. Default: no title.
    xlabel, ylabel : str, optional
        Labels for the x- and y-axes. Default: "Spectroscopic Classification" and "Photometric Classification".
    ax : matplotlib.pyplot.axes, optional
        Axis on which to plot the confusion matrix. Default: new axis.
    """
    n_per_true_class = confusion_matrix.sum(axis=1)
    n_per_pred_class = confusion_matrix.sum(axis=0)
    if purity:
        cm = confusion_matrix / n_per_pred_class[np.newaxis, :]
    else:
        cm = confusion_matrix / n_per_true_class[:, np.newaxis]
    if ax is None:
        ax = plt.axes()
    ax.imshow(cm, interpolation='nearest', cmap=cmap, aspect='equal')
    ax.set_title(title)
    nclasses = len(classes)
    ax.set_xticks(range(nclasses))
    ax.set_yticks(range(nclasses))
    ax.set_xticklabels(['{}\n({:.0f})'.format(label, n) for label, n in zip(classes, n_per_pred_class)])
    ax.set_yticklabels(['{}\n({:.0f})'.format(label, n) for label, n in zip(classes, n_per_true_class)])
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.set_ylim(nclasses - 0.5, -0.5)

    thresh = np.nanmax(cm) / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f'{cm[i, j]:.2f}\n({confusion_matrix[i, j]:.0f})', ha="center", va="center",
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
        self.mean_ = dict()
        self.cov_ = dict()
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
            self.mean_[class_sample] = np.mean(X_class, axis=0)
            self.cov_[class_sample] = np.cov(X_class, rowvar=False)
            if self.samples_per_class is not None:
                n_samples = self.samples_per_class - X_class.shape[0]
            if n_samples <= 0:
                continue

            self.rs_ = check_random_state(self.random_state)
            X_new = self.rs_.multivariate_normal(self.mean_[class_sample], self.cov_[class_sample], n_samples)
            y_new = np.repeat(class_sample, n_samples)

            X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

    def more_samples(self, n_samples):
        """Draw more samples from the same distribution of an already fitted sampler."""
        if not self.mean_ or not self.cov_:
            raise Exception('Mean and covariance not set. You must first run fit_resample(X, y).')
        classes = sorted(self.sampling_strategy_.keys())
        X = np.vstack([self.rs_.multivariate_normal(self.mean_[class_sample], self.cov_[class_sample], n_samples)
                       for class_sample in classes])
        y = np.repeat(classes, n_samples)
        return X, y


def train_classifier(pipeline, train_data):
    """
    Train a classification pipeline on `test_data`.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline
        The full classification pipeline, including rescaling, resampling, and classification.
    train_data : astropy.table.Table
        Astropy table containing the test data. Must have a 'features' and a 'type' column.
    """
    pipeline.fit(train_data['features'].reshape(len(train_data), -1), train_data['type'])


def classify(pipeline, test_data, aggregate=True):
    """
    Use a trained classification pipeline to classify `test_data`.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline
        The full classification pipeline, including rescaling, resampling, and classification.
    test_data : astropy.table.Table
        Astropy table containing the test data. Must have a 'features' column.
    aggregate : bool, optional
        If True (default), average the probabilities for a given supernova across the multiple model light curves.

    Returns
    -------
    results : astropy.table.Table
        Astropy table containing the supernova metadata and classification probabilities for each supernova
    """
    results = test_data.copy()
    results['probabilities'] = pipeline.predict_proba(results['features'].reshape(len(results), -1))
    if aggregate:
        results = aggregate_probabilities(results)
    results['prediction'] = pipeline.classes_[results['probabilities'].argmax(axis=1)]
    results['confidence'] = results['probabilities'].max(axis=1)
    return results


def mean_axis0(x, axis=0):
    """Equivalent to the numpy.mean function but with axis=0 by default."""
    return x.mean(axis=axis)


def aggregate_probabilities(table):
    """
    Average the classification probabilities for a given supernova across the multiple model light curves.

    Parameters
    ----------
    table : astropy.table.Table
        Astropy table containing the metadata for a supernova and the classification probabilities ('probabilities')

    Returns
    -------
    results : astropy.table.Table
        Astropy table containing the supernova metadata and average classification probabilities for each supernova
    """
    table = table[[col for col in table.colnames if col in meta_columns] + ['probabilities']]
    grouped = table.filled().group_by(table.colnames[:-1])
    results = grouped.groups.aggregate(mean_axis0)
    if 'type' in results.colnames:
        results['type'] = np.ma.array(results['type'])
    return results


def validate_classifier(pipeline, train_data, test_data=None, aggregate=True):
    """
    Validate the performance of a machine-learning classifier using leave-one-out cross-validation.

    Parameters
    ----------
    pipeline : imblearn.pipeline.Pipeline
        The full classification pipeline, including rescaling, resampling, and classification.
    train_data : astropy.table.Table
        Astropy table containing the training data. Must have a 'features' column and a 'type' column.
    test_data : astropy.table.Table, optional
        Astropy table containing the test data. Must have a 'features' column to which to apply the trained classifier.
        If None, use the training data itself for validation.
    aggregate : bool, optional
        If True (default), average the probabilities for a given supernova across the multiple model light curves.

    Returns
    -------
    results : astropy.table.Table
        Astropy table containing the supernova metadata and classification probabilities for each supernova
    """
    classes, n_per_class = np.unique(train_data['type'], return_counts=True)
    if np.any(n_per_class <= train_data.meta['ndraws']):
        raise ValueError('Training data must have at least two samples per class for cross-validation')
    if test_data is None:
        test_data = train_data
    train_classifier(pipeline, train_data)
    test_data['probabilities'] = pipeline.predict_proba(test_data['features'].reshape(len(test_data), -1))
    for filename in tqdm(np.unique(train_data['filename']), desc='Cross-validation'):
        train_index = train_data['filename'] != filename
        test_index = test_data['filename'] == filename
        train_classifier(pipeline, train_data[train_index])
        test_features = test_data['features'][test_index].reshape(np.count_nonzero(test_index), -1)
        test_data['probabilities'][test_index] = pipeline.predict_proba(test_features)
    if aggregate:
        test_data = aggregate_probabilities(test_data)
    test_data['prediction'] = pipeline.classes_[test_data['probabilities'].argmax(axis=1)]
    test_data['confidence'] = test_data['probabilities'].max(axis=1)
    test_data['correct'] = test_data['prediction'] == test_data['type']
    return test_data


def make_confusion_matrix(results, classes=None, p_min=0., saveto=None, purity=False, binary=False, title=None):
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
    binary : bool, optional
        If True, plot a SNIa vs non-SNIa (CCSN) binary confusion matrix.
    title : str, optional
        A title for the plot. If the plot is big enough, statistics ($N$, $A$, $F_1$) are appended in parentheses.
        Default: 'Completeness' or 'Purity' depending on `purity`.
    """
    results = results[~results['type'].mask]
    if classes is None:
        classes = np.unique(results['type'])
    if binary:
        results['type'] = ['SNIa' if sntype == 'SNIa' else 'CCSN' for sntype in results['type']]
        SNIa_probs = results['probabilities'][:, np.where(classes == 'SNIa')[0][0]]
        classes = np.array(['CCSN', 'SNIa'])
        predicted_types = np.choose(np.round(SNIa_probs).astype(int), classes)
        include = (SNIa_probs > p_min) | (SNIa_probs < 1. - p_min)
    else:
        predicted_types = classes[np.argmax(results['probabilities'], axis=1)]
        include = results['probabilities'].max(axis=1) > p_min
    cnf_matrix = confusion_matrix(results['type'][include], predicted_types[include])
    if title is None:
        title = 'Purity' if purity else 'Completeness'
    size = (len(classes) + 1.) * 5. / 6.
    if size > 3.:  # only add stats to title if figure is big enough
        accuracy = accuracy_score(results['type'][include], predicted_types[include])
        f1 = f1_score(results['type'][include], predicted_types[include], average='macro')
        title += f' ($N={include.sum():d}$, $A={accuracy:.2f}$, $F_1={f1:.2f}$)'
        xlabel = 'Photometric Classification'
        ylabel = 'Spectroscopic Classification'
    else:
        xlabel = 'Phot. Class.'
        ylabel = 'Spec. Class.'
    fig = plt.figure(figsize=(size, size))
    plot_confusion_matrix(cnf_matrix, classes, purity=purity, title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)


def load_results(filename):
    results = Table.read(filename, format='ascii')
    if 'type' in results.colnames:
        results['type'] = np.ma.array(results['type'])
    classes = np.array([col for col in results.colnames if col not in meta_columns])
    results['probabilities'] = np.stack([results[sntype].data for sntype in classes]).T
    results.meta['classes'] = classes
    results.remove_columns(classes)
    return results


def write_results(test_data, classes, filename, max_lines=None, latex=False, latex_title='Classification Results',
                  latex_label='tab:results'):
    """
    Write the classification results to a text file.

    Parameters
    ----------
    test_data : astropy.table.Table
        Astropy table containing the supernova metadata and the classification probabilities ('probabilities').
    classes : list
        The labels that correspond to the columns in 'probabilities'
    filename : str
        Name of the output file
    max_lines : int, optional
        Maximum number of table rows to write to the file
    latex : bool, optional
        If False (default), write in the Astropy 'ascii.fixed_width_two_line' format. If True, write in the Astropy
        'ascii.aastex' format and add fancy table headers, etc.
    latex_title : str, optional
        Table caption if written in AASTeX format. Default: 'Classification Results'
    latex_label : str, optional
        LaTeX label if written in AASTeX format. Default: 'tab:results'
    """
    test_data = test_data[:max_lines]
    output = test_data[[col for col in test_data.colnames if col in meta_columns]]
    output['MWEBV'].format = '%.4f'
    output['redshift'].format = '%.4f'
    output['confidence'].format = '%.3f'
    for i, classname in enumerate(classes):
        col = f'$p_\\mathrm{{{classname}}}$' if latex else classname
        output[col] = test_data['probabilities'][:, i]
        output[col].format = '%.3f'
    if latex:
        # latex formatting for data
        output['filename'] = [name.replace('_', '\\_') for name in output['filename']]
        if 'type' in output.colnames:
            output['type'] = [classname.replace("SNI", "SN~I") for classname in output['type']]
        output['prediction'] = [classname.replace("SNI", "SN~I") for classname in output['prediction']]

        # AASTeX header and footer
        latexdict = {'tabletype': 'deluxetable*'}
        if latex_title:
            if latex_label:
                latex_title += f'\\label{{{latex_label}}}'
            latexdict['caption'] = latex_title
        if max_lines is not None:
            latexdict['tablefoot'] = '\\tablecomments{The full table is available in machine-readable form.}'

        # human-readable column headers
        column_headers = {
            'filename': 'Transient Name',
            'redshift': 'Redshift',
            'MWEBV': '$E(B-V)$',
            'type': 'Spec. Class.',
            'prediction': 'Phot. Class.',
            'confidence': 'Confidence',
        }
        for column in output.colnames:
            output.rename_column(column, column_headers.get(column, column))

        output.write(filename, format='ascii.aastex', overwrite=True, latexdict=latexdict,
                     fill_values=[(masked, '\\nodata'), ('inf', '\\infty')])
    else:
        output.write(filename, format='ascii.fixed_width_two_line', overwrite=True)
    logging.info(f'classification results saved to {filename}')


def plot_feature_importance(pipeline, train_data, width=0.8, nsamples=1000, saveto=None):
    """
    Plot a bar chart of feature importance using mean decrease in impurity, with permutation importances overplotted.

    Mean decrease in impurity is assumed to be stored in `pipeline.feature_importances_`. If the classifier does not
    have this attribute (e.g., SVM, MLP), only permutation importance is calculated.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
        The trained pipeline for which to plot feature importances. Steps should be named 'classifier' and 'sampler'.
    train_data : astropy.table.Table
        Data table containing 'features' and 'type' for training when calculating permutation importances. Must also
        include 'featnames' and 'filters' in `train_data.meta`.
    width : float, optional
        Total width of the bars in units of the separation between bars. Default: 0.8.
    nsamples : int, optional
        Number of samples to draw for the fake validation data set. Default: 1000.
    saveto : str, optional
        Filename to which to save the plot. Default: show instead of saving.
    """
    logging.info('calculating feature importance')
    featnames = train_data.meta['featnames']
    filters = train_data.meta['filters']

    featnames = np.append(featnames, 'Random Number')
    xoff = 0.5 * width / filters.size * np.linspace(1 - filters.size, filters.size - 1, filters.size)
    xranges = np.arange(featnames.size) + xoff[:, np.newaxis]
    random_feature_train = np.random.random(len(train_data))
    random_feature_validate = np.random.random(nsamples * pipeline.classes_.size)
    has_mdi = hasattr(pipeline.named_steps['classifier'], 'feature_importances_')
    fig, ax = plt.subplots(1, 1)
    for real_features, xrange, fltr in zip(np.moveaxis(train_data['features'], 1, 0), xranges, filters):
        X = np.hstack([real_features, random_feature_train[:, np.newaxis]])
        pipeline.fit(X, train_data['type'])

        X_val, y_val = pipeline.named_steps['sampler'].more_samples(nsamples)
        X_val[:, -1] = random_feature_validate
        result = permutation_importance(pipeline.named_steps['classifier'], X_val, y_val, n_jobs=-1)
        importance = result.importances_mean
        std = result.importances_std

        c = filter_colors.get(fltr)
        if has_mdi:
            importance0 = pipeline.named_steps['classifier'].feature_importances_
            ax.barh(xrange[:-1], importance0[:-1], width / filters.size, color=c)
        ax.errorbar(importance, xrange, xerr=std, fmt='o', color=c, mfc='w')

    if has_mdi:
        proxy_artists = [Patch(color='gray'), ax.errorbar([], [], xerr=[], fmt='o', color='gray', mfc='w')]
        ax.legend(proxy_artists, ['Mean Decrease in Impurity', 'Permutation Importance'], loc='best')
    for i, featname in enumerate(featnames):
        ax.text(-0.03, i, featname, ha='right', va='center', transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_yticks(xranges.flatten(), minor=True)
    ax.set_yticklabels(np.repeat(filters, featnames.size), minor=True, size='x-small', ha='center')
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_xlim(0., ax.get_xlim()[1])
    fig.tight_layout()
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
    plt.close(fig)


def cumhist(data, reverse=False, mark=None, ax=None, **kwargs):
    """
    Plot a cumulative histogram of `data`, optionally with certain indices marked with an x.

    Parameters
    ----------
    data : array-like
        Data to include in the histogram
    reverse : bool, optional
        If False (default), the histogram increases with increasing `data`. If True, it decreases with increasing `data`
    mark : array-like, optional
        An array of indices to mark with an x
    ax : matplotlib.pyplot.axes, optional
        Axis on which to plot the confusion matrix. Default: current axis.
    kwargs : dict, optional
        Keyword arguments to be passed to `matplotlib.pyplot.step`

    Returns
    -------
    p : list
        The list of `matplotlib.lines.Line2D` objects returned by `matplotlib.pyplot.step`
    """
    if mark is None:
        mark = np.zeros(len(data), bool)
    if ax is None:
        ax = plt.gca()
    i = np.argsort(data)
    x = data[i]
    mark = mark[i]
    x = np.append(x, x[-1])
    y = np.linspace(0., 1., x.size)
    if reverse:
        y = y[::-1]
    p = ax.step(x, y, **kwargs)
    ax.scatter(data[i][mark], (y[:-1] + 0.5 * np.diff(y))[mark], marker='x')
    return p


def plot_results_by_number(results, xval='confidence', class_kwd='prediction', title=None, saveto=None):
    """
    Plot cumulative histograms of the results for each class against a specified table column.

    If `results` contains the column 'correct', incorrect classifications will be marked with an x.

    Parameters
    ----------
    results : astropy.table.Table
        Table of classification results. Must contain columns 'type'/'prediction' and the column specified by `xval`.
    xval : str, optional
        Table column to use as the horizontal axis of the histogram. Default: 'confidence'.
    class_kwd : str, optional
        Table column to use as the class grouping. Default: 'prediction'.
    title : str, optional
        Title for the plot. Default: "Training/Test Set, Grouped by {class_kwd}", where the first word is determined
        by the presence of the column 'correct' in `results`.
    saveto : str, optional
        Save the plot to this filename. If None, the plot is displayed and not saved.
    """
    if 'correct' in results.colnames:
        label = '{ngroup:d} {snclass}, {correct:d} correct'
        if title is None:
            title = f'Training Set, Grouped by {CLASS_KEYWORDS.get(class_kwd, class_kwd)}'
    else:
        label = '{ngroup:d} {snclass}'
        if title is None:
            title = f'Test Set, Grouped by {CLASS_KEYWORDS.get(class_kwd, class_kwd)}'
    grouped = results.group_by(class_kwd)

    fig = plt.figure()
    for group, snclass in zip(grouped.groups, grouped.groups.keys[class_kwd]):
        correct = group['correct'] if 'correct' in results.colnames else np.ones(len(group), bool)
        grouplabel = label.format(snclass=snclass, ngroup=len(group), correct=correct.sum())
        cumhist(group[xval], lw=1, label=grouplabel, mark=~correct)
    if 'correct' in results.colnames:
        plt.plot([], [], 'kx', label='incorrect')
    plt.legend(loc='best', frameon=False)
    plt.tick_params(labelsize='large')
    plt.xlabel(xval.title(), size='large')
    plt.ylabel('Cumulative Fraction', size='large')
    plt.ylim(0, 1)
    plt.title(title)
    fig.tight_layout()
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)


def calc_metrics(results, param_set, save=False):
    """
    Calculate completeness, purity, accuracy, and F1 score for a table of validation results.

    The metrics are returned in a dictionary and saved in a json file.

    Parameters
    ----------
    results : astropy.table.Table
        Astropy table containing the results. Must have columns 'type' and 'prediction'.
    param_set : dict
        A dictionary containing metadata to store along with the metrics.
    save : bool, optional
        If True, save the results to a json file in addition to returning the results. Default: False

    Returns
    -------
    param_set : dict
        A dictionary containing the input metadata and the calculated metrics.
    """
    param_names = sorted(param_set.keys())
    classes = results.meta.get('classes', np.unique(results['prediction']))

    cnf_matrix = confusion_matrix(results['type'], results['prediction'], labels=classes)
    correct = np.diag(cnf_matrix)
    n_per_spec_class = cnf_matrix.sum(axis=1)
    n_per_phot_class = cnf_matrix.sum(axis=0)
    param_set['completeness'] = list(correct / n_per_spec_class)
    param_set['purity'] = list(correct / n_per_phot_class)
    param_set['accuracy'] = accuracy_score(results['type'], results['prediction'])
    param_set['f1_score'] = f1_score(results['type'], results['prediction'], average='macro', labels=classes)

    if save:
        filename = '_'.join([str(param_set[key]) for key in param_names]) + '.json'
        with open(filename, 'w') as f:
            json.dump(param_set, f)
    return param_set


def plot_metrics_by_number(validation, xval='confidence', classes=None, saveto=None):
    """
    Plot completeness, purity, accuracy, F1 score, and fractions remaining as a function of confidence threshold.

    Parameters
    ----------
    validation : astropy.table.Table
        Astropy table containing the results. Must have columns 'type', 'prediction', 'probabilities', and 'confidence'.
    xval : str, optional
        Table column to use as the horizontal axis of the plot. Default: 'confidence'.
    classes : array-like, optional
        The classes for which to calculate completeness and purity. Default: all classes in the 'type' column.
    saveto : str, optional
        Save the plot to this filename. If None, the plot is displayed and not saved.
    """
    if classes is None:
        classes = np.unique(validation['type'])
    validation.sort(xval)
    metrics = Table([calc_metrics(validation[i:], {xval: validation[xval][i]}) for i in range(len(validation))])
    ccsne = validation[validation['type'] != 'SNIa']

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True, figsize=(6., 8.))
    lines1 = ax1.step(metrics[xval], metrics['completeness'])
    ax2.step(metrics[xval], metrics['purity'])
    for ax in [ax1, ax2]:
        lines2 = ax.step(metrics[xval], metrics['accuracy'], 'k-')
        lines2 += ax.step(metrics[xval], metrics['f1_score'], 'k--')
        lines2 += cumhist(validation[xval], reverse=True, ax=ax, color='k', ls='-.')
        lines2 += cumhist(ccsne[xval], reverse=True, ax=ax, color='k', ls=':')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize='large')
    fig.legend(lines2, ['accuracy', '$F_1$', 'frac.', 'CCSN frac.'], ncol=len(lines2), loc='upper center',
               bbox_to_anchor=(0.5, 0.975), frameon=False)
    ax1.legend(lines1, classes, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False)
    ax1.set_ylabel('Completeness', size='large')
    ax2.set_ylabel('Purity', size='large')
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(f'Minimum {xval}', size='large')
    fig.tight_layout()
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)


def bar_plot(vresults, tresults, saveto=None):
    """
    Make a stacked bar plot showing the class breakdown in the training set compared to the test set.

    Parameters
    ----------
    vresults : astropy.table.Table
        Astropy table containing the training data. Must have a 'type' column and a 'prediction' column.
    tresults : astropy.table.Table
        Astropy table containing the test data. Must have a 'prediction' column.
    saveto : str, optional
        Save the plot to this filename. If None, the plot is displayed and not saved.
    """
    labels, n_per_true_class = np.unique(vresults['type'], return_counts=True)
    labels_pred, n_per_pred_class = np.unique(tresults['prediction'], return_counts=True)
    if np.any(labels_pred != labels):
        raise ValueError('photometric and spectroscopic class labels do not match')
    purity = confusion_matrix(vresults['type'], vresults['prediction'], normalize='pred')
    corrected = purity @ n_per_pred_class

    names = ['Spectroscopically\nClassified', 'Photometrically\nClassified', 'Phot. Class.\n(Corrected)']
    rawcounts = np.transpose([n_per_true_class, n_per_pred_class, corrected])
    fractions = rawcounts / rawcounts.sum(axis=0)
    cumulative_fractions = fractions.cumsum(axis=0)

    fig = plt.figure()
    ax = plt.axes()
    for counts, fracs, cumfracs, label in zip(rawcounts, fractions, cumulative_fractions, labels):
        plt.bar(names, -fracs, bottom=cumfracs)
        heights = cumfracs - fracs / 2.
        for count, frac, name, height in zip(counts, fracs, names, heights):
            if height < 0.05:
                h = 0.005
                va = 'top'
            elif height > 0.95:
                h = 0.995
                va = 'bottom'
            else:
                h = height
                va = 'center'
            ax.text(name, h, f'{frac:.2f} ({count:.0f})', ha='center', va=va, color='w')
        ax.text(2.5, heights[-1], label, ha='left', va='center')
    ax.set_ylim(1., 0.)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False,
                   labeltop=True, labelsize='large')
    fig.tight_layout()
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)


def _bar_plot_from_file():
    parser = ArgumentParser()
    parser.add_argument('validation_results', help='Filename of the validation results.')
    parser.add_argument('test_results', help='Filename of the classification results.')
    parser.add_argument('--saveto', help='Filename to which to save the bar plot.')
    args = parser.parse_args()

    vresults = load_results(args.validation_results)
    tresults = load_results(args.test_results)
    bar_plot(vresults, tresults, args.saveto)


def _plot_confusion_matrix_from_file():
    parser = ArgumentParser()
    parser.add_argument('filename', type=str, help='Filename containing the table of classification results.')
    parser.add_argument('--pmin', type=float, default=0.,
                        help='Minimum confidence to be included in the confusion matrix.')
    parser.add_argument('--saveto', type=str, help='If provided, save the confusion matrix to this file.')
    parser.add_argument('--purity', action='store_true', help='Aggregate by column instead of by row.')
    parser.add_argument('--binary', action='store_true', help='Plot a SNIa vs non-SNIa (CCSN) binary confusion matrix.')
    args = parser.parse_args()

    results = load_results(args.filename)
    make_confusion_matrix(results, p_min=args.pmin, saveto=args.saveto, purity=args.purity, binary=args.binary)


def _train():
    parser = ArgumentParser()
    parser.add_argument('train_data', help='Filename of the metadata table for the training set.')
    parser.add_argument('--classifier', choices=['rf', 'svm', 'mlp'], default='rf', help='The classification algorithm '
                        'to use. Current choices are "rf" (random forest; default), "svm" (support vector machine), or '
                        '"mlp" (multilayer perceptron).')
    parser.add_argument('--sampler', choices=['mvg', 'smote'], default='mvg', help='The resampling algorithm to use. '
                        'Current choices are "mvg" (multivariate Gaussian; default) or "smote" (synthetic minority '
                        'oversampling technique).')
    parser.add_argument('--random-state', type=int, help='Seed for the random number generator (for reproducibility).')
    parser.add_argument('--output', default='pipeline.pickle',
                        help='Filename to which to save the pickled classification pipeline.')
    args = parser.parse_args()

    logging.info('started training')
    train_data = load_data(args.train_data)
    if train_data['type'].mask.any():
        raise ValueError('training data is missing values in the "type" column')

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

    pipeline = Pipeline([('scaler', StandardScaler()), ('sampler', sampler), ('classifier', clf)])
    train_classifier(pipeline, train_data)
    with open(args.output, 'wb') as f:
        pickle.dump(pipeline, f)
    logging.info('finished training')


def _classify():
    parser = ArgumentParser()
    parser.add_argument('pipeline', help='Filename of the pickled classification pipeline.')
    parser.add_argument('test_data', help='Filename of the metadata table for the test set.')
    parser.add_argument('--output', default='test_data', help='Filename (without extension) to save the results.')
    args = parser.parse_args()

    logging.info('started classification')
    with open(args.pipeline, 'rb') as f:
        pipeline = pickle.load(f)
    test_data = load_data(args.test_data)

    results = classify(pipeline, test_data)
    write_results(results, pipeline.classes_, f'{args.output}_results.txt')
    plot_results_by_number(results, saveto=f'{args.output}_confidence.pdf')

    test_data_full = join(test_data, results)
    plot_histograms(test_data_full, 'params', 'prediction', var_kwd='paramnames', row_kwd='filters',
                    saveto=f'{args.output}_parameters.pdf')
    plot_histograms(test_data_full, 'features', 'prediction', var_kwd='featnames', row_kwd='filters',
                    saveto=f'{args.output}_features.pdf')

    logging.info('finished classification')


def _validate_args(args):
    with open(args.pipeline, 'rb') as f:
        pipeline = pickle.load(f)

    validation_data = load_data(args.validation_data)
    if validation_data['type'].mask.any():
        raise ValueError('validation data is missing values in the "type" column')

    if args.train_data is None:
        train_data = validation_data
    else:
        train_data = load_data(args.train_data)
        if train_data['type'].mask.any():
            raise ValueError('training data is missing values in the "type" column')
    return pipeline, train_data, validation_data


def _validate():
    parser = ArgumentParser()
    parser.add_argument('pipeline', help='Filename of the pickled classification pipeline.')
    parser.add_argument('validation_data', help='Filename of the metadata table for the validation set.')
    parser.add_argument('--train-data', help='Filename of the metadata table for the training set, if different than '
                                             'the validation set.')
    parser.add_argument('--pmin', type=float, default=0.,
                        help='Minimum confidence to be included in the confusion matrix.')
    args = parser.parse_args()
    pipeline, train_data, validation_data = _validate_args(args)

    logging.info('started validation')
    plot_feature_importance(pipeline, train_data, saveto='feature_importance.pdf')

    results_validate = validate_classifier(pipeline, train_data, validation_data)
    write_results(results_validate, pipeline.classes_, 'validation_results.txt')
    make_confusion_matrix(results_validate, pipeline.classes_, args.pmin, 'confusion_matrix.pdf')
    make_confusion_matrix(results_validate, pipeline.classes_, args.pmin, 'confusion_matrix_purity.pdf', purity=True)
    plot_results_by_number(results_validate, class_kwd='type', saveto='validation_confidence_specclass.pdf')
    plot_results_by_number(results_validate, saveto='validation_confidence_photclass.pdf')
    plot_metrics_by_number(results_validate, classes=pipeline.classes_, saveto='threshold.pdf')
    logging.info('finished validation')


def _latex():
    parser = ArgumentParser()
    parser.add_argument('filename', help='Filename of the results to format into a LaTeX table')
    parser.add_argument('-m', '--max-lines', type=int, help='Maximum number of table rows to write')
    parser.add_argument('-t', '--title', default='Classification Results', help='Table caption')
    parser.add_argument('-l', '--label', default='tab:results', help='LaTeX label')
    args = parser.parse_args()

    results = load_results(args.filename)
    write_results(results, results.meta['classes'], args.filename.split('.')[0] + '.tex', max_lines=args.max_lines,
                  latex=True, latex_title=args.title, latex_label=args.label)
