import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, hstack, join
from astropy.stats import mad_std, sigma_clip
import logging

filter_colors = {'g': '#00CCFF', 'r': '#FF7D00', 'i': '#90002C', 'z': '#000000', 'y': 'y',
                 'U': '#3C0072', 'B': '#0057FF', 'V': '#79FF00', 'R': '#FF7000', 'I': '#66000B'}
meta_columns = ['filename', 'type', 'MWEBV', 'redshift', 'prediction', 'confidence']
CLASS_KEYWORDS = {'type': 'Spectroscopic Classification', 'prediction': 'Photometric Classification'}
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True


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
    if 'type' in t.colnames:
        t['type'] = np.ma.array(t['type'])
    stored = np.load(data_file)
    meta_keys = set(stored.keys()) & {'filters', 'ndraws', 'paramnames', 'featnames'}
    column_keys = set(stored.keys()) - meta_keys
    t_stored = Table({key: stored[key] for key in column_keys})
    if set(t.colnames) & column_keys:
        data_table = join(t, t_stored)
    else:
        data_table = hstack([t[np.repeat(np.arange(len(t)), stored['ndraws'])], t_stored])
    for col in data_table.colnames:
        if data_table[col].dtype.type is np.str_:
            data_table[col].fill_value = ''
    for key in meta_keys:
        data_table.meta[key] = stored[key]
    logging.info(f'data loaded from {meta_file} and {data_file}')
    return data_table


def subplots_layout(n):
    """
    Calculate the number of rows and columns for a multi-panel plot, staying as close to a square as possible.

    Parameters
    ----------
    n : int
        The number of subplots required.

    Returns
    -------
    nrows, ncols : int
        The number of rows and columns in the layout.
    """
    nrows = round(n ** 0.5)
    ncols = -(-n // nrows)
    return nrows, ncols


def plot_histograms(data_table, colname, class_kwd='type', var_kwd=None, row_kwd=None, saveto=None):
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
    var_kwd : str, optional
        Keyword in `data_table.meta` containing the parameter names to list on the x-axes. Default: no labels.
    row_kwd : str, optional
        Keyword in `data_table.meta` containing labels for the leftmost y-axes.
    saveto : str, optional
        Filename to which to save the plot. Default: display the plot instead of saving it.
    """
    _, nrows, ncols = data_table[colname].shape
    if class_kwd:
        data_table = data_table.group_by(class_kwd)
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
                xlims.append(b)
                ylims.append(n)
            axarr[i, j].set_ylim(0., 1.05 * np.max(ylims))
            axarr[i, j].set_yticks([])
        xlims = sigma_clip(xlims, stdfunc=mad_std, masked=False)
        xmin, xmax = np.percentile(xlims, (0., 100.))
        axarr[-1, j].set_xlim(1.05 * xmin - 0.05 * xmax, 1.05 * xmax - 0.05 * xmin)
        axarr[-1, j].xaxis.set_major_locator(plt.MaxNLocator(2))
    if class_kwd:
        fig.legend(data_table.groups.keys['patch'], data_table.groups.keys[class_kwd], loc='upper center', ncol=ngroups,
                   title=CLASS_KEYWORDS.get(class_kwd, class_kwd))
    if var_kwd is not None:
        for ax, var in zip(axarr[-1], data_table.meta[var_kwd]):
            ax.set_xlabel(var, size='small')
            ax.tick_params(labelsize='small')
            if 'mag' in var.lower():
                ax.invert_xaxis()
    if row_kwd is not None:
        for ax, filt in zip(axarr[:, 0], data_table.meta[row_kwd]):
            ax.set_ylabel(filt, rotation=0, va='center')
    fig.tight_layout(h_pad=0., w_pad=0., rect=(0., 0., 1., 0.9 if class_kwd else 1.))
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
    plt.close(fig)
