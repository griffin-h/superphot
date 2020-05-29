import numpy as np
import matplotlib.pyplot as plt

filter_colors = {'g': '#00CCFF', 'r': '#FF7D00', 'i': '#90002C', 'z': '#000000', 'y': 'y',
                 'U': '#3C0072', 'B': '#0057FF', 'V': '#79FF00', 'R': '#FF7000', 'I': '#66000B'}
meta_columns = ['filename', 'type', 'MWEBV', 'redshift']
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True


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


def plot_histograms(data_table, colname, class_kwd='type', varnames=(), rownames=(), no_autoscale=(), saveto=None):
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
    varnames : iterable, optional
        Parameter names to list on the x-axes of the plot. Default: no labels.
    rownames : iterable, optional
        Labels for the leftmost y-axes.
    no_autoscale : tuple or list, optional
        Class names not to use in calculating the axis limits. Default: include all.
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
                if not class_kwd or data_table.groups.keys[class_kwd][k] not in no_autoscale:
                    xlims.append(b)
                    ylims.append(n)
            axarr[i, j].set_ylim(0., 1.05 * np.max(ylims))
            axarr[i, j].set_yticks([])
        xmin, xmax = np.percentile(xlims, (0., 100.))
        axarr[-1, j].set_xlim(1.05 * xmin - 0.05 * xmax, 1.05 * xmax - 0.05 * xmin)
        axarr[-1, j].xaxis.set_major_locator(plt.MaxNLocator(2))
    if class_kwd:
        fig.legend(data_table.groups.keys['patch'], data_table.groups.keys[class_kwd], loc='upper center', ncol=ngroups,
                   title={'type': 'Spectroscopic Class', 'prediction': 'Photometric Class'}.get(class_kwd, class_kwd))
    for ax, var in zip(axarr[-1], varnames):
        ax.set_xlabel(var, size='small')
        ax.tick_params(labelsize='small')
        if 'mag' in var.lower():
            ax.invert_xaxis()
    for ax, filt in zip(axarr[:, 0], rownames):
        ax.set_ylabel(filt, rotation=0, va='center')
    fig.tight_layout(h_pad=0., w_pad=0., rect=(0., 0., 1., 0.9 if class_kwd else 1.))
    if saveto is None:
        plt.show()
    else:
        fig.savefig(saveto)
    plt.close(fig)
