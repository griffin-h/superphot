from astropy.stats import mad_std
from astropy.table import Table
import matplotlib.pyplot as plt

filter_colors = {'g': '#00CCFF', 'r': '#FF7D00', 'i': '#90002C', 'z': '#000000'}
meta_columns = ['filename', 'type', 'MWEBV', 'redshift']
# Default time range to use in fitting
PHASE_MIN = -50.
PHASE_MAX = 180.

plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True


def read_light_curve(filename):
    """
    Read light curve data from a text file as an Astropy table. SNANA files are recognized.

    Parameters
    ----------
    filename : str
        Path to light curve data file.

    Returns
    -------
    t : astropy.table.Table
        Table of light curve data.
    """
    with open(filename) as f:
        text = f.read()
    data_lines = []
    metadata = {}
    for line in text.splitlines():
        if ':' in line:
            key, val = line.split(':')
            if key in ['VARLIST', 'OBS']:
                data_lines.append(val)
            elif val:
                key0 = key.strip('# ')
                val0 = val.split()[0]
                try:
                    metadata[key0] = float(val0) if '.' in val0 else int(val0)
                except ValueError:
                    metadata[key0] = val0
        else:
            data_lines.append(line)
    t = Table.read(data_lines, format='ascii', fill_values=[('NULL', '0'), ('nan', '0'), ('', '0')])
    t.meta = metadata
    if 'REDSHIFT_FINAL' in t.meta and 'REDSHIFT' not in t.meta:
        t.meta['REDSHIFT'] = t.meta['REDSHIFT_FINAL']
    if 'SEARCH_PEAKMJD' in t.meta and 'PHASE' not in t.colnames:
        t['PHASE'] = t['MJD'] - t.meta['SEARCH_PEAKMJD']
    return t


def cut_outliers(t, nsigma):
    """
    Make an Astropy table containing only data that is below the cut off threshold.

    Parameters
    ----------
    t : astropy.table.Table
        Astropy table containing the light curve data.
    nsigma : float
        Determines at what value (flux < nsigma * mad_std) to cut outlier data points.

    Returns
    -------
    t_cut : astropy.table.Table
        Astropy table containing only data that is below the cut off threshold.

    """
    madstd = mad_std(t['FLUXCAL'])
    t_cut = t[t['FLUXCAL'] < nsigma * madstd]
    return t_cut


def select_event_data(t, phase_min=PHASE_MIN, phase_max=PHASE_MAX, nsigma=None):
    """
    Select data only from the period containing the peak flux, with outliers cut.

    Parameters
    ----------
    t : astropy.table.Table
        Astropy table containing the light curve data.
    phase_min, phase_max : float, optional
        Include only points within [`phase_min`, `phase_max`) days of SEARCH_PEAKMJD.
    nsigma : float, optional
        Determines at what value (flux < nsigma * mad_std) to reject outlier data points. Default: no rejection.

    Returns
    -------
    t_event : astropy.table.Table
        Table containing the reduced light curve data from the period containing the peak flux.
    """
    t_event = t[(t['PHASE'] >= phase_min) & (t['PHASE'] < phase_max)]
    if nsigma is not None:
        t_event = cut_outliers(t_event, nsigma)
    return t_event


def select_labeled_events(t, key='type'):
    """Returns rows from a table where the column `key` is not masked."""
    return t[~t.mask[key]] if t.has_masked_values else t.copy()
