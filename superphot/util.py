from astropy.stats import mad_std
from astropy.table import Table
import pkg_resources
import os

filter_colors = {'g': '#00CCFF', 'r': '#FF7D00', 'i': '#90002C', 'z': '#000000'}
meta_columns = ['id', 'A_V', 'filename', 'redshift', 'type']


def read_snana(filename):
    """
    Read light curve data from a SNANA file as an Astropy table.

    Parameters
    ----------
    filename : str
        Path to SNANA file.

    Returns
    -------
    t : astropy.table.Table
        Table of light curve data, with columns MJD, FLT, FIELD, FLUXCAL, FLUXCALERR, MAG, and MAGERR.  The `.meta`
        attribute contains SURVEY, SNID, IAUC, RA, DECL, MWEBV, REDSHIFT_FINAL, SEARCH_PEAKMJD, FILTERS, NOBS, and NVAR,
        all stored as strings, as well as REDSHIFT, A_V, and PEAKMJD converted to floats.
    """
    meta = {}
    header_start = 14
    with open(filename) as f:
        for _ in range(15):
            line = f.readline()
            if ':' in line:
                key, val = line.split(':')
                meta[key] = val.strip()
    data_end = int(meta['NOBS']) + header_start + 1
    t = Table.read(filename, format='ascii', header_start=header_start, data_end=data_end, guess=False, comment=None,
                   exclude_names=['VARLIST:'], fill_values=[('NULL', '0'), ('nan', '0')])
    t.meta = meta
    t.meta['REDSHIFT'] = float(t.meta['REDSHIFT_FINAL'].split()[0])
    t.meta['A_V'] = float(t.meta['MWEBV'].split()[0]) * 3.1
    t.meta['PEAKMJD'] = float(t.meta['SEARCH_PEAKMJD'])
    return t


def cut_outliers(t, nsigma):
    """
    Make an Astropy table containing only data that is below the cut off threshold.

    Parameters
    ----------
    t : astropy.table.Table
        Astropy table containing the light curve data.
    nsigma: float
        Determines at what value (flux < nsigma * mad_std) to cut outlier data points.

    Returns
    -------
    t_cut : astropy.table.Table
        Astropy table containing only data that is below the cut off threshold.

    """
    madstd = mad_std(t['FLUXCAL'])
    t_cut = t[t['FLUXCAL'] < nsigma * madstd]
    return t_cut


def light_curve_event_data(file, phase_min=-50., phase_max=180., nsigma=None):
    """
    Make a 2-D array containing the time, flux, and flux uncertainties data only from the period containing the peak
    flux, with outliers cut.

    Parameters
    ----------
    file : str
        Path to .snana.dat file containing the light curve data.
    phase_min, phase_max : float, optional
        Include only points within [`phase_min`, `phase_max`] days of SEARCH_PEAKMJD (inclusive). Default: [-50., 180.].
    nsigma : float, optional
        Determines at what value (flux < nsigma * mad_std) to reject outlier data points. Default: no rejection.

    Returns
    -------
    t_event : astropy.table.Table
        Table containing the reduced light curve data from the period containing the peak flux.
    """
    t = read_snana(file)
    t['PHASE'] = t['MJD'] - t.meta['PEAKMJD']
    t_event = t[(t['PHASE'] >= phase_min) & (t['PHASE'] <= phase_max)]
    if nsigma is not None:
        t_event = cut_outliers(t_event, nsigma)
    return t_event


def get_VAV19(filename):
    """
    Locate one of the data tables from the paper by V. Ashley Villar et al. (2019)

    Parameters
    ----------
    filename : str
        Name of the saved data table

    Returns
    -------
    path : str
        Full path to the saved data table in the installed package
    """
    path = pkg_resources.resource_filename(__name__, os.path.join('VAV19', filename))
    return path
