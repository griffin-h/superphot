import numpy as np
from astropy.stats import mad_std
from astropy.table import Table


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
        Table of light curve data, with columns MJD, FLT, FIELD, FLUXCAL, FLUXCALERR, MAG, MAGERR, and FLTID.  The
        `.meta` attribute contains SURVEY, SNID, IAUC, RA, DECL, MWEBV, REDSHIFT_FINAL, SEARCH_PEAKMJD, FILTERS, NOBS,
        and NVAR, all stored as strings, as well as REDSHIFT and A_V converted to floats.
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
    t['FLTID'] = ['griz'.index(flt) for flt in t['FLT']]
    t.meta['REDSHIFT'] = float(t.meta['REDSHIFT_FINAL'].split()[0])
    t.meta['A_V'] = float(t.meta['MWEBV'].split()[0]) * 3.1
    t.meta['PEAKMJD'] = float(t.meta['SEARCH_PEAKMJD'])
    return t


def filter_func(t, fltr):
    """
    Make an Astropy table containing data only from one filter.

    Parameters
    ----------
    t : astropy.table.Table
        Astropy table containing the light curve data.
    fltr: int
        Integer 0-3, corresponding to the filters g, r, i, z.

    Returns
    -------
    t_filt : astropy.table.Table
        Astropy table containing data from only one filter.
    """
    t_filt = t[t['FLTID'] == fltr]
    return t_filt


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


def light_curve_event_data(file, fltr, period=180., nsigma=20.):
    """
    Make a 2-D array containing the time, flux, and flux uncertainties data only from the period containing the peak
    flux, with outliers cut, and in a single filter .

    Parameters
    ----------
    file : str
        Path to .snana.dat file containing the light curve data.
    period: float
        Include only points within `period` of SEARCH_PEAKMJD.
    nsigma: float
        Determines at what value (flux < nsigma * mad_std) to cut outlier data points.
    fltr: int
        Integer 0-3, corresponding to the filters g, r, i, z.

    Returns
    -------
    t_event : astropy.table.Table
        Astropy table containing the reduced light curve data which will be used to extract parameters for our model.

    """
    t = read_snana(file)
    peak_time = float(t.meta['PEAKMJD'])
    t['PHASE'] = t['MJD'] - peak_time
    t_event = t[np.abs(t['PHASE']) < period]
    t_cut = cut_outliers(t_event, nsigma)
    t_filt = filter_func(t_cut, fltr)
    return t_filt
