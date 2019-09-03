import numpy as np
from astropy.stats import mad_std


def s2c(file):
    """
    Make a PS1_PS1MD_PSc file into a 2-D list.

    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.

    Returns
    -------
    lst : list
        2-D list containing the light curve data.

    """
    with open(file) as infile:
        lst = []
        for line in infile:
            lst.append(line.split())
    return lst


def filter_func(lst, fltr):
    """
    Make a 2-D list containing data only from one filter.

    Parameters
    ----------
    lst : list
        2-D list containing the light curve data.
    fltr: int
        Integer 0-3, corresponding to the filters g, r, i, z.

    Returns
    -------
    lst_new : list
        2-D list containing data from only one filter.

    """
    lst_fltr = []
    for row in lst:
        if row[2] == 'griz'[fltr]:
            lst_fltr.append(row)
    return lst_fltr


def cut_outliers(lst, multiplier):
    """
    Make a 2-D list containing only data that is below the cut off threshold.

    Parameters
    ----------
    lst : list
        2-D list containing the light curve data.
    multiplier: float
        Determines at what value (multiplier * mad_std) to cut outlier data points.

    Returns
    -------
    lst_cut : list
        2-D list containing only data that is below the cut off threshold.

    """
    lst_cut = []
    table = np.transpose(lst)
    if len(table) > 0:
        flux = table[4].astype(float)
        madstd = mad_std(flux)
        for row in lst:
            if float(row[4]) < multiplier * madstd:
                lst_cut.append(row)
    return lst_cut


def light_curve_event_data(file, period, multiplier, fltr):
    """
    Make a 2-D array containing the time, flux, and flux uncertainties data only from the period containing the peak
    flux, with outliers cut, and in a single filter .

    Parameters
    ----------
    file : path (.snana.dat file)
        File extention containing the light curve data.
    period: float
        Half the length of time from which you want to observe.
    multiplier: float
        Determines at what value (multiplier * mad_std) to cut outlier data points.
    fltr: int
        Integer 0-3, corresponding to the filters g, r, i, z.

    Returns
    -------
    lc_event_data : 2-D array
        2-D array containing the reduced light curve data which will be used to extract parameters for our model.

    """
    data = s2c(file)
    peak_time = float(data[7][1])
    event = peak_event(data[17:-1], period, peak_time)
    table = np.transpose(filter_func(cut_outliers(event, multiplier), fltr))
    if len(table) > 0:
        time = table[1].astype(float) - peak_time
        flux = table[4].astype(float)
        flux_unc = table[5].astype(float)
        lc_event_data = np.array([time, flux, flux_unc])
        return lc_event_data


def peak_event(lst, period, peak_time):
    """
    Make a 2-D list containing only data from the observational period with the peak flux.

    Parameters
    ----------
    lst : list
        2-D list containing the light curve data.
    period: float
        Half the length of time from which you want to observe.
    peak_time: float
        The estimated light curve peak time (which is also the discovery time).

    Returns
    -------
    lst_peak : list
        2-D list containing data from only the observational period with the peak flux.

    """
    lst_peak = []
    for row in lst:
        time_ij = float(row[1])
        if peak_time + period > time_ij > peak_time - period:
            lst_peak.append(row)
    return lst_peak


