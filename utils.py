#!/usr/bin/env python
"""
Collection of Python utility functions that I've found useful
for energy data analysis, mostly relating to time-series data
from power meters or provided by power utilities.

Contents:
 - calculate_3_phase_power
 - load_process_data
 - read_mv90_csv_file
 - convert_to_numeric
 - calculate_running_status
 - drop_values_when_not_running
 - combine_running_status
 - split_into_groups_of_values
 - groups_of_true_values
 - plot_daily_load
 - plot_daily_loads
 - filename_from_string
"""

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def calculate_3_phase_power(voltage, current, power_factor=0.95):
    """Returns electrical power (W) given voltage (V) and
    current (I) using the standard 3-phase power equation.
    Unless specified, power_factor is assumed to be 0.95.
    """

    return voltage*current*np.sqrt(3)*power_factor


def estimate_motor_FLA(capacity, voltage, units='kW', efficiency=0.95,
                       three_phase=True):
    """Returns an estimate of full-load-amps for an electric
    motor of given capacity (kW), voltage (V).  In reality,
    FLA is dependent on motor size, design and condition so
    only use this estimate if you do not have a manufacturer's
    estimate.
    """

    if units.lower() == 'kw':
        capacity_kw = capacity
    elif units.lower() == 'hp':
        capacity_kw = capacity*0.7457
    elif units.lower() == 'mw':
        capacity_kw = capacity*1e3
    elif units.lower() == 'w':
        capacity_kw = capacity*1e-3
    else:
        raise ValueError("Motor capacity units not valid.")

    # Adjustment factor, calibrated to produce 72 A for a
    # 4160 V, 600 hp  motor
    #adj = 1.1
    adj = 1.052  # Using this for current project

    if three_phase is True:
        return capacity_kw*1000/(voltage*np.sqrt(3))/efficiency*adj
    else:
        return capacity_kw*1000/voltage/efficiency*adj


def load_process_data(filename, path=None, sheet_name=0, header=0,
                      usecols=None, dtype=None, index_col=0,
                      date_col=None, index_name='Date/Time',
                      first_data_row=0, first_data_col=1,
                      rename_map=None, dt_rounding=None, show=True):
    """Loads time-series data from an excel workbook.

    Args:
        filename (str): Filename.
        path (str): Path to directory.
        sheet_name (str): Excel sheet to load.
        header (int): Row (0-indexed) to use for column labels.
        index_col (int): Column (0-indexed) to use as row labels.
        date_col (str): Column name containing dates to use as row labels.
        index_name (str): Name to assign to index.
        first_data_row (int): Use to skip some rows.
        first_data_col (int): Use to skip some columns.
        rename_map (dict): Dictionary containing old and new column names.
        dt_rounding (str): Use to round datetimes in index (e.g. '1 min').
    """

    if path is not None:
        filepath = os.path.join(path, filename)
    else:
        filepath = filename

    data = pd.read_excel(filepath,
                         sheet_name=sheet_name,
                         header=header,
                         usecols=usecols,
                         dtype=dtype,
                         index_col=index_col)

    # Drop initial rows/columns
    data = data.iloc[first_data_row:, first_data_col - 1:]

    # Make index from column with dates
    if index_col is None:
        if date_col is None:
            for col in ['Date', 'Time', 'Datetime', 'Date/Time']:
                if col in data.columns:
                    date_col = col
                    break
        assert date_col is not None, "Specify column that contains datetimes"\
                                     "using index_col (int) or date_col (str)."
        data[index_name] = pd.to_datetime(data[date_col])
        data.set_index(index_name)
    else:
        try:
            data.index = pd.to_datetime(data.index)
        except TypeError:
            raise TypeError("Could not convert index values to datetimes.")
        data.index = data.index.rename(index_name)

    # Drop blank columns from Excel sheet
    columns_to_drop = [col for col in data.columns
                       if str(col).startswith('Unnamed')]

    data = data.drop(columns_to_drop, axis=1)

    # Make all columns strings (prevents names like equipment numbers
    # becoming integers when some are strings)
    if not isinstance(data.columns, pd.core.index.MultiIndex):
        # TODO: Doesn't work with multi-indexes
        data.columns = data.columns.astype(str)

    # Change column names
    if rename_map is not None:
        data = data.rename(columns=rename_map)

    # Round the datetimes in case of precision errors
    if dt_rounding is not None:
        data.index = data.index.round(dt_rounding)

    # Some final checks
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.shape[0] > 0, "No data found"

    if show:
        print(f"Data file shape {data.shape} loaded")

    return data


def read_mv90_csv_file(filename):
    """Reads csv file from SaskPower containing MV90 power
    meter data.  File usually has 'Meter ID' in the first
    column, followed by 'Date / Time' then the various
    meter data which may include KVAR, KVA, KWH, etc.

    If the first row after the headings is empty it is
    removed.

    Returns a dataframe of the values with datetime values
    as the index
    """

    mv90_data = pd.read_csv(filename)

    # Remove first row which is empty
    if mv90_data.loc[0].isna().all():
        mv90_data = mv90_data.drop(0)

    # Remove whitespace from column headings
    rename_map = {col: col.strip() for col in mv90_data.columns}
    mv90_data = mv90_data.rename(index=str, columns=rename_map)

    mv90_data['Date / Time'] = pd.to_datetime(mv90_data['Date / Time'])
    mv90_data = mv90_data.set_index('Date / Time')
    mv90_data['Meter ID'] = mv90_data['Meter ID'].astype(int)

    return mv90_data


def convert_to_numeric(data, show=True):
    """Converts the values in a dataframe or series into
    numeric values, inserting NaN values where data is
    missing or not numeric (see Pandas.to_numeric for
    details).

    If show is True, gives a summary of the number of
    NaN values and counts of the non-numeric data found.
    """

    # This works for Series and Dataframes
    converted_data = data.apply(pd.to_numeric, errors='coerce')

    if show:
        nan_values = converted_data.isna()
        if len(converted_data.shape) > 1:

            # For a data frame, Display table of nan counts
            nan_counts = nan_values.sum().rename("NaN Count")
            pct_complete = (100*nan_counts/nan_values.count()).round(1) \
                            .rename("% Missing")
            nan_count_table = pd.concat([nan_counts, pct_complete], axis=1)
            print(nan_count_table)

        else:
            # For a series, display unique nan values as well
            nan_counts = nan_values.sum()
            print("NaN Value Count:", nan_counts)

            all_nan_values = data.loc[nan_values].value_counts()
            if all_nan_values.sum() > 0:
                print("Non-numeric values found:")
                for value, count in all_nan_values.items():
                    print("%s (%d times)" % (value.__repr__(), count))

    return converted_data


def calculate_running_status(data, min_value=None, min_prop=0.1,
                             label='Running Status', rename_map=None):
    """Estimates when a piece of equipment was operating and when
    it was shut down based on the data.  For example, if data
    is a series of motor amp readings, it will return a series
    of boolean values (True/False).  The threshold used to
    determine running / not-running status is determined by the
    min_prop or min_value parameters (see below).

    Args:
        data (pd.Series or pd.DataFrame): Data inputs
        min_value (float): If not None, this value is used as
            the threshold to determine if running status is
            True.
        min_prop (float): Minimum percentile range used to
            set the running status threshold value if
            min_value is None.
    """

    if min_value is not None:
        running_status = (data >= min_value)
    else:
        min_values = data.quantile(q=0.95)*min_prop
        running_status = (data >= min_values)

    if rename_map is not None:
        if isinstance(running_status, pd.Series):
            running_status.rename(rename_map[running_status.name])
        elif isinstance(running_status, pd.DataFrame):
            running_status = running_status.rename(columns=rename_map)

    return running_status


def drop_values_when_not_running(data, running_status,
                                 rename_map=None):
    """Replaces all values in data where running_status is False
    with np.nan values.  This is useful when you have multiple
    time-series in a dataframe and want to calculate things like
    average amps when running.  This only works when the column
    names in running_status match those in data.  If they are
    different then use rename_map to indicate the columns in
    data to which each column in running_status should be applied.

    Example use:
    >>> data = pd.DataFrame({
    ...     'Pump 1 Amps': [ 0., 86.4, 0., 62.3, 91.5],
    ...     'Pump 2 Amps': [60.6, 63.,  0.,  0., 95.4],
    ... })
    >>> running_status = calculate_running_status(data, min_value=10)
    >>> amps_when_running = drop_values_when_not_running(data, running_status)
    >>> amps_when_running
       Pump 1 Amps  Pump 2 Amps
    0          NaN         60.6
    1         86.4         63.0
    2          NaN          NaN
    3         62.3          NaN
    4         91.5         95.4
    >>> amps_when_running.mean()
    Pump 1 Amps    80.066667
    Pump 2 Amps    73.000000
    dtype: float64
    >>>
    """

    if rename_map is not None:
        data = data.rename(columns=rename_map)

    assert all([col in running_status for col in data])

    return data.mask(~running_status, other=np.nan)


def combine_running_status(data, index=None, sep=', '):
    """Combines a series of boolean values (True/False) into a
    string containing the index values corresponding to True
    values in data.  If all values in data are False, returns
    'None'.

    Use this to create unique labels for different combinations
    of running equipment.

    E.g.
    >>> pump_nos = [1, 2, 3, 4]
    >>> running_status = [True, True, False, True]
    >>> combine_running_status(running_status, index=pump_nos,
    ...                        sep='')
    '124'
    """

    if any(data) is False:
        return 'None'
    if index is None:
        index = range(1, len(data) + 1)
    return sep.join([str(i) for i, x in zip(index, data)
                     if x is True])


def split_into_groups_of_values(x):
    """Takes a series of discrete values x and returns a dataframe
    containing integers representing the count of each group of
    consecutive values that are the same.  This is useful for
    counting the number and duration of certain events.

    Args:
        x (pd.Series): Series of discrete values, usually labels or
            categories.

    Example:
    >>> running_status = pd.Series(
    ...     ['Fully loaded', 'Fully loaded', 'Idling', 'Fully loaded',
    ...      'Fully loaded', 'Idling', 'Idling', 'Idling']
    ... )
    >>> running_status_counts = split_into_groups_of_values(running_status)
    >>> running_status_counts
       Fully loaded  Idling
    0             1       0
    1             1       0
    2             0       1
    3             2       0
    4             2       0
    5             0       2
    6             0       2
    7             0       2
    >>> running_status_counts['Idling'].value_counts().drop(0)
    2    3
    1    1
    """

    values = x.unique()
    group_counts = []
    for value in values:
        group_counts.append(groups_of_true_values(x == value).rename(value))

    return pd.concat(group_counts, axis=1)


def groups_of_true_values(x):
    """Returns an array of integers where each True value in x
    has been replaced by the count of the group of consecutive
    True values that it was found in.  False values are replaced
    with zeros.  (This is used by split_into_groups_of_values.)

    Example:
    >>> x = [True, False, True, True, False]
    >>> groups_of_true_values(x)
    array([1, 0, 2, 2, 0])
    """

    return (np.diff(np.concatenate(([0], np.array(x, dtype=int)))) == 1
            ).cumsum()*x


def plot_daily_load(load_data, date, name, filename=None, ylabel='KVA'):
    """Plots the daily load profile on selected date.
    """

    load_profile = load_data.loc[date:(date + pd.Timedelta('1D'))]
    load_profile.plot(style='.-')
    plt.title("Load Profile on %s - %s" % (date, name))
    plt.ylim(0, )
    plt.ylabel(ylabel)
    plt.grid()

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_daily_loads(load_data, dates, title, filename=None, ylabel='KVA',
                     figsize=(8, 4)):
    """Plots the daily load profiles on selected dates.
    """

    plt.figure(figsize=figsize)
    for date in dates:
        load_profile = load_data.loc[date:(date + pd.Timedelta('1D'))]
        load_profile.index = pd.Index(load_profile.index.time,
                                      name='Time of day')
        load_profile.plot(style='.-', label=date)

    plt.title(title)
    plt.ylim(0, )
    plt.ylabel(ylabel)
    plt.grid()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def filename_from_string(s, allow_chars=('-', '.', '_')):
    """Returns a safe filename from string s by removing
    non-alphanumeric characters and replacing spaces with
    hyphens.

    Example:
    >>> label = 'PUMP #1 MOTOR LOAD'  # Contains invalid char
    >>> filename = f'{label}-plot.pdf'
    >>> utils.filename_from_string(filename)
    'PUMP-1-MOTOR-LOAD-plot.pdf'
    """

    safe_chars = []
    for x in s.strip():
        if x == ' ':
            safe_chars.append('-')
        elif x.isalnum():
            safe_chars.append(x)
        elif x in allow_chars:
            safe_chars.append(x)

    return "".join(safe_chars)
