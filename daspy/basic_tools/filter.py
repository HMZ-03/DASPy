# Purpose: Filter the waveform
# Author: Minzhe Hu
# Date: 2024.6.17
# Email: hmz2018@mail.ustc.edu.cn
# Modified from https://docs.obspy.org/_modules/obspy/signal/filter.html
import warnings
import numpy as np
from scipy.signal import cheb2ord, cheby2, hilbert, iirfilter, zpk2sos, sosfilt


def _preprocessing(data, detrend, taper):
    from daspy.basic_tools.preprocessing import demeaning, detrending, \
        cosine_taper
    if detrend in [True, 'linear', 'detrend']:
        data = detrending(data)
    elif detrend in ['constant', 'demean']:
        data = demeaning(data)
    
    if taper:
        taper = (taper, 0.1)[taper is True]
        data = cosine_taper(data, p=taper)

    return data


def bandpass(data, fs, freqmin, freqmax, corners=4, zerophase=True,
             detrend=True, taper=False):
    """
    Filter data from 'freqmin' to 'freqmax' using Butterworth bandpass filter of
    'corners' corners.

    :param data: numpy.ndarray. Data to filter.
    :param fs: Sampling rate in Hz.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :param detrend : str or bool. Specifies whether and how to detrend each
        segment.  'linear' or 'detrend' or True = detrend, 'constant' or
        'demean' = demean.
    :param taper: bool or float. Float means decimal percentage of Tukey taper
        for time dimension (ranging from 0 to 1). True for 0.1 which tapers 5%
        from the beginning and 5% from the end.
    :return: Filtered data.
    """
    data = _preprocessing(data, detrend, taper)
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = ('Selected high corner frequency ({}) of bandpass is at or ' +
               'above Nyquist ({}). Applying a high-pass instead.').format(
            freqmax, fe)
        warnings.warn(msg)
        return highpass(data, freq=freqmin, fs=fs, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = 'Selected low corner frequency is above Nyquist.'
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    data_flt = sosfilt(sos, data)
    if zerophase:
        data_flt = sosfilt(sos, data_flt[:, ::-1])[:, ::-1]

    if len(data_flt) == 1:
        data_flt = data_flt[0]
    return data_flt


def bandstop(data, fs, freqmin, freqmax, corners=4, zerophase=False,
             detrend=True, taper=False):
    """
    Filter data removing data between frequencies 'freqmin' and 'freqmax' using
    Butterworth bandstop filter of 'corners' corners.

    :param data: numpy.ndarray. Data to filter.
    :param fs: Sampling rate in Hz.
    :param freqmin: Stop band low corner frequency.
    :param freqmax: Stop band high corner frequency.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :param detrend : str or bool. Specifies whether and how to detrend each
        segment.  'linear' or 'detrend' or True = detrend, 'constant' or
        'demean' = demean.
    :param taper: bool or float. Float means decimal percentage of Tukey taper
        for time dimension (ranging from 0 to 1). True for 0.1 which tapers 5%
        from the beginning and 5% from the end.
    :return: Filtered data.
    """
    data = _preprocessing(data, detrend, taper)
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    fe = 0.5 * fs
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high > 1:
        high = 1.0
        msg = 'Selected high corner frequency is above Nyquist. Setting ' + \
              'Nyquist as high corner.'
        warnings.warn(msg)
    if low > 1:
        msg = 'Selected low corner frequency is above Nyquist.'
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high],
                        btype='bandstop', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, k)
    data_flt = sosfilt(sos, data)
    if zerophase:
        data_flt = sosfilt(sos, data_flt[:, ::-1])[:, ::-1]

    if len(data_flt) == 1:
        data_flt = data_flt[0]
    return data_flt


def lowpass(data, fs, freq, corners=4, zerophase=False, detrend=True,
            taper=False):
    """
    Filter data removing data over certain frequency 'freq' using Butterworth
    lowpass filter of 'corners' corners.

    :param data: numpy.ndarray. Data to filter.
    :param fs: Sampling rate in Hz.
    :param freq: Filter corner frequency.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :param detrend : str or bool. Specifies whether and how to detrend each
        segment.  'linear' or 'detrend' or True = detrend, 'constant' or
        'demean' = demean.
    :param taper: bool or float. Float means decimal percentage of Tukey taper
        for time dimension (ranging from 0 to 1). True for 0.1 which tapers 5%
        from the beginning and 5% from the end.
    :return: Filtered data.
    """
    data = _preprocessing(data, detrend, taper)
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    fe = 0.5 * fs
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        f = 1.0
        msg = 'Selected corner frequency is above Nyquist. Setting Nyquist ' + \
              'as high corner.'
        warnings.warn(msg)
    z, p, k = iirfilter(corners, f, btype='lowpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    data_flt = sosfilt(sos, data)
    if zerophase:
        data_flt = sosfilt(sos, data_flt[:, ::-1])[:, ::-1]

    if len(data_flt) == 1:
        data_flt = data_flt[0]
    return data_flt


def lowpass_cheby_2(data, fs, freq, maxorder=12, ba=False, freq_passband=False):
    """
    Filter data by passing data only below a certain frequency. The main purpose
    of this cheby2 filter is downsampling. This method will iteratively design a
    filter, whose pass band frequency is determined dynamically, such that the
    values above the stop band frequency are lower than -96dB.

    :param data: numpy.ndarray. Data to filter.
    :param fs: Sampling rate in Hz.
    :param freq: The frequency above which signals are attenuated with 95 dB.
    :param maxorder: Maximal order of the designed cheby2 filter.
    :param ba: If True return only the filter coefficients (b, a) instead of
        filtering.
    :param freq_passband: If True return additionally to the filtered data, the
        iteratively determined pass band frequency.
    :return: Filtered data.
    """
    if data.ndim == 1:
        data = data[np.newaxis, :]

    nyquist = fs * 0.5
    # rp - maximum ripple of passband, rs - attenuation of stopband
    rp, rs, order = 1, 96, 1e99
    ws = freq / nyquist  # stop band frequency
    wp = ws  # pass band frequency
    # raise for some bad scenarios
    if ws > 1:
        ws = 1.0
        msg = "Selected corner frequency is above Nyquist. " + \
              "Setting Nyquist as high corner."
        warnings.warn(msg)
    while True:
        if order <= maxorder:
            break
        wp = wp * 0.99
        order, wn = cheb2ord(wp, ws, rp, rs, analog=0)
    if ba:
        return cheby2(order, rs, wn, btype='low', analog=0, output='ba')
    z, p, k = cheby2(order, rs, wn, btype='low', analog=0, output='zpk')
    sos = zpk2sos(z, p, k)
    data_flt = sosfilt(sos, data)
    if len(data_flt) == 1:
        data_flt = data_flt[0]
    if freq_passband:
        return data_flt, wp * nyquist
    return data_flt


def highpass(data, fs, freq, corners=4, zerophase=False, detrend=True,
             taper=False):
    """
    Filter data removing data below certain frequency 'freq' using Butterworth
    highpass filter of 'corners' corners.

    :param data: numpy.ndarray. Data to filter.
    :param fs: Sampling rate in Hz.
    :param freq: Filter corner frequency.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :param detrend : str or bool. Specifies whether and how to detrend each
        segment.  'linear' or 'detrend' or True = detrend, 'constant' or
        'demean' = demean.
    :param taper: bool or float. Float means decimal percentage of Tukey taper
        for time dimension (ranging from 0 to 1). True for 0.1 which tapers 5%
        from the beginning and 5% from the end.
    :return: Filtered data.
    """
    data = _preprocessing(data, detrend, taper)
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    fe = 0.5 * fs
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = 'Selected corner frequency is above Nyquist.'
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter',
                        output='zpk')
    sos = zpk2sos(z, p, k)
    data_flt = sosfilt(sos, data)
    if zerophase:
        data_flt = sosfilt(sos, data_flt[:, ::-1])[:, ::-1]

    if len(data_flt) == 1:
        data_flt = data_flt[0]
    return data_flt


def envelope(data):
    """
    Computes the envelope of the given data. The envelope is determined by
    adding the squared amplitudes of the data and it's Hilbert-Transform and
    then taking the square-root. The envelope at the start/end should not be
    taken too seriously.

    :param data: numpy.ndarray. Data to make envelope of.
    :return: Envelope of input data.
    """
    return abs(hilbert(data, axis=-1))
