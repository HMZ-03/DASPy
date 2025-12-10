# Purpose: Some preprocess methods
# Author: Minzhe Hu
# Date: 2025.10.30
# Email: hmz2018@mail.ustc.edu.cn
import warnings
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import detrend
from scipy.signal.windows import tukey
from daspy.basic_tools.filter import lowpass_cheby_2


def phase2strain(data, lam, e, n, gl):
    """
    Convert the optical phase shift in radians to strain.

    :param data: numpy.ndarray. Data to convert.
    :param lam: float. Operational optical wavelength in vacuum.
    :param e: float. photo-slastic scaling factor for logitudinal strain in
        isotropic material.
    :param n: float. Refractive index of the sensing fiber.
    :paran guage_length: float. Gauge length.
    :return: Strain data.
    """
    return data * lam / (e * 4 * np.pi * n * gl)


def normalization(data, method='z-score'):
    """
    Normalize for each individual channel using Z-score method.

    :param data: numpy.ndarray. Data to normalize.
    :param method: str. Method for normalization, should be one of 'max',
        'z-score', 'MAD' or 'one-bit'.
    :return: Normalized data.
    """
    if data.ndim == 1:
        data = data.reshape(1, len(data))
    elif data.ndim != 2:
        raise ValueError("Data should be 1-D or 2-D array")

    if method.lower() == 'max':
        amp = np.max(abs(data), 1, keepdims=True)
        amp[amp == 0] = amp[amp > 0].min()
        return data / amp
    elif method.lower() == 'z-score':
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        std[std == 0] = std[std > 0].min()
        return (data - mean) / std
    elif method.lower() == 'mad':
        median = np.median(data, axis=1, keepdims=True)
        mad = np.median(abs(data - median), axis=1, keepdims=True)
        mad[mad == 0] = mad[mad > 0].min()
        return (data - median) / mad
    elif method.lower() == 'one-bit':
        return np.sign(data)


def demeaning(data):
    """
    Demean signal by subtracted mean of each channel.

    :param data: numpy.ndarray. Data to demean.
    :return: Detrended data.
    """
    return detrend(data, type='constant')


def detrending(data):
    """
    Detrend signal by subtracted a linear least-squares fit to data.

    :param data: numpy.ndarray. Data to detrend.
    :return: Detrended data.
    """
    return detrend(data, type='linear')


def stacking(data: np.ndarray, N: int, step: int = None, average: bool = True):
    """
    Stack several channels to increase the signal-noise ratio(SNR).

    :param data: numpy.ndarray. Data to stack.
    :param N: int. N adjacent channels stacked into 1.
    :param step: int. Interval of data stacking.
    :param average: bool. True for calculating the average.
    :return: Stacked data.
    """
    if N == 1:
        return data
    if step is None:
        step = N
    nch, nsp = data.shape
    begin = np.arange(0, nch - N + 1, step)
    end = begin + N
    nx1 = len(begin)
    data_stacked = np.zeros((nx1, nsp))
    for i in range(nx1):
        data_stacked[i, :] = np.sum(data[begin[i]:end[i], :], axis=0)
    if average:
        data_stacked /= N
    return data_stacked


def cosine_taper(data, p=0.1, side='both'):
    """
    Taper using Tukey window.

    :param data: numpy.ndarray. Data to taper.
    :param p: float or sequence of floats. Each float means decimal percentage
        of Tukey taper for corresponding dimension (ranging from 0 to 1).
        Default is 0.1 which tapers 5% from the beginning and 5% from the end.
        If only one float is given, it only do for time dimension.
    :param side: str. 'both', 'left', or 'right'.
    :return: Tapered data.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    nch, nsp = data.shape
    if not isinstance(p, (tuple, list, np.ndarray)):
        win = tukey(nsp, p)
        if side == 'left':
            win[round(nsp/2):] = 1
        elif side == 'right':
            win[:round(nsp/2)] = 1
        return data * np.tile(win, (nch, 1))
    else:
        if p[0] > 0:
            data = data * np.tile(tukey(nch, p[0]), (nsp, 1)).T
        return cosine_taper(data, p[1], side=side)


def downsampling(data, xint=None, tint=None, stack=True, lowpass_filter=True):
    """
    Downsample DAS data.

    :param data: numpy.ndarray. Data to downsample can be 1-D or 2-D.
    :param xint: int. Spatial downsampling factor.
    :param tint: int. Time downsampling factor.
    :param lowpass_filter: bool. Lowpass cheby2 filter before time downsampling
        or not.
    :return: Downsampled data.
    """
    data_ds = data.copy()
    if xint and xint > 1:
        if stack:
            data_ds = stacking(data, xint)
        else:
            data_ds = data_ds[::xint].copy()
    if tint and tint > 1:
        if lowpass_filter:
            data_ds = lowpass_cheby_2(data_ds, 1, 1 / 2 / tint)
        if len(data_ds.shape) == 1:
            data_ds = data_ds[::tint].copy()
        else:
            data_ds = data_ds[:, ::tint].copy()
    return data_ds


def _trimming_index(nch, nsp, dx=None, fs=None, start_channel=0,
                    start_distance=0, start_time=0, xmin=None, xmax=None,
                    chmin=None, chmax=None, tmin=None, tmax=None, spmin=None,
                    spmax=None):
    assert None in [tmin, spmin], \
        "Please do not set tmin and spmin at the same time."
    assert None in [tmax, spmax], \
        "Please do not set tmax and spmax at the same time."
    assert None in [xmin, chmin], \
        "Please do not set xmin and chmin at the same time."
    assert None in [xmax, chmax], \
        "Please do not set xmax and chmax at the same time."
    if dx is None:
        assert xmin is None and xmax is None, "Please set dx"
    if fs is None:
        assert tmin is None and tmax is None, "Please set fs"

    if xmin is None:
        if chmin is None:
            i0 = 0
        else:
            i0 = int(chmin - start_channel)
            if i0 < 0:
                warnings.warn('chmin < start_channel . Set chmin to '
                                'start_channel.')
                i0 = 0
            elif i0 >= nch:
                raise ValueError('chmin >= end_channel.')
    else:
        i0 = round((xmin - start_distance) / dx)
        if i0 < 0:
            warnings.warn('xmin is smaller than start_distance. Set xmin '
                            'to 0.')
            i0 = 0
        elif i0 >= nch:
            raise ValueError('xmin is later than end_distance.')

    if xmax is None:
        if chmax is None:
            i1 = nch
        else:
            i1 = int(chmax - start_channel)
            if i1 <= 0:
                raise ValueError('chmax <= start_channel.')
            elif i1 > nch:
                warnings.warn('chmax > end_channel. Set chmax to '
                                'end_channel.')
                i1 = nch
    else:
        i1 = round((xmax - start_distance) / dx)
        if i1 <= 0:
            raise ValueError('xmax is smaller than start_distance.')
        if i1 > nch:
            warnings.warn('xmax is later than end_distance. Set xmax '
                            'to the array length.')
            i1 = nch

    if tmin is None:
        if spmin is None:
            j0 = 0
        else:
            j0 = int(spmin)
            if j0 < 0:
                warnings.warn('spmin < 0. Set spmin to 0.')
                j0 = 0
            elif j0 >= nsp:
                raise ValueError('spmin > nsp.')
    else:
        try:
            j0 = round((tmin - start_time) * fs)
        except TypeError:
            j0 = round(tmin * fs)
        if j0 < 0:
            warnings.warn('tmin is earlier than start_time. Set tmin '
                            'to start_time.')
            j0 = 0
        elif j0 >= nsp:
            raise ValueError('tmin is later than end_time.')

    if tmax is None:
        if spmax is None:
            j1 = nsp
        else:
            j1 = int(spmax)
            if j1 <= 0:
                raise ValueError('spmax < 0.')
            elif j1 > nsp:
                warnings.warn('spmax > nsp. Set spmax to nsp.')
                j1 = nsp
    else:
        try:
            j1 = round((tmax - start_time) * fs)
        except TypeError:
            j1 = round(tmax * fs)
        if j1 <= 0:
            raise ValueError('tmax is earlier than start_time.')
        if j1 > nsp:
            warnings.warn('tmax is later than end_time. Set tmax to the'
                            ' end_time.')
            j1 = nsp
    return i0, i1, j0, j1


def trimming(data, dx=None, fs=None, xmin=None, xmax=None, chmin=None,
             chmax=None, tmin=None, tmax=None, spmin=None, spmax=None,
             **kwargs):
    """
    Cut data to given start and end distance/channel or time/sampling points.

    :param data: numpy.ndarray. Data to trim can be 1-D or 2-D.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param xmin, xmax: float. Range of distance.
    :param chmin, chmax: int. Channel number range.
    :param tmin, tmax: float or DASDateTime. Range of time.
    :param spmin, spmax: int. Sampling point range.
    :return: Trimmed data.
    """
    # Compatible with old interfaces and remind users
    if 'mode' in kwargs:
        warnings.warn('In future versions, the mode parameter will be '
                        'deprecated. xmin/xmax will only control the distance'
                        ' range, tmin/tmax will only control the time range; '
                        'please use chmin/chmax to control the channel number'
                        ' range, and spmin/spmax to control the sampling '
                        'point range', FutureWarning)
        if kwargs['mode'] == 0:
            chmin, chmax = xmin, xmax
            xmin, xmax = None, None
            spmin, spmax = tmin, tmax
            tmin, tmax = None, None
    nch, nsp = data.shape
    i0, i1, j0, j1 = _trimming_index(nch, nsp, dx=dx, fs=fs, xmin=xmin,
                                     xmax=xmax, chmin=chmin, chmax=chmax,
                                     tmin=tmin, tmax=tmax, spmin=spmin,
                                     spmax=spmax)

    return data[i0:i1, j0:j1].copy()


def padding(data, dn, reverse=False):
    """
    Pad DAS data with 0.

    :param data: numpy.ndarray. 2D DAS data to pad.
    :param dn: int or sequence of ints. Number of points to pad for both
        dimensions.
    :param reverse: bool. Set True to reverse the operation.
    :return: Padded data.
    """
    nch, nsp = data.shape
    if isinstance(dn, int):
        dn = (dn, dn)

    pad = (dn[0] // 2, dn[0] - dn[0] // 2, dn[1] // 2, dn[1] - dn[1] // 2)
    if reverse:
        return data[pad[0]:nch - pad[1], pad[2]:nsp - pad[3]]
    else:
        data_pd = np.zeros((nch + dn[0], nsp + dn[1]))
        data_pd[pad[0]:nch + pad[0], pad[2]:nsp + pad[2]] = data
        return data_pd


def time_integration(data, fs, domain='time', c=0):
    """
    Integrate DAS data in time.

    :param data: numpy.ndarray. 2D DAS data.
    :param fs: Sampling rate in Hz.
    :param c: float. A constant added to the result.
    :return: Integrated data.
    """
    if domain == 'time':
        return np.cumsum(data, axis=1) / fs + c
    elif domain in ['frequency', 'freq']:
        nsp = data.shape[1]
        freqs = rfftfreq(nsp, d=1/fs)
        spectrum = rfft(data, axis=1)
        H = np.zeros_like(freqs, dtype=complex)
        nonzero = freqs != 0
        H[nonzero] = 1 / (1j * 2 * np.pi * freqs[nonzero])
        return np.real(irfft(spectrum * H))


def time_differential(data, fs, domain='time', prepend=0):
    """
    Differentiate DAS data in time.

    :param data: numpy.ndarray. 2D DAS data.
    :param fs: Sampling rate in Hz.
    :param prepend: 'mean' or values to prepend to `data` along axis prior to
        performing the difference. 
    :return: Differentiated data.
    """
    if domain == 'time':
        if prepend == 'mean':
            prepend = np.mean(data, axis=1).reshape((-1, 1))
        return np.diff(data, axis=1, prepend=prepend) * fs
    elif domain in ['frequency', 'freq']:
        nsp = data.shape[1]
        freqs = rfftfreq(nsp, d=1./fs)
        spectrum = rfft(data, axis=1)
        H = 1j * 2 * np.pi * freqs
        return np.real(irfft(spectrum * H))


def distance_integration(data, dx, c=0):
    """
    Integrate DAS data in distance.

    :param data: numpy.ndarray. 2D DAS data.
    :param dx: Channel interval in m.
    :param c: float. A constant added to the result.
    :return: Integrated data.
    """
    return np.cumsum(data, axis=1) * dx + c