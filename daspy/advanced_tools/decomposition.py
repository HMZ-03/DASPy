# Purpose: Waveform decomposition
# Author: Minzhe Hu
# Date: 2024.5.13
# Email: hmz2018@mail.ustc.edu.cn
import numpy as np
from numpy.fft import irfft2, ifftshift
from daspy.basic_tools.preprocessing import padding, cosine_taper
from daspy.basic_tools.freqattributes import next_pow_2, fk_transform
from daspy.advanced_tools.denoising import curvelet_denoising


def fk_fan_mask(f, k, fmin=None, fmax=None, kmin=None, kmax=None, vmin=None,
                vmax=None, edge=0.1, flag=None):
    """
    Make a fan mask in f-k domain for f-k filter.

    :param f: Frequency sequence.
    :param k: Wavenumber sequence.
    :param fmin, fmax, kmin, kmax, vmin, vmax: float or or sequence of 2 floats.
        Sequence of 2 floats represents the start and end of taper.
    :param edge: float. The width of fan mask taper edge.
    :param flag: -1 keep only negative apparent velocities, 0 keep both postive
        and negative apparent velocities, 1 keep only positive apparent
        velocities.
    :return: Fan mask.
    """
    ff = np.tile(f, (len(k), 1))
    kk = np.tile(k, (len(f), 1)).T
    vv = - np.divide(ff, kk, out=np.ones_like(ff) * 1e10, where=kk != 0)
    mask = np.ones(vv.shape)
    for phy_quan in ['f', 'k', 'v']:
        p = eval(phy_quan * 2)
        pmin = eval(phy_quan + 'min')
        if pmin:
            if isinstance(pmin, (tuple, list, np.ndarray)):
                tp_b, tp_e = min(pmin), max(pmin)
            else:
                tp_b, tp_e = pmin * max(1 - edge / 2, 0), pmin * (1 + edge / 2)
            tp_wid = tp_e - tp_b
            mask[(abs(p) <= tp_b)] = 0
            area = (abs(p) > tp_b) & (abs(p) < tp_e)
            mask[area] *= 0.5 - 0.5 * \
                np.cos(((abs(p[area]) - tp_b) / tp_wid) * np.pi)

        pmax = eval(phy_quan + 'max')
        if pmax:
            if isinstance(pmax, (tuple, list, np.ndarray)):
                tp_b, tp_e = max(pmax), min(pmax)
            else:
                tp_b, tp_e = pmax * (1 + edge / 2), pmax * (1 - edge / 2)
            tp_wid = tp_b - tp_e
            mask[(abs(p) >= tp_b)] = 0
            area = (abs(p) > tp_e) & (abs(p) < tp_b)
            mask[area] *= 0.5 - 0.5 * \
                np.cos(((tp_b - abs(p[area])) / tp_wid) * np.pi)

    if flag:
        mask[np.sign(vv) == flag] = 0
    return mask


def fk_filter(data, dx, fs, taper=(0.02, 0.05), pad='default', mode='decompose',
              fmin=None, fmax=None, kmin=None, kmax=None, vmin=None, vmax=None,
              edge=0.1, flag=None, verbose=False):
    """
    Transform the data to the f-k domain using 2-D Fourier transform method, and
    transform back to the x-t domain after filtering.

    :param data: numpy.ndarray. Data to do fk filter.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param taper: float or sequence of floats. Each float means decimal
        percentage of Tukey taper for corresponding dimension (ranging from 0 to
        1). Default is 0.1 which tapers 5% from the beginning and 5% from the
        end.
    :param pad: Pad the data or not. It can be float or sequence of floats. Each
        float means padding percentage before FFT for corresponding dimension.
        If set to 0.1 will pad 5% before the beginning and after the end.
        'default' means pad both dimensions to next power of 2. None or False
        means don't pad data before or during Fast Fourier Transform.
    :param mode: str. 'remove' for denoising, 'retain' for extraction, and
        'decompose' for decomposition.
    :param fmin, fmax, kmin, kmax, vmin, vmax: float or or sequence of 2 floats.
        Sequence of 2 floats represents the start and end of taper.
    :param edge: float. The width of fan mask taper edge.
    :param flag: -1 keep only negative apparent velocities, 0 keep both postive
        and negative apparent velocities, 1 keep only positive apparent
        velocities.
    :param verbose: If True, return filtered data, f-k spectrum, frequency
        sequence, wavenumber sequence and f-k mask.
    :return: Filtered data and some variables in the process if verbose==True.
    """
    data_tp = cosine_taper(data, taper)
    if pad == 'default':
        nch, nt = data.shape
        dn = (next_pow_2(nch) - nch, next_pow_2(nt) - nt)
        nfft = None
    elif pad is None or pad is False:
        dn = 0
        nfft = None
    else:
        dn = np.round(np.array(pad) * data.shape).astype(int)
        nfft = 'default'

    data_pd = padding(data_tp, dn)
    nch, nt = data_pd.shape

    fk, f, k = fk_transform(data_pd, dx, fs, taper=0, nfft=nfft)

    mask = fk_fan_mask(f, k, fmin, fmax, kmin, kmax, vmin, vmax, edge=edge,
                       flag=flag)

    if mode == 'remove':
        mask = 1 - mask

    if mode == 'decompose':
        data_flt1 = irfft2(ifftshift(fk * mask, axes=0)).real[:nch, :nt]
        data_flt1 = padding(data_flt1, dn, reverse=True)
        data_flt2 = irfft2(ifftshift(fk * (1 - mask), axes=0)).real[:nch, :nt]
        data_flt2 = padding(data_flt2, dn, reverse=True)
        if verbose:
            return data_flt1, data_flt2, fk, f, k, mask
        else:
            return data_flt1, data_flt2
    else:
        data_flt = irfft2(ifftshift(fk * mask, axes=0)).real[:nch, :nt]
        data_flt = padding(data_flt, dn, reverse=True)
        if verbose:
            return data_flt, fk, f, k, mask
        else:
            return data_flt


def curvelet_windowing(data, dx, fs, mode='decompose', vmin=0, vmax=np.inf,
                       flag=None, pad=0.3, scale_begin=3, nbscales=None,
                       nbangles=16, finest=1):
    """
    Use curevelet transform to keep cooherent signal with certain velocity
    range. {Atterholt et al., 2022 , Geophys. J. Int.}

    :param data: numpy.ndarray. Data to decomposite.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param mode: str. 'remove' for denoising, 'retain' for extraction, and
        'decompose' for decomposition.
    :param vmin, vmax: float. Velocity range in m/s.
    :param flag: -1 keep only negative apparent velocities, 0 keep both postive
        and negative apparent velocities, 1 keep only positive apparent
        velocities.
    :param pad: float or sequence of floats. Each float means padding percentage
        before FFT for corresponding dimension. If set to 0.1 will pad 5% before
        the beginning and after the end.
    :param scale_begin: int. The beginning scale to do coherent denoising.
    :param nbscales: int. Number of scales including the coarsest wavelet level.
        Default set to ceil(log2(min(M,N)) - 3).
    :param nbangles: int. Number of angles at the 2nd coarsest level,
        minimum 8, must be a multiple of 4.
    :param finest: int. Objects at the finest scale. 1 for curvelets, 2 for
        wavelets. Curvelets are more precise while wavelets are more efficient.
    :return: numpy.ndarray. Decomposed data.
    """
    return curvelet_denoising(data, choice=1, pad=pad, vmin=vmin, vmax=vmax,
                              flag=flag, dx=dx, fs=fs, mode=mode,
                              scale_begin=scale_begin, nbscales=nbscales,
                              nbangles=nbangles, finest=finest)
