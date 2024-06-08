# Purpose: Convert strain rate data to velocity
# Author: Minzhe Hu
# Date: 2024.6.8
# Email: hmz2018@mail.ustc.edu.cn
import numpy as np
from numpy.fft import irfft2, ifftshift
from scipy.signal import hilbert
from daspy.basic_tools.freqattributes import next_pow_2, fk_transform
from daspy.basic_tools.preprocessing import padding, cosine_taper
from daspy.basic_tools.filter import bandpass
from daspy.advanced_tools.fdct import fdct_wrapping, ifdct_wrapping
from daspy.advanced_tools.denoising import _velocity_bin
from daspy.advanced_tools.decomposition import fk_fan_mask


def fk_rescaling(data, dx, fs, taper=(0.02, 0.05), pad='default', fmax=None,
                 kmin=(1 / 2000, 1 / 3000), vmax=(15000, 30000), edge=0.2,
                 turning=None, verbose=False):
    """
    Convert strain/strain rate to velocity/acceleration by fk rescaling.

    :param data: numpy.ndarray. Data to do fk rescaling.
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
    :param fmax, kmin, vmax: float or or sequence of 2 floats. Sequence of 2
        floats represents the start and end of taper. Setting these parameters
        can reduce artifacts.
    :param edge: float. The width of fan mask taper edge.
    :param turning: Sequence of int. Channel number of turning points.
    :param verbose: If True, return converted data, f-k spectrum, frequency
        sequence, wavenumber sequence and f-k mask.
    :return: Converted data and some variables in the process if verbose==True.
    """
    if turning is not None:
        data_vel = np.zeros_like(data)
        start_ch = [0, *turning]
        end_ch = [*turning, len(data)]
        for (s, e) in zip(start_ch, end_ch):
            data_vel[s:e] = fk_rescaling(data[s:e], dx, fs, taper=taper,
                                         pad=pad, fmax=fmax, kmin=kmin,
                                         vmax=vmax, edge=edge, verbose=False)
    else:
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

        fk, f, k = fk_transform(data_pd, dx, fs, taper=taper, nfft=nfft)

        ff = np.tile(f, (len(k), 1))
        kk = np.tile(k, (len(f), 1)).T
        vv = - np.divide(ff, kk, out=np.ones_like(ff) * 1e10, where=kk != 0)

        mask = fk_fan_mask(f, k, fmax=fmax, kmin=kmin, vmax=vmax, edge=edge) * vv
        mask[kk == 0] = 0

        data_vel = irfft2(ifftshift(fk * mask, axes=0)).real[:nch, :nt]
        data_vel = padding(data_vel, dn, reverse=True)

        if verbose:
            return data_vel, fk, f, k, mask
    return data_vel


def curvelet_conversion(data, dx, fs, pad=0.3, scale_begin=2, nbscales=None,
                        nbangles=16, turning=None):
    """
    Use curevelet transform to convert strain/strain rate to
    velocity/acceleration. {Yang et al. , 2023, Geophys. Res. Lett.}

    :param data: numpy.ndarray. Data to convert.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param pad: float or sequence of floats. Each float means padding percentage
        before FFT for corresponding dimension. If set to 0.1 will pad 5% before
        the beginning and after the end.
    :param scale_begin: int. The beginning scale to do conversion.
    :param nbscales: int. Number of scales including the coarsest wavelet level.
        Default set to ceil(log2(min(M,N)) - 3).
    :param nbangles: int. Number of angles at the 2nd coarsest level,
        minimum 8, must be a multiple of 4.
    :param turning: Sequence of int. Channel number of turning points.
    :return: numpy.ndarray. Converted data.
    """
    if turning is not None:
        print(1)
        data_vel = np.zeros_like(data)
        start_ch = [0, *turning]
        end_ch = [*turning, len(data)]
        for (s, e) in zip(start_ch, end_ch):
            data_vel[s:e] = curvelet_conversion(data[s:e], dx, fs, pad=pad,
                                                scale_begin=scale_begin,
                                                nbscales=nbscales,
                                                nbangles=nbangles, turning=None)
    else:
        if pad is None or pad is False:
            pad = 0
        dn = np.round(np.array(pad) * data.shape).astype(int)
        data_pd = padding(data, dn)

        C = fdct_wrapping(data_pd, is_real=True, finest=1, nbscales=nbscales,
                          nbangles_coarse=nbangles)

        # rescale with velocity
        np.seterr(divide='ignore')
        for s in range(0, scale_begin - 1):
            for w in range(len(C[s])):
                C[s][w] *= 0

        for s in range(scale_begin - 1, len(C)):
            nbangles = len(C[s])
            velocity = _velocity_bin(nbangles, fs, dx)
            factors = np.mean(velocity, axis=1)
            for w in range(nbangles):
                if abs(factors[w]) == np.inf:
                    factors[w] = abs(velocity[w]).min() * \
                        np.sign(velocity[w, 0]) * 2
                C[s][w] *= factors[w]

        data_vel = ifdct_wrapping(C, is_real=True, size=data_pd.shape)
        data_vel = padding(data_vel, dn, reverse=True)

    return data_vel


def slowness(g, dx, fs, slm, sls, swin=2):
    """
    Estimate the slowness time series by calculate semblance.
    {Lior et al., 2021, Solid Earth}

    :param g: 2-dimensional array. time series of adjacent channels used for
        estimating slowness
    :param dx: float. Spatical sampling rate (in m)
    :param fs: float. Sampling rate of records
    :param slm: float. Slowness x max
    :param sls: float. Slowness step
    :param swin: int. Slowness smooth window
    :return: Sequences of slowness and sembalence.
    """
    L = (len(g) - 1) // 2
    nt = len(g[0])
    h = np.imag(hilbert(g))
    grdpnt = round(slm / sls)
    sem = np.zeros((2 * grdpnt + 1, nt))
    gap = round(slm * dx * L * fs)

    h_ex = np.zeros((len(g), nt + 2 * gap))
    h_ex[:, gap:-gap] = h
    g_ex = np.zeros((len(g), nt + 2 * gap))
    g_ex[:, gap:-gap] = g

    for i in range(2 * grdpnt + 1):
        px = (i - grdpnt) * sls
        if abs(px) < 1e-5:
            continue
        gt = np.zeros(g.shape)
        ht = np.zeros(h.shape)
        for j in range(-L, L):
            shift = round(px * j * dx * fs)
            gt[j + L] = g_ex[j + L, gap + shift:gap + shift + nt]
            ht[j + L] = h_ex[j + L, gap + shift:gap + shift + nt]
        sem[i] = (np.sum(gt, axis=0)**2 + np.sum(ht, axis=0)**2) / \
            np.sum(gt**2 + ht**2, axis=0) / (2 * L + 1)
    p = (np.argmax(sem, axis=0) - grdpnt) * sls
    # smooth P
    for i in range(swin, nt - swin):
        win = p[i - swin:i + swin + 1]
        sign = np.sign(sum(np.sign(win)))
        win = [px for px in win if np.sign(px) == sign]
        p[i] = np.mean(win)

    return p, sem


def slant_stacking(data, dx, fs, L=None, slm=0.01,
                   sls=0.000125, frqlow=0.1, frqhigh=15, turning=None,
                   channel='all'):
    """
    Convert strain to velocity based on slant-stack.

    :param data: 2-dimensional array. Axis 0 is channel number and axis 1 is
        time series
    :param dx: float. Spatical sampling rate (in m)
    :param L: int. the number of adjacent channels over which slowness is
        estimated
    :param slm: float. Slowness x max
    :param sls: float. slowness step
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param turning: Sequence of int. Channel number of turning points.
    :param channel: int or list or 'all'. convert a certain channel number /
        certain channel range / all channels.
    :return: Converted velocity data
    """
    if L is None:
        L = round(50 / dx)

    nch, nt = data.shape
    if isinstance(channel, str) and channel == 'all':
        channel = list(range(nch))
    elif isinstance(channel, int):
        channel = [channel]

    if turning is not None:
        data_vel = np.zeros((0, len(data[0])))
        start_ch = [0, *turning]
        end_ch = [*turning, len(data)]
        for (s, e) in zip(start_ch, end_ch):
            channel_seg = [ch-s for ch in range(s,e) if ch in channel]
            if len(channel_seg):
                d_vel = slant_stacking(data[s:e], dx, fs, L=L, slm=slm, sls=sls,
                                       frqlow=frqlow, frqhigh=frqhigh,
                                       turning=None, channel=channel_seg)
                data_vel = np.vstack((data_vel, d_vel))
    else:
        data_ex = padding(data, (2 * L, 0))
        swin = int(max((1 / frqhigh * fs) // 2, 1))
        data_vel = np.zeros((len(channel), nt))
        for i, ch in enumerate(channel):
            p, _ = slowness(data_ex[ch:ch + 2 * L + 1], dx, fs, slm, sls,
                            swin=swin)
            data_vel[i] = bandpass(data[ch] / p, fs=fs, freqmin=frqlow,
                                   freqmax=frqhigh)

    return data_vel
