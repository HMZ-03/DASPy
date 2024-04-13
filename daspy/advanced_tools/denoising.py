# Purpose: Remove noise from data
# Author: Minzhe Hu, Zefeng Li
# Date: 2024.4.13
# Email: hmz2018@mail.ustc.edu.cn
import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d
from daspy.basic_tools.preprocessing import padding
from daspy.advanced_tools.fdct import fdct_wrapping, ifdct_wrapping


def spike_removal(data, nch=50, nsp=5, thresh=10):
    """
    Use a median filter to remove high-strain spikes in the data. Modified from
    https://github.com/atterholt/curvelet-denoising/blob/main/MedianFilter.m

    :param data: numpy.ndarray. Data to remove spikes from.
    :param nch: int. Number of channels over which to compute the median.
    :param nsp: int. Number of sampling points over which to compute the median.
    :param thresh: Ratio threshold over the median over which a number is
        considered to be an outlier.
    :return: numpy.ndarray. Data with spikes removed.
    """
    absdata = np.abs(data)

    medians1 = median_filter(absdata, (nch, 1))
    medians = median_filter(medians1, (1, nsp))
    ratio = absdata / medians  # comparisons matrix

    # find the bad values and interpolate with their neighbors
    data_dn = data.copy()
    out_i, out_j = np.where(ratio > thresh)
    for j in set(out_j):
        bch = out_i[out_j == j]
        gch = list(set(range(len(data))) - set(bch))
        f = interp1d(gch, data[gch, j], bounds_error=False,
                     fill_value=(data[gch[0], j], data[gch[-1], j]))
        data_dn[bch, j] = f(bch)

    return data_dn


def common_mode_noise_removal(data):
    """
    Remove common mode noise (sometimes called horizontal noise) from data.

    :param data: numpy.ndarray. Data to remove common mode noise.
    :return: numpy.ndarray. Denoised data.
    """
    nch, nt = data.shape
    common = np.median(data, 0)
    xx = np.sum(common ** 2)
    data_dn = np.zeros((nch, nt))
    for i in range(nch):
        xc = np.sum(common * data[i])
        data_dn[i] = data[i] - xc / xx * common

    return data_dn


def _noise_level(data, percentile=95, nbscales=None, nbangles=16):
    """
    Find threshold for curvelet denoising with noise record.

    :param data: numpy.ndarray. Noise data.
    :param nbscales: int. Number of scales including the coarsest wavelet level.
        Default set to ceil(log2(min(M,N)) - 3).
    :param nbangles: int. Number of angles at the 2nd coarsest level,
        minimum 8, must be a multiple of 4.
    :return: 2-D list. Threshold for curvelet coefficients.
    """
    C = fdct_wrapping(data, is_real=False, finest=2, nbscales=nbscales,
                      nbangles_coarse=nbangles)

    E_noise = []
    for s in range(len(C)):
        E_noise.append([])
        for w in range(len(C[s])):
            threshold = np.percentile(abs(C[s][w]), percentile)
            E_noise[s].append(threshold)

    return E_noise


def _knee_points(data, factor=0.25, nbscales=None, nbangles=16):
    """
    Find threshold for curvelet denoising without noise record.

    :param data: numpy.ndarray. Data to denoise.
    :param facetor: float. Multiplication factor from 0 to 1. Small factor
        corresponds to conservative strategy.
    :param nbscales: int. Number of scales including the coarsest wavelet level.
        Default set to ceil(log2(min(M,N)) - 3).
    :param nbangles: int. Number of angles at the 2nd coarsest level,
        minimum 8, must be a multiple of 4.
    :return: 2-D list. Threshold for curvelet coefficients.
    """
    C = fdct_wrapping(data, is_real=False, finest=2, nbscales=nbscales,
                      nbangles_coarse=nbangles)
    E_knee = []
    for s in range(len(C)):
        E_knee.append([])
        for w in range(len(C[s])):
            F, x = np.histogram(abs(C[s][w]), density=True)
            x = (x[1:] + x[:-1]) / 2
            F = np.cumsum(F) / np.sum(F)
            slope = (x[-1] - x[0]) / (F[-1] - F[0])
            tiltedplot = x - (slope * F)
            idx = np.argmin(tiltedplot)
            E_knee[s].append(x[idx] * factor)

    return E_knee


def _velocity_bin(nbangles, fs, dx):
    v_bounds = np.zeros(nbangles // 4 + 1)
    half = nbangles // 8
    v_bounds[half] = fs * dx
    np.seterr(divide='ignore')
    for i in range(half):
        v_bounds[i] = i / half * fs * dx
        v_bounds[half + i + 1] = np.divide(fs * dx, 1 - (i + 1) / half)

    np.seterr(divide='warn')
    v_lows = list(range(half - 1, -1, -1)) + list(range(half * 2)) + \
        list(range(2 * half - 1, half - 1, -1))
    velocity = []
    for i in range(nbangles // 2):
        v_low = v_bounds[v_lows[i]]
        v_high = v_bounds[v_lows[i] + 1]
        velocity.append([v_low, v_high])
    velocity = np.array(velocity * 2)
    for i in range(half):
        velocity[i] = -1 * velocity[i][::-1]
        velocity[3 * half + i] = -1 * velocity[3 * half + i][::-1]
        velocity[4 * half + i] = -1 * velocity[4 * half + i][::-1]
        velocity[7 * half + i] = -1 * velocity[7 * half + i][::-1]
    return velocity


def _mask_factor(velocity, vmin, vmax, flag=None, mode='remove'):
    if flag:
        if flag == -1:
            vmin = -vmax
            vmax = -vmin
    else:
        velocity = abs(velocity)

    factors = np.zeros(len(velocity))
    for i, (v_low, v_high) in enumerate(velocity):
        v1 = max(v_low, vmin)
        v2 = min(v_high, vmax)
        if v1 < v2:
            if v_high == np.inf or v_low == -np.inf:
                factors[i] = 1
            else:
                factors[i] = np.divide(v2 - v1, v_high - v_low)
    if mode == 'retain':
        return factors
    elif mode == 'remove':
        return 1 - factors


def curvelet_denoising(data, choice=0, pad=0.3, noise=None, soft_thresh=True,
                       v_range=None, flag=None, dx=None, fs=None, mode='remove',
                       scale_begin=3, nbscales=None, nbangles=16):
    """
    Use curevelet transform to filter stochastic or/and cooherent noise.
    Modified from
    https://github.com/atterholt/curvelet-denoising/blob/main/CurveletDenoising.m
    {Atterholt et al., 2022 , Geophys. J. Int.}

    :param data: numpy.ndarray. Data to denoise.
    :param choice: int. 0 for Gaussian denoising using soft thresholding, 1 for
        velocity filtering using the standard FK methodology and 2 for both.
    :param pad: float or sequence of floats. Each float means padding percentage
        before FFT for corresponding dimension. If set to 0.1 will pad 5% before
        the beginning and after the end.
    :param noise: numpy.ndarray. Noise record as reference.
    :param soft_thresh: bool. True for soft thresholding and False for hard
        thresholding.
    :param vrange: tuple or list. (vmin vmax) for filter out cooherent noise of
        velocity between vmin and vmax m/s.
    :param flag: -1 choose only negative apparent velocities, 0 choose both
        postive and negative apparent velocities, 1 choose only positive
        apparent velocities.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param mode: str. 'remove' for denoising and 'retain' for decomposition.
    :param scale_begin: int. The beginning scale to do coherent denoising.
    :param nbscales: int. Number of scales including the coarsest wavelet level.
        Default set to ceil(log2(min(M,N)) - 3).
    :param nbangles: int. Number of angles at the 2nd coarsest level,
        minimum 8, must be a multiple of 4.
    :return: numpy.ndarray. Denoised data.
    """
    if pad is None or pad is False:
        pad = 0
    dn = np.round(np.array(pad) * data.shape).astype(int)
    data_pd = padding(data, dn)

    C = fdct_wrapping(data_pd, is_real=False, finest=2, nbscales=nbscales,
                      nbangles_coarse=nbangles)

    # apply Gaussian denoising
    if choice in (0, 2):
        # define soft threshold
        if noise is None:
            E = _knee_points(data_pd, nbscales=nbscales, nbangles=nbangles)
        else:
            noise_pd = padding(noise,
                               np.array(data_pd.shape) - np.array(noise.shape))
            E = _noise_level(noise_pd, nbscales=nbscales, nbangles=nbangles)
        for s in range(1, len(C)):
            for w in range(len(C[s])):
                # first do a hard threshold
                C[s][w] = C[s][w] * (abs(C[s][w]) > abs(E[s][w]))
                if soft_thresh:
                    # soften the existing coefficients
                    C[s][w] = np.sign(C[s][w]) * (abs(C[s][w]) - abs(E[s][w]))

    # apply velocity filtering
    if choice in (1, 2):
        if dx is None or fs is None:
            msg = 'Please set both dx and fs.'
            raise ValueError(msg)

        vmin, vmax = v_range
        for s in range(scale_begin - 1, len(C) - 1):
            nbangles = len(C[s])
            velocity = _velocity_bin(nbangles, fs, dx)
            factors = _mask_factor(velocity, vmin, vmax, flag=flag, mode=mode)
            for w in range(nbangles):
                C[s][w] *= factors[w]

    # perform the inverse curvelet transform
    data_dn = ifdct_wrapping(C, is_real=True)

    return padding(data_dn, dn, reverse=True)
