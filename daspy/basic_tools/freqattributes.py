# Purpose: Analyze frequency attribute and transform in frequency domain
# Author: Minzhe Hu
# Date: 2024.6.8
# Email: hmz2018@mail.ustc.edu.cn
import numpy as np
from numpy.fft import rfft, rfft2, fftshift, fftfreq, rfftfreq
from scipy.signal import stft
from daspy.basic_tools.preprocessing import demeaning, detrending, cosine_taper


def next_pow_2(i):
    """
    Find the next power of two.

    :param i: float or int.
    :return: int. The next power of two for i.
    """
    buf = np.ceil(np.log2(i))
    return np.power(2, buf).astype(int)


def spectrum(data, fs, taper=0.05, nfft='default'):
    """
    Computes the spectrum of the given data.

    :param data: numpy.ndarray. Data to make spectrum of.
    :param fs: Sampling rate in Hz.
    :param taper: Decimal percentage of Tukey taper.
    :param nfft: Number of points for FFT. None = sampling points, 'default' =
        next power of 2 of sampling points.
    :return: Spectrum and frequency sequence.
    """
    if len(data.shape) == 1:
        data = data.reshape(1, len(data))
    elif len(data.shape) != 2:
        raise ValueError("Data should be 1-D or 2-D array")
    data = cosine_taper(data, (0, taper))

    if nfft == 'default':
        nfft = next_pow_2(len(data[0]))
    elif nfft is None:
        nfft = len(data[0])

    spec = rfft(data, n=nfft, axis=1)
    f = rfftfreq(nfft, d=1 / fs)

    return spec, f


def spectrogram(data, fs, nperseg=256, noverlap=None, nfft=None, detrend=False,
                boundary='zeros'):
    """
    Computes the spectrogram of the given data.

    :param data: 1-D or 2-D numpy.ndarray. Data to make spectrogram of.
    :param fs: Sampling rate in Hz.
    :param nperseg: int. Length of each segment.
    :param noverlap: int. Number of points to overlap between segments. If None,
        noverlap = nperseg // 2.
    :param nfft: int. Length of the FFT used. None = nperseg.
    :param detrend : str or bool. Specifies whether and how to detrend each
        segment.  'linear' or 'detrend' or True = detrend, 'constant' or
        'demean' = demean.
    :param boundary: str or None. Specifies whether the input signal is extended
        at both ends, and how to generate the new values, in order to center the
        first windowed segment on the first input point. This has the benefit of
        enabling reconstruction of the first input point when the employed
        window function starts at zero. Valid options are ['even', 'odd',
        'constant', 'zeros', None].
    :return: Spectrogram, frequency sequence and time sequence.
    """
    if detrend in [True, 'linear', 'detrend']:
        detrend = detrending
    elif detrend in ['constant', 'demean']:
        detrend = demeaning
    if data.ndim == 1:
        f, t, Zxx = stft(data, fs=fs, nperseg=nperseg, noverlap=noverlap,
                         nfft=nfft, detrend=detrend, boundary=boundary)
    elif len(data) == 1:
        f, t, Zxx = stft(data[0], fs=fs, nperseg=nperseg, noverlap=noverlap,
                         nfft=nfft, detrend=detrend, boundary=boundary)
    else:
        Zxx = []
        for d in data:
            f, t, Zxxi = stft(d, fs=fs, nperseg=nperseg, noverlap=noverlap,
                              nfft=nfft, detrend=detrend, boundary=boundary)
            Zxx.append(abs(Zxxi))
        Zxx = np.mean(np.array(Zxx), axis=0)

    return Zxx, f, t


def fk_transform(data, dx, fs, taper=(0, 0.05), nfft='default'):
    """
    Transform the data to the fk domain using 2-D Fourier transform method.

    :param data: numpy.ndarray. Data to do fk transform.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param taper: float or sequence of floats. Each float means decimal
        percentage of Tukey taper for corresponding dimension (ranging from 0 to
        1). Default is 0.1 which tapers 5% from the beginning and 5% from the
        end.
    :param nfft: Number of points for FFT. None means sampling points; 'default'
        means next power of 2 of sampling points, which makes result smoother.
    """
    nch, nt = data.shape
    data = cosine_taper(data, taper)
    if nfft == 'default':
        nfft = (next_pow_2(nch), next_pow_2(nt))
    elif not nfft:
        nfft = (nch, nt)

    fk = fftshift(rfft2(data, s=nfft), axes=0)
    f = rfftfreq(nfft[1], d=1. / fs)
    k = fftshift(fftfreq(nfft[0], d=dx))
    return fk, f, k
