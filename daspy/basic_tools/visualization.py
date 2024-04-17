# Purpose: Plot data
# Author: Minzhe Hu
# Date: 2024.4.17
# Email: hmz2018@mail.ustc.edu.cn
import numpy as np
import matplotlib.pyplot as plt


def plot(data, dx=None, fs=None, ax=None, obj='waveform', dpi=150, title=None,
         transpose=False, t0=0, x0=0, pick=None, f=None, k=None, t=None, c=None,
         cmap=None, vmin=None, vmax=None, xmode='distance', tmode='time',
         xlim=None, ylim=None, xlog=False, ylog=False, xinv=False, yinv=False,
         xaxis=True, yaxis=True, colorbar=True):
    """
    Plot several types of 2-D seismological data.

    :param data: numpy.ndarray. Data to plot.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param ax: Matplotlib.axes.Axes. Axes to plot. If not specified, the
        function will directly display the image using matplotlib.pyplot.show().
    :param obj: str. Type of data to plot. It should be one of 'waveform',
        'phasepick', 'spectrum', 'spectrogram', 'fk', or 'dispersion'.
    :param dpi: int. The resolution of the figure in dots-per-inch.
    :param title: str. The title of this axes.
    :param transpose: bool. Transpose the figure or not.
    :param t0, x0: The beginning of time and space.
    :param pick: Sequence of picked phases. Required if obj=='phasepick'.
    :param f: Sequence of frequency. Required if obj is one of 'spectrum',
        'spectrogram', 'fk' or 'dispersion'.
    :param k: Wavenumber sequence. Required if obj=='fk'.
    :param t: Time sequence. Required if obj=='spectrogram'.
    :param c: Phase velocity sequence. Required if obj=='dispersion'.
    :param cmap: str or Colormap. The Colormap instance or registered colormap
        name used to map scalar data to colors.
    :param vmin, vmax: Define the data range that the colormap covers.
    :param xmode: str. 'distance' or 'channel'.
    :param tmode: str. 'time' or 'sampling'.
    :param xlim, ylim: Set the x-axis and y-axis view limits.
    :param xlog, ylog: bool. If True, set the x-axis' or y-axis' scale as log.
    :param xinv, yinv: bool. If True, invert x-axis or y-axis.
    :param xaxis, yaxis: bool. Show ticks and labels for x-axis or y-axis.
    :param colorbar: bool, str or Matplotlib.axes.Axes. Bool means plot colorbar
        or not. Str means the location of colorbar. Axes means the Axes into
        which the colorbar will be drawn.
    """
    nch, nt = data.shape
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 5), dpi=dpi)
        show = True
    else:
        show = False

    if obj in ['waveform', 'phasepick']:
        if not cmap:
            cmap = 'RdBu'
        if vmax is None:
            vmax = np.percentile(abs(data), 80)
        if vmin is None:
            vmin = -vmax
        origin = 'upper'
        if fs is None or tmode == 'sampling':
            ylabel = 'Sampling points'
            fs = 1
        elif tmode == 'time':
            ylabel = 'Time (s)'

        if dx is None or xmode == 'channel':
            xlabel = 'Channel'
            extent = [x0, x0 + nch, t0 + nt / fs, t0]
        elif xmode == 'distance':
            xlabel = 'Disitance (km)'
            extent = [x0 * 1e-3, (x0 + nch * dx) * 1e-3, t0 + nt / fs, t0]

        if obj == 'phasepick':
            if len(pick) != nch:
                raise ValueError(
                    "Number of picks must be the same as channels.")
            for i, pk in enumerate(pick):
                if isinstance(pk, (tuple, list, np.ndarray)):
                    n = len(pk)
                    if n != 0:
                        ax.scatter(n * [(x0 + i * dx) * 1e-3], t0 + np.array(pk)
                                   / fs, marker=',', s=0.1, c='black')
                else:
                    ax.scatter((x0 + i * dx) * 1e-3, t0 + np.array(pk) / fs,
                               marker=',', s=0.1, c='black')

    if obj in ['spectrum', 'spectrogram', 'fk', 'dispersion']:
        data = abs(data)
        if not cmap:
            cmap = 'jet'
        if vmax is None:
            vmax = np.percentile(data, 80)
        if vmin is None:
            vmin = np.percentile(data, 20)
        if obj == 'spectrum':
            origin = 'lower'
            if dx is None or xmode == 'channel':
                xlabel = 'Channel'
                extent = [x0, x0 + nch, f.min(), f.max()]
            elif xmode == 'distance':
                xlabel = 'Disitance (km)'
                extent = [x0 * 1e-3, (x0 + nch * dx) * 1e-3, f.min(), f.max()]
            ylabel = 'Frequency (Hz)'
        elif obj == 'spectrogram':
            data = data.T
            origin = 'lower'
            xlabel = 'Time (s)'
            ylabel = 'Frequency (Hz)'
            extent = [t0 + min(t), t0 + max(t), min(f), max(f)]
        elif obj == 'fk':
            origin = 'lower'
            xlabel = 'Wavenumber (m$^{-1}$)'
            ylabel = 'Frequency (Hz)'
            extent = [min(k), max(k), min(f), max(f)]
        elif obj == 'dispersion':
            data = data.T
            origin = 'lower'
            xlabel = 'Frequency (Hz)'
            ylabel = 'Phase Velocity (m/s)'
            extent = [min(f), max(f), min(c), max(c)]

    if transpose:
        if origin == 'lower':
            extent = [extent[2], extent[3], extent[0], extent[1]]
        else:
            origin = 'lower'
            extent = [extent[3], extent[2], extent[0], extent[1]]
        (xlabel, ylabel) = (ylabel, xlabel)
        data = data.T

    bar = ax.imshow(data.T, vmin=vmin, vmax=vmax, extent=extent, aspect='auto',
                    origin=origin, cmap=cmap)
    if title:
        ax.set_title(title)
    if xaxis:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])
    if yaxis:
        ax.set_ylabel(ylabel)
    else:
        ax.set_yticklabels([])
    if xinv:
        ax.invert_xaxis()
    if yinv:
        ax.invert_yaxis()
    if ylim:
        ax.set_ylim(ylim)
    if xlim:
        ax.set_xlim(xlim)
    if xlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
    if colorbar:
        if colorbar is True:
            plt.colorbar(bar, ax=ax, location='right')
        elif isinstance(colorbar, (str, bool)):
            plt.colorbar(bar, ax=ax, location=colorbar)
        else:
            plt.colorbar(bar, cax=colorbar)

    if show:
        plt.show()
    else:
        return ax
