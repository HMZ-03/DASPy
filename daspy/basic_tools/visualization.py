# Purpose: Plot data
# Author: Minzhe Hu
# Date: 2024.11.8
# Email: hmz2018@mail.ustc.edu.cn
import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence


def plot(data: np.ndarray, dx=None, fs=None, ax=None, obj='waveform', dpi=150,
         title=None, transpose=False, t0=0, x0=0, pick=None, f=None, k=None,
         t=None, c=None, cmap=None, vmin=None, vmax=None, dB=False,
         xmode='distance', tmode='time', xlim=None, ylim=None, xlog=False,
         ylog=False, xinv=False, yinv=False, xlabel=True, ylabel=True,
         xticklabels=True, yticklabels=True, colorbar=True, colorbar_label=None,
         savefig=None):
    """
    Plot several types of 2-D seismological data.

    :param data: numpy.ndarray. Data to plot.
    :param dx: Channel interval in m.
    :param fs: Sampling rate in Hz.
    :param ax: Matplotlib.axes.Axes or tuple. Axes to plot. A tuple for new
        figsize. If not specified, the function will directly display the image
        using matplotlib.pyplot.show().
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
    :param dB: bool. Transfer data unit to dB and take 1 as the reference value.
    :param xmode: str. 'distance' or 'channel'.
    :param tmode: str. 'time' or 'sampling'.
    :param xlim, ylim: Set the x-axis and y-axis view limits.
    :param xlog, ylog: bool. If True, set the x-axis' or y-axis' scale as log.
    :param xlabel, yinv: bool. If True, invert x-axis or y-axis.
    :param xlabel, ylabel: bool or str. Whether to plot a label or what label to
        plot for x-axis or y-axis.
    :param xticklabels, yticklabels: bool or sequence of str. Whether to plot
        ticklabels or what ticklabels to plot for x-axis or y-axis.
    :param colorbar: bool, str or Matplotlib.axes.Axes. Bool means plot colorbar
        or not. Str means the location of colorbar. Axes means the Axes into
        which the colorbar will be drawn.
    :param savefig: str or bool. Figure name to save if needed. If True,
        it will be set to parameter obj.
    """
    nch, nt = data.shape
    if ax is None:
        ax = (6, 5)
    if isinstance(ax, tuple):
        fig, ax = plt.subplots(1, figsize=ax, dpi=dpi)
        show = True
    else:
        show = False

    if obj in ['waveform', 'phasepick']:
        cmap = 'RdBu' if cmap is None else cmap
        vmax = np.percentile(abs(data), 80) if vmax is None else vmax
        vmin = -vmax if vmin is None else vmin
        origin = 'upper'
        if fs is None or tmode == 'sampling':
            ylabel_default = 'Sampling points'
            fs = 1
        elif tmode == 'time':
            ylabel_default = 'Time (s)'

        if dx is None or xmode.lower() == 'channel':
            xlabel_default = 'Channel'
            extent = [x0, x0 + nch, t0 + nt / fs, t0]
        elif xmode.lower() == 'distance':
            xlabel_default = 'Disitance (km)'
            extent = [x0 * 1e-3, (x0 + nch * dx) * 1e-3, t0 + nt / fs, t0]

        if obj == 'phasepick' and len(pick):
            pck = np.array(pick).astype(float)
            if xmode.lower() == 'distance':
                pck[:, 0] = (x0 + pck[:, 0] * dx) * 1e-3
            elif xmode.lower() == 'channel':
                pck[:, 0] = x0 + pck[:, 0]
            if tmode.lower() == 'sampling':
                pck[:, 1] = pck[:, 1] / fs
            ax.scatter(pck[:,0], t0 + pck[:,1], marker=',', s=0.1, c='black')

    elif obj in ['spectrum', 'spectrogram', 'fk', 'dispersion']:
        if isinstance(data[0,0], (complex, np.complex64)):
            data = abs(data)
        if dB:
            data = 20 * np.log10(data)
        cmap = 'jet' if cmap is None else cmap
        vmax = np.percentile(abs(data), 80) if vmax is None else vmax
        vmin = np.percentile(abs(data), 20) if vmin is None else vmin
        if obj == 'spectrum':
            origin = 'lower'
            if dx is None or xmode.lower() == 'channel':
                xlabel_default = 'Channel'
                extent = [x0, x0 + nch, min(f), max(f)]
            elif xmode.lower() == 'distance':
                xlabel_default = 'Disitance (km)'
                extent = [x0 * 1e-3, (x0 + nch * dx) * 1e-3, min(f), max(f)]
            ylabel_default = 'Frequency (Hz)'
        elif obj == 'spectrogram':
            data = data.T
            origin = 'lower'
            xlabel_default = 'Time (s)'
            ylabel_default = 'Frequency (Hz)'
            extent = [t0 + min(t), t0 + max(t), min(f), max(f)]
        elif obj == 'fk':
            origin = 'lower'
            xlabel_default = 'Wavenumber (m$^{-1}$)'
            ylabel_default = 'Frequency (Hz)'
            extent = [min(k), max(k), min(f), max(f)]
        elif obj == 'dispersion':
            data = data.T
            origin = 'lower'
            xlabel_default = 'Frequency (Hz)'
            ylabel_default = 'Velocity (m/s)'
            extent = [min(f), max(f), min(c), max(c)]

    if transpose:
        if origin == 'lower':
            extent = [extent[2], extent[3], extent[0], extent[1]]
        else:
            origin = 'lower'
            extent = [extent[3], extent[2], extent[0], extent[1]]
        (xlabel_default, ylabel_default) = (ylabel_default, xlabel_default)
        data = data.T

    xlabel = xlabel if isinstance(xlabel, str) else \
            xlabel_default if xlabel else None
    ylabel = ylabel if isinstance(ylabel, str) else \
            ylabel_default if ylabel else None

    bar = ax.imshow(data.T, vmin=vmin, vmax=vmax, extent=extent, aspect='auto',
                    origin=origin, cmap=cmap)
    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if isinstance(xticklabels, Sequence):
        ax.set_xticklabels(xticklabels)
    elif not xticklabels:
        ax.set_xticklabels([])
    
    if isinstance(yticklabels, Sequence):
        ax.set_yticklabels(yticklabels)
    elif not yticklabels:
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
            cbar = plt.colorbar(bar, ax=ax, location='right')
        elif isinstance(colorbar, str):
            cbar = plt.colorbar(bar, ax=ax, location=colorbar)
        else:
            cbar = plt.colorbar(bar, cax=colorbar)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)

    if savefig:
        if not isinstance(savefig, str):
            savefig = obj + '.png'
        plt.tight_layout()
        plt.savefig(savefig)
        plt.close()
    elif show:
        plt.show()
    else:
        return ax
