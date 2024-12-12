# Purpose: Module for handling Section objects.
# Author: Minzhe Hu
# Date: 2024.12.12
# Email: hmz2018@mail.ustc.edu.cn
import warnings
import os
import numpy as np
from copy import deepcopy
from typing import Iterable
from datetime import datetime
from daspy.core.dasdatetime import DASDateTime, utc
from daspy.core.write import write
from daspy.basic_tools.visualization import plot
from daspy.basic_tools.preprocessing import (phase2strain, normalization,
                                             demeaning, detrending, stacking,
                                             cosine_taper, downsampling,
                                             padding, trimming,
                                             time_integration,
                                             time_differential,
                                             distance_integration)
from daspy.basic_tools.filter import (bandpass, bandstop, lowpass,
                                      lowpass_cheby_2, highpass, envelope)
from daspy.basic_tools.freqattributes import (spectrum, spectrogram,
                                              fk_transform)
from daspy.advanced_tools.channel import channel_checking, turning_points
from daspy.advanced_tools.denoising import (curvelet_denoising,
                                            common_mode_noise_removal,
                                            spike_removal)
from daspy.advanced_tools.decomposition import fk_filter, curvelet_windowing
from daspy.advanced_tools.strain2vel import (slant_stacking, fk_rescaling,
                                             curvelet_conversion)


class Section(object):
    def __init__(self, data, dx, fs, start_channel=0, start_distance=0,
                 start_time=0, **kwargs):
        """
        :param data: numpy.ndarray. Data recorded by DAS.
        :param dx: number. Channel interval in m.
        :param fs: number. Sampling rate in Hz.
        :param start_channel: int. Channel number of the first channel.
        :param start_distance: number. Distance of the first channel, in m.
        :param start_time: number or DASDateTime. Time of the first
            sampling point. If number, the unit is s.
        :param origin_time: number or DASDateTime. Ocurance time of the event.
        :param gauge_length: number. Gauge length in m.
        :param data_type: str. Can be 'phase shift', 'phase change rate',
            'strain', 'strain rate', 'displacement', 'velocity', 'acceleration',
            or normalized above parameters.
        :param scale: number. Scale or gain of data.
        :param geometry: numpy.ndarray. Should include latitude and longitude
            (first two columns), and can also include depth (last column).
        :param turning_channels: sequnce of channel numbers. Channel numbers of
            turning points.
        :param headers: dict. Other headers.
        :param source: str or pathlib.PosixPath. Path to the source file.
        :param source_type: str. Raw type it read from.
        """
        if data.ndim == 1:
            data = data[np.newaxis, :]
        self.data = data
        self.dx = dx
        self.fs = fs
        self.start_channel = start_channel
        self.start_distance = start_distance
        self.start_time = start_time
        opt_attrs = ['origin_time', 'gauge_length', 'data_type', 'scale',
                     'geometry', 'turning_channels', 'headers', 'source',
                     'source_type']
        for attr in opt_attrs:
            if attr in kwargs:
                setattr(self, attr, kwargs.pop(attr))

    def __str__(self):
        n = max(map(len, self.__dict__.keys()))
        describe = '{}: shape{}\n'.format('data'.rjust(n), self.data.shape)
        for key, value in self.__dict__.items():
            if key == 'data':
                continue
            if key == 'geometry':
                describe += '{}: shape{}\n'.format(key.rjust(n), value.shape)
            elif key in ['dx', 'start_distance', 'gauge_length']:
                describe += '{}: {} m\n'.format(key.rjust(n), value)
            elif key == 'fs':
                describe += '{}: {} Hz\n'.format(key.rjust(n), value)
            elif key == 'start_time':
                if isinstance(value, DASDateTime):
                    describe += '{}: {}\n'.format(key.rjust(n), value)
                else:
                    describe += '{}: {} s\n'.format(key.rjust(n), value)
            else:
                describe += '{}: {}\n'.format(key.rjust(n), value)
        return describe

    __repr__ = __str__

    def __add__(self, other):
        """
        Join two sections in time.
        """
        out = self.copy()
        if isinstance(other, Section):
            if other.dx != self.dx:
                if self.dx is None:
                    out.dx = other.dx
                elif other.dx is not None:
                    raise ValueError('These two Sections have different dx, '
                                     'please check.')
            if other.fs != self.fs:
                if self.fs is None:
                    out.fs = other.fs
                elif other.fs is not None:
                    raise ValueError('These two Sections have different fs, '
                                     'please check.')
            if isinstance(self.start_time, DASDateTime) and \
                isinstance(other.start_time, DASDateTime):
                if abs(other.start_time - self.end_time) > 0.1:
                    if abs(other.end_time - self.start_time) <= 0.1:
                        warnings.warn('According to the time information of the'
                                      ' two Sections, the order of addition is '
                                      'reversed.')
                        return other + self
                    else:
                        warnings.warn('The start time of the second Section '
                                      f'({other.start_time}) is inconsistent '
                                      'with the end time of the first Section ('
                                      f'{self.end_time}).')
            data = other.data
        elif isinstance(other, np.ndarray):
            data = other
        elif isinstance(other, list):
            data = np.array(other)
        else:
            raise TypeError('The input should be Section or np.ndarray.')

        if len(data) != self.nch:
            if len(data[0]) == self.nch:
                data = data.T
            else:
                raise ValueError('These two Sections have different number of '
                                 'channels, please check.')
        if out.data is None:
            out.data = data
        else:
            out.data = np.hstack((out.data, data))

        return out

    @property
    def shape(self):
        return self.data.shape

    @property
    def dt(self):
        return 1 / self.fs

    @property
    def nch(self):
        return len(self.data)

    @property
    def nt(self):
        return self.data.shape[1]

    @property
    def end_channel(self):
        return self.start_channel + self.nch - 1

    @property
    def distance(self):
        return self.nch * self.dx

    @property
    def end_distance(self):
        return self.start_distance + self.nch * self.dx

    @property
    def duration(self):
        return self.nt / self.fs

    @property
    def end_time(self):
        return self.start_time + self.nt / self.fs

    def copy(self):
        return deepcopy(self)

    @classmethod
    def from_obspy_stream(cls, st, channel_no='auto'):
        """
        Construct a Section from a obspy.core.stream.Stream instance.

        :param patch: obspy.core.stream.Stream. An instance of
            obspy.core.stream.Stream for construction.
        :param channel_no: None or str. None for no channel number information,
            'channel', 'station' for use the channel or station information of
            each trace and 'auto' for automatically detect.
        """
        stime = min([tr.stats['starttime'] for tr in st])
        etime = max([tr.stats['endtime'] for tr in st])
        st.trim(starttime=stime, endtime=etime, pad=True, fill_value=np.nan)
        matadata = [(tr.stats['sampling_rate'], tr.stats['delta'],
                     tr.stats['npts'], tr.stats['calib']) for tr in st]
        assert len(set(matadata)) == 1, ('The metadata of all traces in the '
                                         'stream should be the same.')
        nch = len(st)
        nt = st[0].stats.npts
        fs = st[0].stats.sampling_rate
        start_time = DASDateTime.from_datetime(st[0].stats.starttime.datetime).\
            replace(tzinfo=utc)
        scale = st[0].stats.calib
        source = type(st)
        data = np.zeros((nch, nt))

        if channel_no == 'channel':
            channel_no = np.array([int(tr.stats.channel) for tr in st])
        elif channel_no == 'station':
            channel_no = np.array([int(tr.stats.station) for tr in st])
        elif channel_no:
            if str.isdigit(st[0].stats.channel):
                channel_no = np.array([int(tr.stats.channel) for tr in st])
            elif str.isdigit(st[0].stats.station):
                channel_no = np.array([int(tr.stats.station) for tr in st])
        else:
            channel_no = np.arange(nch).astype(int)
        start_channel = min(channel_no)
        channel_no -= start_channel
        data = np.zeros((max(channel_no) + 1, nt))
        for i, tr in enumerate(st):
            data[channel_no[i]] = tr.data

        warnings.warn('obspy.core.stream.Stream doesn\'t include channel '
                      'interval. Please set dx manually.')
        return cls(data.astype(float), None, fs, start_channel=start_channel,
                   start_time=start_time, scale=scale, source=source)

    @classmethod
    def from_dascore_patch(cls, patch):
        """
        Construct a Section from a dascore.core.patch.Patch instance.

        :param patch: dascore.core.patch.Patch. An instance of
            dascore.core.patch.Patch for construction.
        :return: daspy.Section.
        """
        kwargs = {}
        if patch.dims == ('time', 'distance'):
            data = patch.data.T
            dx = patch.coords.coord_map['distance'].step
            kwargs['start_distance'] = patch.coords.coord_map['distance'].start
        elif patch.dims == ('time', 'channel'):
            data = patch.data.T
            dx = 1
            warnings.warn('This dascore.core.patch.Patch instance doesn\'t '
                          'include channel interval. Set dx to 1.')
            kwargs['start_channel'] = patch.coords.coord_map['channel'].start
        elif patch.dims == ('distance', 'time'):
            data = patch.data
            dx = patch.coords.coord_map['distance'].step
            kwargs['start_distance'] = patch.coords.coord_map['distance'].start
        elif patch.dim == ('channel', 'time'):
            data = patch.data
            dx = 1
            warnings.warn('This dascore.core.patch.Patch instance doesn\'t '
                          'include channel interval. Set dx to 1.')
            kwargs['start_channel'] = patch.coords.coord_map['channel'].start

        if isinstance(patch.coords.coord_map['time'].step, np.timedelta64):
            start_time = DASDateTime.fromtimestamp(
                patch.coords.coord_map['time'].start.item() / 1e9, tz=utc)
            fs = np.timedelta64(1, 's') / patch.coords.coord_map['time'].step
        else:
            start_time = patch.coords.coord_map['time'].start
            fs = patch.coords.coord_map['time'].step

        if hasattr(patch.attrs, 'gauge_length'):
            kwargs['gauge_length'] = patch.attrs.gauge_length
        if patch.attrs.data_type:
            kwargs['data_type'] = ' '.join(patch.attrs.data_type.split('_'))
        sec = cls(data, dx, fs, start_time=start_time,
                  headers=patch.attrs, source=type(patch), **kwargs)
        return sec

    @classmethod
    def from_lightguide_blast(cls, blast):
        """
        Construct a Section from a lightguide.blast.Blast instance.

        :param blast: lightguide.blast.Blast. An instance of
            lightguide.blast.Blast for construction.
        :return: daspy.Section.
        """
        sec = cls(blast.data, blast.channel_spacing, blast.sampling_rate,
                  start_time=DASDateTime.from_datetime(blast.start_time),
                  start_channel=blast.start_channel, data_type=blast.unit,
                  source=type(blast))
        return sec

    def to_obspy_stream(self):
        """
        Construct an instance of obspy.core.stream.Stream.

        :return: obspy.core.stream.Stream.
        """
        from obspy import Stream, Trace, UTCDateTime
        st = Stream()
        header = {'sampling_rate': self.fs}
        if not isinstance(datetime, self.start_time):
            warnings.warn('The type of start_time is not DASDateTime. The '
                          'starttime of Trace instances may be wrong')
            header['starttime'] = UTCDateTime(self.start_time)
        else:
            header['starttime'] = UTCDateTime(self.start_time.timestamp())
        if hasattr(self, 'scale'):
            header['calib'] = self.scale
        for i in range(self.nch):
            header_tr = deepcopy(header)
            header_tr['channel'] = str(self.start_channel + i)
            tr = Trace(self.data[i], header_tr)
            st += tr

        warnings.warn('obspy.core.stream.Stream doesn\'t include channel '
                      'interval.')
        return st

    def to_dascore_patch(self):
        """
        Construct an instance of dascore.core.patch.Patch.

        :return: dascore.core.patch.Patch.
        """
        from pint import Quantity
        from datetime import datetime
        from dascore.core import Patch, CoordManager
        from dascore.core.coords import CoordRange
        from dascore.utils.mapping import FrozenDict

        dims = ('time', 'distance')
        if isinstance(self.start_time, datetime):
            if self.start_time.tzinfo:
                stime = np.datetime64(self.start_time.astimezone(utc).
                                      replace(tzinfo=None))
                etime = np.datetime64(self.end_time.astimezone(utc).
                                      replace(tzinfo=None))
            else:
                stime = np.datetime64(self.start_time.replace(tzinfo=None))
                etime = np.datetime64(self.end_time.replace(tzinfo=None))

            time_range = CoordRange(units=(Quantity(1, 'second')),
                                    step=np.timedelta64(int(self.dt * 1e9),
                                                        'ns'),
                                    start=stime, stop=etime)
        else:
            stime = self.start_time
            etime = self.end_time
            time_range = CoordRange(units=(Quantity(1, 'second')), step=self.dt,
                                    start=stime, stop=etime)

        dist_range = CoordRange(units=(Quantity(1, 'meter')), step=self.dx,
                                start=self.start_distance,
                                stop=self.end_distance)
        coord_map = FrozenDict({'time': time_range, 'distance': dist_range})
        dim_map = FrozenDict({'time': ('time',), 'distance': ('distance', )})
        coords = CoordManager(dims=dims, coord_map=coord_map,
                              dim_map=dim_map)

        if self.source == Patch:
            attrs = self.headers
        else:
            from dascore.io.prodml.core import ProdMLPatchAttrs
            attrs = ProdMLPatchAttrs(coords=coords)
            kwargs = {}
            if hasattr(self, 'data_type'):
                kwargs['data_type'] = '_'.join(self.data_type.split(' '))
            if hasattr(self, 'gauge_length'):
                kwargs['gauge_length'] = self.gauge_length
                kwargs['gauge_length_units'] = Quantity(1, 'meter')

        return Patch(self.data.T, coords, dims, attrs)

    def to_lightguide_blast(self):
        """
        Construct an instance of lightguide.blast.Blast

        :return: lightguide.blast.Blast.
        """
        from lightguide.blast import Blast
        if not isinstance(self.start_time, datetime):
            warnings.warn('The type of start_time is not DASDateTime. The '
                          'starttime of Trace instances may be wrong')
            start_time = datetime.fromtimestamp(self.start_time)
        elif isinstance(self.start_time, DASDateTime):
            start_time = self.start_time.to_datetime()
        if start_time.tzinfo is None:
            start_time = start_time.astimezone(utc)
        if hasattr(self, 'data_type'):
            for key in ['strain rate', 'strain', 'displacement', 'velocity',
                        'acceleration']:
                if key in self.data_type.lower():
                    if self.data_type.lower() == key:
                        unit = key
                    else:
                        warnings.warn(f'Data type {self.data_type} is not '
                                      f'supported in lightguide. Set to {key}.')
                    break
                else:
                    warnings.warn(f'Data type {unit} is not supported in '
                                  'lightguide. Set to default (starin rate).')
                    unit = 'strain rate'
        else:
            print('Set unit to default (starin rate).')
            unit = 'strain rate'
        return Blast(self.data, start_time, self.fs,
                     start_channel=self.start_channel, channel_spacing=self.dx,
                     unit=unit)

    def save(self, fname=None, ftype=None, keep_format=False):
        """
        Save the instance as a pickle file or update the raw file and resave as
        new file.

        :param fname: str or pathlib.PosixPath. Path of new DAS data file to
            save.
        :param ftype: None or str. None for automatic detection), or 'pkl',
            'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy', 'npy'.
        :param keep_format: bool. If True, we will make a copy of the
            self.source file and make changes to it. This will strictly preserve
            the original format, but will cost more IO resources.
        """
        if fname is None:
            if hasattr(self, 'source'):
                fname_list = self.source.split('.')
                fname_list[-2] += '_new'
                fname = '.'.join(fname_list)
            else:
                fname = 'section.pkl'

        if ftype is None:
            ftype = str(fname).lower().split('.')[-1]

        for rtp in [('pickle', 'pkl'), ('hdf5', 'h5'), ('segy', 'sgy')]:
            ftype = ftype.replace(*rtp)

        if keep_format:
            if not hasattr(self, 'source'):
                raise ValueError('self.source not exit.')
            if not os.path.isfile(self.source):
                raise ValueError('self.source is not a file.')
            if ftype != self.source_type:
                raise ValueError('self.source_type is different from ftype.')
            write(self, fname, ftype=ftype, raw_fname=self.source)
        else:
            write(self, fname, ftype=ftype)

        return self

    def channel_data(self, use_channel, replace=False):
        """
        Extract data of one channel or several channels.
        """
        channel = np.array(use_channel).astype(int)
        channel -= self.start_channel
        data = self.data[channel]
        if replace:
            self.data = data
            self.start_channel += channel[0]
            self.start_distance += channel[0] * self.dx
            return self
        else:
            return data

    def plot(self, xmode='distance', tmode='origin', obj='waveform',
             kwargs_pro={}, **kwargs):
        """
        Plot several types of 2-D seismological data.

        :param xmode: str. 'distance' or 'channel'.
        :param tmode: str. 'origin', 'start', 'time' or 'sampling'. If
            origin_time is not defined, 'origin' and 'start' is the same.
        :param obj: str. Type of data to plot. It should be one of 'waveform',
            'phasepick', 'spectrum', 'spectrogram', 'fk', or 'dispersion'.
        :param kwargs_pro: dict. If obj is one of 'spectrum', 'spectrogram',
            'fk' and data is not specified, this parameter will be used to
            process the data to plot.
        :param ax: Matplotlib.axes.Axes or tuple. Axes to plot. A tuple for new
            figsize. If not specified, the function will directly display the
            image using matplotlib.pyplot.show().
        :param dpi: int. The resolution of the figure in dots-per-inch.
        :param title: str. The title of this axes.
        :param transpose: bool. Transpose the figure or not.
        :param cmap: str or Colormap. The Colormap instance or registered
            colormap name used to map scalar data to colors.
        :param vmin, vmax: Define the data range that the colormap covers.
        :param xlim, ylim: Set the x-axis and y-axis view limits.
        :param dB: bool. Transfer data unit to dB and take 1 as the reference
            value.
        :param xlog, ylog: bool. If True, set the x-axis' or y-axis' scale as
            log.
        :param xinv, yinv: bool. If True, invert x-axis or y-axis.
        :param xaxis, yaxis: bool. Show ticks and labels for x-axis or y-axis.
        :param colorbar: bool, str or Matplotlib.axes.Axes. Bool means plot
            colorbar or not. Str means the location of colorbar. Axes means the
            Axes into which the colorbar will be drawn.
        :param t0, x0: The beginning of time and space. Use instance's
            properties by default
        :param pick: Sequence of picked phases. Required if obj=='phasepick'.
        :param c: Phase velocity sequence. Required if obj=='dispersion'.
        :param data: numpy.ndarray. Data to plot. Required if obj is not
            'spectrum', 'spectrogram' and 'fk'.
        :param f: Frequency sequence. Required if obj is one of 'spectrum',
            'spectrogram', 'fk' and data is specified, or obj is 'dispersion'.
        :param k: Wavenumber sequence. Required if obj=='fk' and data is
            specified.
        :param t: Time sequence. Required if obj=='spectrogram' and data is
            specified.
        :param savefig: str or bool. Figure name to save if needed. If True,
            it will be set to parameter obj.
        """
        if 'data' not in kwargs.keys():
            if obj == 'waveform':
                data = deepcopy(self.data)
            elif obj == 'spectrum':
                data, f = self.spectrum(**kwargs_pro)
                kwargs['f'] = f
            elif obj == 'spectrogram':
                data, f, t = self.spectrogram(**kwargs_pro)
                kwargs['f'] = f
                kwargs['t'] = t
            elif obj == 'fk':
                data, f, k = self.fk_transform(**kwargs_pro)
                kwargs['f'] = f
                kwargs['k'] = k
            if hasattr(self, 'scale'):
                data *= self.scale
        else:
            data = kwargs.pop('data')

        if 'ax' not in kwargs.keys() and 'title' not in kwargs.keys():
            kwargs['title'] = obj
            if hasattr(self, 'data_type'):
                kwargs['title'] += f' ({self.data_type})'

        if xmode == 'channel':
            kwargs.setdefault('x0', self.start_channel)
        elif xmode == 'distance':
            kwargs.setdefault('x0', self.start_distance)
        if tmode in ['origin', 'start', 'time']:
            kwargs.setdefault('t0', self.start_time)
            if tmode == 'origin':
                if hasattr(self, 'origin_time'):
                    kwargs['t0'] -= self.origin_time
                    kwargs.setdefault('ylabel', 'Times(s) after occurance')
                else:
                    tmode == 'start'
            if tmode == 'start':
                kwargs['t0'] -= self.start_time
            tmode = 'time'
        if hasattr(self, 'data_type'):
            kwargs.setdefault('colorbar_label', self.data_type)

        plot(data, self.dx, self.fs, obj=obj, xmode=xmode, tmode=tmode,
             **kwargs)

    def rescaling(self, scale=None):
        """
        Scale data according to a scale factor.

        :param scale: float. It is required if the Section instance does
            not specify the attribute 'scale'.
        """
        if scale is None:
            try:
                self.data *= self['scale']
            except ValueError:
                print('Please specify a scale factor.')
        else:
            if hasattr(self, 'scale') and self.scale != scale:
                warnings.warn('The set scale is different from the previous '
                              'self.scale.')
            self.data *= scale
        self.scale = 1
        return self

    def phase2strain(self, lam, e, n, gl=None):
        """
        Convert the optical phase shift in radians to strain, or phase change
        rate to strain rate.

        :param lam: float. Operational optical wavelength in vacuum.
        :param e: float. photo-slastic scaling factor for logitudinal strain in
            isotropic material.
        :param n: float. Refractive index of the sensing fiber.
        :paran gl: float. Gauge length. Required if self.gauge_length has not
            been set.
        """
        if gl:
            self.gauge_length = gl
        self.data = phase2strain(self.data, lam, e, n, self.gauge_length)
        if hasattr(self, 'data_type'):
            if 'phase' not in self.data_type:
                warnings.warn('The data type is {}, not phase shift. But it' +
                              'still takes effect.'.format(self.data_type))
            else:
                self.data_type = self.data_type.replace(
                    'phase shift', 'strain')
                self.data_type = self.data_type.replace('phase change rate',
                                                        'strain rate')
        return self

    def normalization(self, method='z-score'):
        """
        Normalize for each individual channel using Z-score method.

        :param method: str. Method for normalization, should be one of 'max' or
            'z-score'.
        """
        self.data = normalization(self.data, method=method)
        if hasattr(self, 'data_type'):
            self.data_type = 'normed ' + self.data_type
        return self

    def demeaning(self):
        """
        Demean signal by subtracted mean of each channel.
        """
        self.data = demeaning(self.data)
        return self

    def detrending(self):
        """
        Detrend signal by subtracted a linear least-squares fit to data.
        """
        self.data = detrending(self.data)
        return self

    def stacking(self, N, step=None):
        """
        Stack several channels to increase the signal-noise ratio(SNR).

        :param N: int. N adjacent channels stacked into 1.
        :param step: int. Interval of data stacking.
        """
        if step is None:
            step = N
        self.data = stacking(self.data, N, step=step)
        self.dx *= step
        if hasattr(self, 'gauge_length'):
            self.gauge_length += self.dx * (N - 1)
        return self

    def taper(self, *args, **kwargs):
        """
        See cosin_taper.
        """
        self.cosine_taper(*args, **kwargs)
        return self

    def cosine_taper(self, p=0.1, side='both'):
        """
        Taper using Tukey window.

        :param p: float or sequence of floats. Each float means decimal
            percentage of Tukey taper for corresponding dimension (ranging from
            0 to 1). Default is 0.1 which tapers 5% from the beginning and 5%
            from the end. If only one float is given, it only do for time
            dimension.
        :param side: str. 'both', 'left', or 'right'.
        """
        self.data = cosine_taper(self.data, p=p, side=side)
        return self

    def downsampling(self, xint=None, tint=None, stack=True,
                     lowpass_filter=True):
        """
        Downsample DAS data.

        :param xint: int. Spatial downsampling factor.
        :param tint: int. Time downsampling factor.
        :param stack: bool. If True, stacking will replace decimation.
        :param lowpass_filter: bool. Lowpass cheby2 filter before time
            downsampling or not.
        :return: Downsampled data.
        """
        self.data = downsampling(self.data, xint=xint, tint=tint, stack=stack,
                                 lowpass_filter=lowpass_filter)
        if xint and xint > 1:
            self.dx *= xint
            if hasattr(self, 'gauge_length'):
                self.gauge_length += self.dx * (xint - 1)
        if tint and tint > 1:
            self.fs /= tint
        return self

    def trimming(self, mode=1, xmin=None, xmax=None, tmin=None, tmax=None):
        """
        Cut data to given start and end distance/channel or time/sampling
        points.

        :param mode: int. 0 means the unit of boundary is channel number and
            sampling points; 1 means the unit of boundary is meters and seconds.
        :param xmin, xmax: int or float. Boundary of channel number (mode=0)
            or boundary of distance (mode=1).
            in meters.
        :param tmin, tmax: int, float or DASDateTime. Boundary of sampling
            points (mode=0) or boundary of time (mode=1). When mode=1,
            start_time is of type DASDateTime and tmin/tmax are floats,
            tmin/tmax means the time relative to start_time.
        """
        if mode == 1:
            if tmin is not None:
                try:
                    tmin = round((tmin - self.start_time) * self.fs)
                except TypeError:
                    tmin = round(tmin * self.fs)
                if tmin < 0:
                    warnings.warn('tmin is earlier than start_time. Set tmin '
                                  'to start_time.')
                    tmin = 0
                elif tmin >= self.nt:
                    raise ValueError('tmin is later than end_time.')
            else:
                tmin = 0

            if tmax is not None:
                try:
                    tmax = round((tmax - self.start_time) * self.fs)
                except TypeError:
                    tmax = round(tmax * self.fs)
                if tmax <= 0:
                    raise ValueError('tmax is earlier than start_time.')
                if tmax > self.nt:
                    warnings.warn('tmax is later than end_time. Set tmax to the'
                                  ' end_time.')
                    tmax = self.nt

            if xmin is not None:
                xmin = round((xmin - self.start_distance) / self.dx)
                if xmin < 0:
                    warnings.warn('xmin is smaller than start_distance. Set '
                                  'xmin to 0.')
                    xmin = 0
                elif xmin >= self.nch:
                    raise ValueError('xmin is later than end_distance.')
            else:
                xmin = 0

            if xmax is not None:
                xmax = round((xmax - self.start_distance) / self.dx)
                if xmax <= 0:
                    raise ValueError('xmax is smaller than start_distance.')
                if xmax > self.nch:
                    warnings.warn('xmax is later than end_distance. Set xmax '
                                  'to the array length.')
                    xmax = self.nch

        elif mode == 0:
            if tmin is None:
                tmin = 0
            else:
                tmin = int(tmin)
            if xmin is None:
                xmin = 0
            else:
                xmin = int(xmin - self.start_channel)
            if xmax is not None:
                xmax = int(xmax - self.start_channel)

        self.data = trimming(self.data, dx=self.dx, fs=self.fs, xmin=xmin,
                             xmax=xmax, tmin=tmin, tmax=tmax)

        self.start_time += tmin / self.fs
        self.start_distance += xmin * self.dx
        self.start_channel += xmin

        return self

    def padding(self, dn, reverse=False):
        """
        Pad DAS data with 0.

        :param dn: int or sequence of ints. Number of points to pad for both
            dimensions.
        :param reverse: bool. Set True to reverse the operation.
        """
        self.data = padding(self.data, dn, reverse=reverse)
        return self

    def _time_int_dif_attr(self, mode=0):
        for type_group in [['phase change rate', 'phase shift'],
                           ['strain rate', 'strain'],
                           ['acceleration', 'velocity', 'displacement']]:
            for (i, tp) in enumerate(type_group):
                if tp in self.data_type:
                    try:
                        self.data_type = self.data_type.replace(
                            tp, type_group[i + mode])
                    except BaseException:
                        operate = ('differentiate', 'integrate')[mode > 0]
                        print(f'Data type conversion error. Can not {operate} '
                              f'{self.data_type} data.')
                    return self
        warnings.warn('Unable to convert data type.')

    def time_integration(self, c=0):
        """
        Integrate DAS data in time.

        :param c: float. A constant added to the result.
        """
        self.data = time_integration(self.data, self.fs, c=c)
        if hasattr(self, 'data_type'):
            self._time_int_dif_attr(mode=1)
        return self

    def time_differential(self, prepend=0):
        """
        Differentiate DAS data in time.

        :param prepend: 'mean' or values to prepend to `data` along axis prior to
            performing the difference. 
        """
        self.data = time_differential(self.data, self.fs, prepend=prepend)
        if hasattr(self, 'data_type'):
            self._time_int_dif_attr(mode=-1)
        return self

    def distance_integration(self, c=0):
        """
        Differentiate DAS data in distance.

        :param c: float. A constant added to the result.
        """
        self.data = distance_integration(self.data, self.dx, c=c)
        self._strain2vel_attr()
        return self

    def bandpass(self, freqmin, freqmax, zi=None, **kwargs):
        """
        Filter data from 'freqmin' to 'freqmax' using Butterworth bandpass
        filter of 'corners' corners.

        :param freqmin: Pass band low corner frequency.
        :param freqmax: Pass band high corner frequency.
        :param zi : None, 0, or array_like. Initial conditions for the cascaded
            filter delays. It is a vector of shape (n_sections, nch, 2). Set to
            0 to trigger a output of the final filter delay values.
        :param corners: Filter corners / order.
        :param zerophase: If True, apply filter once forwards and once
            backwards. This results in twice the number of corners but zero
            phase shift in the resulting filtered data. Only valid when zi is
            None.
        """
        if zi is None:
            self.data = bandpass(self.data, self.fs, freqmin, freqmax, **kwargs)
            return self
        else:
            self.data, zf = bandpass(self.data, self.fs, freqmin, freqmax,
                                     zi=zi, **kwargs)
            return zf

    def bandstop(self, freqmin, freqmax, zi=None, **kwargs):
        """
        Filter data removing data between frequencies 'freqmin' and 'freqmax'
        using Butterworth bandstop filter of 'corners' corners.

        :param freqmin: Stop band low corner frequency.
        :param freqmax: Stop band high corner frequency.
        :param zi : None, 0, or array_like. Initial conditions for the cascaded
            filter delays. It is a vector of shape (n_sections, nch, 2). Set to
            0 to trigger a output of the final filter delay values.
        :param corners: Filter corners / order.
        :param zerophase: If True, apply filter once forwards and once
            backwards. This results in twice the number of corners but zero
            phase shift in the resulting filtered data. Only valid when zi is
            None.
        """

        if zi is None:
            self.data = bandstop(self.data, self.fs, freqmin, freqmax, **kwargs)
            return self
        else:
            self.data, zf = bandstop(self.data, self.fs, freqmin, freqmax,
                                     zi=zi, **kwargs)
            return zf

    def lowpass(self, freq, zi=None, **kwargs):
        """
        Filter data removing data over certain frequency 'freq' using
        Butterworth lowpass filter of 'corners' corners.

        :param freq: Filter corner frequency.
        :param zi : None, 0, or array_like. Initial conditions for the cascaded
            filter delays. It is a vector of shape (n_sections, nch, 2). Set to
            0 to trigger a output of the final filter delay values.
        :param corners: Filter corners / order.
        :param zerophase: If True, apply filter once forwards and once
            backwards. This results in twice the number of corners but zero
            phase shift in the resulting filtered data. Only valid when zi is
            None.
        """
        if zi is None:
            self.data = lowpass(self.data, self.fs, freq, **kwargs)
            return self
        else:
            self.data, zf = lowpass(self.data, self.fs, freq, zi=zi, **kwargs)
            return zf

    def lowpass_cheby_2(self, freq, **kwargs):
        """
        Filter data by passing data only below a certain frequency. The main
        purpose of this cheby2 filter is downsampling. This method will
        iteratively design a filter, whose pass band frequency is determined
        dynamically, such that the values above the stop band frequency are
        lower than -96dB.

        :param freq: The frequency above which signals are attenuated with 95
            dB.
        :param maxorder: Maximal order of the designed cheby2 filter.
        :param zi : None, 0, or array_like. Initial conditions for the cascaded
            filter delays. It is a vector of shape (n_sections, nch, 2). Set to 0 to
            trigger a output of the final filter delay values.
        :param ba: If True return only the filter coefficients (b, a) instead of
            filtering.
        :param freq_passband: If True return additionally to the filtered data,
            the iteratively determined pass band frequency.
        """
        output = lowpass_cheby_2(self.data, self.fs, freq, **kwargs)
        if isinstance(output, tuple):
            self.data = output[0]
            if len(output) == 2:
                return output[1]
            else:
                return output[1:]
        else:
            if kwargs.pop('ba', False):
                self.data = output
                return self
            else:
                return output

    def highpass(self, freq, zi=None, **kwargs):
        """
        Filter data removing data below certain frequency 'freq' using
        Butterworth highpass filter of 'corners' corners.

        :param freq: Filter corner frequency.
        :param zi : None, 0, or array_like. Initial conditions for the cascaded
            filter delays. It is a vector of shape (n_sections, nch, 2). Set to
            0 to trigger a output of the final filter delay values.
        :param corners: Filter corners / order.
        :param zerophase: If True, apply filter once forwards and once
            backwards. This results in twice the number of corners but zero
            phase shift in the resulting filtered data. Only valid when zi is
            None.
        """
        if zi is None:
            self.data = highpass(self.data, self.fs, freq, **kwargs)
            return self
        else:
            self.data, zf = highpass(self.data, self.fs, freq, zi=zi, **kwargs)
            return zf

    def envelope(self):
        """
        Computes the envelope of the given data. The envelope is determined by
        adding the squared amplitudes of the data and it's Hilbert-Transform and
        then taking the square-root. The envelope at the start/end should not be
        taken too seriously.
        """
        self.data = envelope(self.data)
        if hasattr(self, 'data_type'):
            self.data_type += ' envelope'
        return self

    def spectrum(self, taper=0.05, nfft='default'):
        """
        Computes the spectrum of the given data.

        :param taper: Decimal percentage of Tukey taper.
        :param nfft: Number of points for FFT. None = sampling points, 'default'
            = next power of 2 of sampling points.
        :return: Spectrum and frequency sequence.
        """
        return spectrum(self.data, self.fs, taper=taper, nfft=nfft)

    def spectrogram(self, **kwargs):
        """
        Computes the spectrogram of the given data.

        :param ch1, ch2, nch: int. Start channel, end channel and channel step
            for calculating the average spectrogram.
        :param nperseg: int. Length of each segment.
        :param noverlap: int. Number of points to overlap between segments. If
            None, noverlap = nperseg // 2.
        :param nfft: int. Length of the FFT used. None = nperseg.
        :param detrend : str or bool. Specifies whether and how to detrend each
            segment.  'linear' or 'detrend' or True = detrend, 'constant' or
            'demean' = demean.
        :param boundary: str or None. Specifies whether the input signal is
            extended at both ends, and how to generate the new values, in order
            to center the first windowed segment on the first input point. This
            has the benefit of enabling reconstruction of the first input point
            when the employed window function starts at zero. Valid options are
            ['even', 'odd', 'constant', 'zeros', None].
        :return: Spectrogram, frequency sequence and time sequence.
        """
        if 'ch1' in kwargs.keys():
            ch1 = int(kwargs.pop('ch1') - self.start_channel)
        else:
            ch1 = 0
        if 'ch2' in kwargs.keys():
            ch2 = int(kwargs.pop('ch2') - self.start_channel)
        else:
            ch2 = self.nch
        if 'nch' in kwargs.keys():
            nch = int(kwargs.pop('nch'))
        else:
            nch = 1

        return spectrogram(self.data[ch1:ch2:nch], self.fs, **kwargs)

    def fk_transform(self, **kwargs):
        """
        Transform the data to the fk domain using 2-D Fourier transform method

        :param taper: float or sequence of floats. Each float means decimal
            percentage of Tukey taper for corresponding dimension (ranging from
            0 to 1). Default is 0.1 which tapers 5% from the beginning and 5%
            from the end.
        :param nfft: Number of points for FFT. None means sampling points;
            'default' means next power of 2 of sampling points, which makes
            result smoother.
        """
        return fk_transform(self.data, self.dx, self.fs, **kwargs)

    def channel_checking(self, use=False, **kwargs):
        """
        Use the energy of each channel to determine which channels are bad.

        :param use: bool. If True, only keep the data of good channels in
            self.data and return self.
        :param deg: int. Degree of the fitting polynomial.
        :param thresh: int or float. The MAD multiple of bad channel energy
            lower than good channels.
        :param continuity: bool. Perform continuity checks on bad channels and
            good channels.
        :param adjacent: int. The number of nearby channels for continuity
            checks.
        :param toleration: int. The number of discontinuous channel allowed in
            each channel (including itself) in the continuity check.
        :param plot: bool or str. False means no plotting. Str or True means
            plotting while str gives a non-default filename.
        :return: self or Good channels and bad channels.
        """
        good_chn, bad_chn = channel_checking(self.data, **kwargs)
        if use:
            self.channel_data(good_chn, replace=True)
            return self
        else:
            return good_chn + self.start_channel, bad_chn + self.start_channel

    def turning_points(self, data_type='default', **kwargs):
        """
        Seek turning points in the DAS channel.

        :param data_type: str. If data_type is 'coordinate', data should include
            latitude and longitude (first two columns), and can also include
            depth (last column). If data_type is 'waveform', data should be
            continuous waveform, preferably containing signal with strong
            coherence (earthquake, traffic signal, etc.).
        :param thresh: For coordinate data, when the angle of the optical cables
            on both sides centered on a certain point exceeds thresh, it is
            considered an turning point. For waveform, thresh means the MAD
            multiple of adjacent channel cross-correlation values lower than
            their median.
        :param depth_info: bool. Optional if data_type is 'coordinate'. Whether
            depth (in meters) is included in the coordinate data and need to be
            used.
        :param channel_gap: int. Optional if data_type is 'coordinate'. The
            smaller the value is, the finer the segmentation will be. It is
            set to half the ratio of gauge length and channel interval by
            default.
        :return: list. Channel index of turning points.
        """
        if data_type == 'default':
            data_type = 'coordinate' if hasattr(self, 'geometry') else \
                'waveform'
        if data_type == 'coordinate':
            if hasattr(self, 'gauge_length'):
                kwargs.setdefault(
                    'channel_gap', self.gauge_length / self.dx / 2)
            if 'data' in kwargs.keys():
                output = turning_points(data_type=data_type, **kwargs)
            elif hasattr(self, 'geometry'):
                output = turning_points(self.geometry, data_type=data_type,
                                        **kwargs)
            else:
                raise ValueError('Geometry needs to be defined in DASdata, or '
                                 'coordinate data should be given.')
        else:
            output = turning_points(self.data, data_type=data_type, **kwargs)

        if isinstance(output, tuple):
            output = np.array(list(set(output[0]) | set(output[1])))
        else:
            output = np.array(output)
        output += self.start_channel
        self.turning_channels = output
        return output

    def spike_removal(self, nch=50, nsp=5, thresh=10):
        """
        Use a median filter to remove high-strain spikes in the data.

        :param nch: int. Number of channels over which to compute the median.
        :param nsp: int. Number of sampling points over which to compute the
            median.
        :param thresh: Ratio threshold over the median over which a number is
            considered to be an outlier.
        """
        self.data = spike_removal(self.data, nch=nch, nsp=nsp, thresh=thresh)
        return self

    def common_mode_noise_removal(self):
        """
        Remove common mode noise (sometimes called horizontal noise) from data.
        """
        self.data = common_mode_noise_removal(self.data)
        return self

    def curvelet_denoising(self, **kwargs):
        """
        Use curevelet transform to filter stochastic or/and cooherent noise.

        :param choice: int. 0 for Gaussian denoising using soft thresholding, 1
            for velocity filtering using the standard FK methodology and 2 for
            both.
        :param pad: float or sequence of floats. Each float means padding
            percentage before FFT for corresponding dimension. If set to 0.1
            will pad 5% before the beginning and after the end.
        :param noise: numpy.ndarray. Noise record as reference.
        :param soft_thresh: bool. True for soft thresholding and False for hard
            thresholding.
        :param vmin, vmax: float. Velocity range in m/s.
        :param flag: -1 choose only negative apparent velocities, 0 choose both
            postive and negative apparent velocities, 1 choose only positive
            apparent velocities.
        :param mode: str. 'remove' for denoising and 'retain' for decomposition.
        :param scale_begin: int. The beginning scale to do coherent denoising.
        :param nbscales: int. Number of scales including the coarsest wavelet
            level. Default set to ceil(log2(min(M,N)) - 3).
        :param nbangles: int. Number of angles at the 2nd coarsest level,
            minimum 8, must be a multiple of 4.
        """
        self.data = curvelet_denoising(self.data, dx=self.dx, fs=self.fs,
                                       **kwargs)
        return self

    def fk_filter(self, mode='retain', verbose=False, **kwargs):
        """
        Transform the data to the f-k domain using 2-D Fourier transform method,
        and transform back to the x-t domain after filtering.

        :param mode: str. 'remove' for denoising, 'retain' for extraction, and
            'decompose' for decomposition and not update self.data.
        :param verbose: If True, return filtered data, f-k spectrum,
            frequency sequence, wavenumber sequence and f-k mask.
        :param taper: float or sequence of floats. Each float means decimal
            percentage of Tukey taper for corresponding dimension (ranging from
            0 to 1). Default is 0.1 which tapers 5% from the beginning and 5%
            from the end.
        :param pad: Pad the data or not. It can be float or sequence of floats.
            Each float means padding percentage before FFT for corresponding
            dimension. If set to 0.1 will pad 5% before the beginning and after
            the end. 'default' means pad both dimensions to next power of 2.
            None or False means don't pad data before or during Fast Fourier
            Transform.
        :param fmin, fmax, kmin, kmax, vmin, vmax: float or or sequence of 2
            floats. Sequence of 2 floats represents the start and end of taper.
        :param edge: float. The width of fan mask taper edge.
        :param flag: -1 keep only negative apparent velocities, 0 keep both
            postive and negative apparent velocities, 1 keep only positive
            apparent velocities.
        """
        output = fk_filter(self.data, self.dx, self.fs, mode=mode,
                           verbose=verbose, **kwargs)
        if mode == 'decompose':
            sec1 = self.copy()
            sec2 = self.copy()
            sec1.data, sec2.data = output[:2]
            if verbose:
                return sec1, sec2, *output[2:]
            else:
                return sec1, sec2
        elif verbose:
            self.data = output[0]
            return output
        else:
            self.data = output
            return self

    def curvelet_windowing(self, mode='retain', **kwargs):
        """
        Use curevelet transform to keep cooherent signal with certain velocity
        range.

        :param mode: str. 'remove' for denoising, 'retain' for extraction, and
            'decompose' for decomposition and not update self.data.
        :param vmin, vmax: float. Velocity range in m/s.
        :param flag: -1 keep only negative apparent velocities, 0 keep both
            postive and negative apparent velocities, 1 keep only positive
            apparent velocities.
        :param pad: float or sequence of floats. Each float means padding
            percentage before FFT for corresponding dimension. If set to 0.1
            will pad 5% before the beginning and after the end.
        :param scale_begin: int. The beginning scale to do coherent denoising.
        :param nbscales: int. Number of scales including the coarsest wavelet
            level. Default set to ceil(log2(min(M,N)) - 3).
        :param nbangles: int. Number of angles at the 2nd coarsest level,
            minimum 8, must be a multiple of 4.
        """
        output = curvelet_windowing(self.data, self.dx, self.fs, mode=mode,
                                    **kwargs)
        if mode == 'decompose':
            sec1 = self.copy()
            sec2 = self.copy()
            sec1.data, sec2.data = output
            return sec1, sec2
        else:
            self.data = output
            return self

    def _strain2vel_attr(self):
        if hasattr(self, 'data_type'):
            if 'strain rate' in self.data_type:
                self.data_type = 'acceleration'
            elif 'strain' in self.data_type:
                self.data_type = 'velocity'
            else:
                warnings.warn(f'The data type is {self.data_type}, neither '
                              'strain nor strain rate. But it still takes '
                              'effect.')
        else:
            self.data_type = 'velocity or acceleration'
        return self

    def fk_rescaling(self, turning=None, verbose=False, **kwargs):
        """
        Convert strain / strain rate to velocity / acceleration by fk rescaling.

        :param turning: Sequence of int. Channel number of turning points. If
            self.turning exists, it will be used by default unless the parameter
            turning is set to False.
        :param verbose: If True and turning is not set, return f-k spectrum,
            frequency sequence, wavenumber sequence and f-k mask.
        :param taper: float or sequence of floats. Each float means decimal
            percentage of Tukey taper for corresponding dimension (ranging from
            0 to 1). Default is 0.1 which tapers 5% from the beginning and 5%
            from the end. If the turning parameter is set, this parameter will
            be invalid.
        :param pad: Pad the data or not. It can be float or sequence of floats.
            Each float means padding percentage before FFT for corresponding
            dimension. If set to 0.1 will pad 5% before the beginning and after
            the end. 'default' means pad both dimensions to next power of 2.
            None or False means don't pad data before or during Fast Fourier
            Transform.
        :param fmax, kmin, vmax: float or or sequence of 2 floats. Sequence of 2
            floats represents the start and end of taper. Setting these
            parameters can reduce artifacts.
        :param edge: float. The width of fan mask taper edge.
        """
        if hasattr(self, 'turning_channels') and turning is None:
            turning = np.array(self.turning_channels) - self.start_channel

        self._strain2vel_attr()
        if verbose and not turning:
            data_res, fk, f, k, mask = fk_rescaling(self.data, self.dx, self.fs,
                                                    verbose=True, **kwargs)
            self.data = data_res
            return fk, f, k, mask
        else:
            self.data = fk_rescaling(self.data, self.dx, self.fs, **kwargs)
            return self

    def curvelet_conversion(self, turning=None, **kwargs):
        """
        Use curevelet transform to convert strain/strain rate to
        velocity/acceleration.

        :param turning: Sequence of int. Channel number of turning points. If
            self.turning exists, it will be used by default unless the parameter
            turning is set to False.
        :param pad: float or sequence of floats. Each float means padding
            percentage before FFT for corresponding dimension. If set to 0.1
            will pad 5% before the beginning and after the end.
        :param scale_begin: int. The beginning scale to do conversion.
        :param nbscales: int. Number of scales including the coarsest wavelet
            level. Default set to ceil(log2(min(M,N)) - 3).
        :param nbangles: int. Number of angles at the 2nd coarsest level,
            minimum 8, must be a multiple of 4.
        """
        if hasattr(self, 'turning_channels') and turning is None:
            turning = np.array(self.turning_channels) - self.start_channel

        self.data = curvelet_conversion(self.data, self.dx, self.fs,
                                        turning=turning, **kwargs)
        self._strain2vel_attr()
        return self

    def slant_stacking(self, channel='all', turning=None, **kwargs):
        """
        Convert strain to velocity based on slant-stack.

        :param channel: int or list or 'all'. convert a certain channel number /
            certain channel range / all channels.
        :param turning: Sequence of int. Channel number of turning points. If
            self.turning exists, it will be used by default unless the parameter
            turning is set to False.
        :param L: int. the number of adjacent channels over which slowness is
            estimated.
        :param slm: float. Slowness x max
        :param sls: float. slowness step
        :param freqmin: Pass band low corner frequency.
        :param freqmax: Pass band high corner frequency.
        """
        if hasattr(self, 'turning_channels') and turning is None:
            turning = np.array(self.turning_channels) - self.start_channel

        if isinstance(channel, int):
            channel = [channel - self.start_channel]
        elif isinstance(channel, Iterable):
            channel = np.array(channel) - self.start_channel
        elif isinstance(channel, str) and channel == 'all':
            channel = list(range(self.nch))

        self.start_channel += channel[0]
        self.start_distance += channel[0] * self.dx
        self.data = slant_stacking(self.data, self.dx, self.fs, channel=channel,
                                   turning=turning, **kwargs)
        self._strain2vel_attr()
        return self