# Purpose: Module for reading DAS data.
# Author: Minzhe Hu, Ji Zhang
# Date: 2025.11.19
# Email: hmz2018@mail.ustc.edu.cn
# Partially modified from
# https://github.com/RobbinLuo/das-toolkit/blob/main/DasTools/DasPrep.py
import warnings
import inspect
import json
import pickle
import numpy as np
import h5py
import segyio
from functools import wraps
from typing import Union
from pathlib import Path
from nptdms import TdmsFile
from daspy.core.util import _device_standardized_name, _h5_file_format, \
    _trimming_slice_metadata
from daspy.core.section import Section
from daspy.core.dasdatetime import DASDateTime, utc


def read(fname=None, output_type='section', ftype=None, file_format='auto',
         headonly=False, dtype=None, **kwargs) -> Union[Section, tuple]:
    """
    Read a .pkl/.pickle, .h5/.hdf5, .tdms, or .segy/.sgy file.

    :param fname: Path of DAS data file.
    :type fname: str or pathlib.PosixPath
    :param output_type: Output type, 'Section' for daspy.Section instance,
        'array' for numpy array and metadata dict.
    :type output_type: str
    :param ftype: File type or function for reading data.
    :type ftype: None, str or function
    :param file_format: Format in which the file is saved. Function is allowed
        to extract dataset and metadata.
    :type file_format: str or function
    :param headonly: If True, only metadata will be read.
    :type headonly: bool
    :param dtype: Data type of the returned data.
    :type dtype: str
    :param chmin: Minimum channel number.
    :type chmin: int
    :param chmax: Maximum channel number.
    :type chmax: int
    :param dch: Channel step.
    :type dch: int
    :param xmin: Minimum distance.
    :type xmin: float
    :param xmax: Maximum distance.
    :type xmax: float
    :param tmin: Minimum time.
    :type tmin: float or DASDateTime
    :param tmax: Maximum time.
    :type tmax: float or DASDateTime
    :param spmin: Minimum sampling point.
    :type spmin: int
    :param spmax: Maximum sampling point.
    :type spmax: int
    :return: daspy.Section instance or tuple of numpy array and metadata dict.
    :rtype: Section or tuple
    """
    fun_map = {'pkl': _read_pkl, 'h5': _read_h5, 'tdms': _read_tdms,
               'sgy': _read_segy, 'npy': _read_npy}
    if fname is None:
        fname = Path(__file__).parent / 'example.pkl'
        ftype = 'pkl'
    elif ftype is None:
        ftype = str(fname).split('.')[-1].lower()

    if 'ch1' in kwargs.keys() or 'ch2' in kwargs.keys():
        warnings.warn("In future versions, parameter 'ch1' and 'ch2' will be "
                      "replaced by 'chmin' and 'chmax'", FutureWarning)
        kwargs['chmin'] = kwargs.pop('ch1', None)
        kwargs['chmax'] = kwargs.pop('ch2', None)
    if callable(ftype):
        ftype = with_trimming(ftype)
        data, metadata = ftype(fname, headonly=headonly, **kwargs)
    else:
        for rtp in [('pickle', 'pkl'), ('hdf5', 'h5'), ('segy', 'sgy')]:
            ftype = ftype.replace(*rtp)
        data, metadata = fun_map[ftype](fname, headonly=headonly,
                                        file_format=file_format, **kwargs)

    if dtype is not None:
        data = data.astype(dtype)
    if output_type.lower() == 'section':
        metadata['source'] = Path(fname)
        metadata['source_type'] = ftype
        data[np.isnan(data)] = 0
        return Section(data, **metadata)
    elif output_type.lower() == 'array':
        return data, metadata


def with_trimming(func):
    """
    Decorator that wraps a custom reader so it automatically supports
    trimming parameters (chmin, chmax, dch, xmin, xmax, tmin, tmax, spmin, spmax).
    """

    @wraps(func)
    def wrapper(fname, headonly=False, **kwargs):
        # trimming-related parameters
        trim_keys = ['chmin', 'chmax', 'dch', 'xmin', 'xmax', 'tmin', 'tmax',
                     'spmin', 'spmax']
        sig = inspect.signature(func)
        reader_params = set(sig.parameters.keys())
        trim_for_reader = {k: kwargs.pop(k) for k in trim_keys if k in kwargs \
                           and k in reader_params}
        trim_for_trimming = {k: kwargs.pop(k) for k in trim_keys if k in \
                             kwargs and k not in reader_params}
        try:
            data, metadata = func(fname, headonly=headonly, **trim_for_reader,
                                  **kwargs)
        except TypeError:
            headonly = False
            data, metadata = func(fname, **trim_for_reader)

        shape = data.shape
        si, sj, metadata = _trimming_slice_metadata(shape, metadata=metadata,
                                                    **trim_for_trimming)
        if headonly:
            data = np.zeros(shape)[si, sj]
        else:
            data = data[si, sj]

        return data, metadata

    return wrapper


class DummyObject:
    def __init__(self, *args, **kwargs):
        pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            print(f"Skip missing module: {module}.{name}")
            return DummyObject
        except AttributeError:
            print(f"Skip missing class: {module}.{name}")
            return DummyObject


def _read_pkl(fname, headonly=False, file_format='auto', chmin=None, chmax=None,
              dch=1, xmin=None, xmax=None, tmin=None, tmax=None, spmin=None,
              spmax=None):
    """
    Read data and metadata from a pickle file.
    """
    with open(fname, 'rb') as f:
        # pkl_data = pickle.load(f)
        pkl_data = SafeUnpickler(f).load()
        if isinstance(pkl_data, np.ndarray):
            warnings.warn('This data format doesn\'t include channel interval'
                          'and sampling rate. Please set manually')
            si, sj, metadata = _trimming_slice_metadata(pkl_data.shape,
                chmin=chmin, chmax=chmax, dch=dch, xmin=xmin, xmax=xmax,
                tmin=tmin, tmax=tmax, spmin=spmin, spmax=spmax)
            if headonly:
                data = np.zeros_like(pkl_data[si, sj])
            else:
                data = pkl_data[si, sj]
            return data, metadata
        elif isinstance(pkl_data, dict):
            data = pkl_data.pop('data')
            si, sj, metadata = _trimming_slice_metadata(data.shape,
                metadata=pkl_data, chmin=chmin, chmax=chmax, dch=dch,
                xmin=xmin, xmax=xmax, tmin=tmin, tmax=tmax, spmin=spmin,
                spmax=spmax)
            if headonly:
                data = np.zeros_like(data[si, sj])
            else:
                data = data[si, sj]
            return data, metadata
        else:
            raise TypeError('Unknown data type.')


def _read_h5_headers(group):
    """
    Recursively read HDF5 group attributes and headers.
    """ 
    headers = {}
    if len(group.attrs) != 0:
        headers['attrs'] = dict(group.attrs)
    if isinstance(group, h5py._hl.dataset.Dataset):
        return headers
    for key, value in group.items():
        try:
            gp_headers = _read_h5_headers(value)
        except AttributeError:
            headers[key] = value
        if len(gp_headers):
            headers[key] = gp_headers

    return headers


def _read_h5(fname, headonly=False, file_format='auto', chmin=None, chmax=None,
             dch=1, xmin=None, xmax=None, tmin=None, tmax=None, spmin=None,
             spmax=None):
    """
    Read data and metadata from an HDF5 file.
    """
    with h5py.File(fname, 'r') as h5_file:
        keys = h5_file.keys()
        group = list(keys)[0]
        if file_format == 'auto':
            file_format = _h5_file_format(h5_file)
        elif isinstance(file_format, str):
            file_format = _device_standardized_name(file_format)
        transpose = False
        if callable(file_format):
            dataset, transpose, metadata = file_format(h5_file)
        elif file_format == 'AP Sensing':
            dataset = h5_file['strain']
            transpose = True
            metadata = {'dx': h5_file['spatialsampling'][()],
                        'fs': h5_file['RepetitionFrequency'][()],
                        'gauge_length': h5_file['GaugeLength'][()]}

        elif file_format == 'Aragón Photonics HDAS':
            for key in keys:
                if 'data' in key.lower():
                    data_key = key
                elif 'header' in key.lower():
                    header_key = key
            dataset = h5_file[data_key]
            header = h5_file[header_key][()]
            if header.ndim == 2:
                header = header[0]
            if 'start_time' in h5_file[data_key].attrs.keys():
                start_time = DASDateTime.fromisoformat(h5_file[data_key].
                                                       attrs['start_time'])
            else:
                start_time = DASDateTime.fromtimestamp(float(header[100])).\
                    utc()
                transpose = True
            metadata = {'dx': header[1],
                        'fs': header[6] / header[15] / header[98],
                        'start_distance': header[11],
                        'start_time': start_time}

        elif file_format == 'ASN OptoDAS': # https://github.com/ASN-Norway/simpleDAS
            dataset = h5_file['data']
            metadata = {'dx': h5_file['header/dx'][()],
                        'fs': 1 / h5_file['header/dt'][()],
                        'start_time': DASDateTime.fromtimestamp(
                            h5_file['header/time'][()]).utc(),
                        'scale': h5_file['header/dataScale'][()]}
            if h5_file['header/gaugeLength'][()] != np.nan:
                metadata['gauge_length'] = h5_file['header/gaugeLength'][()]
            if h5_file['header/dimensionNames'][0] == b'time':
                transpose = True

        elif file_format in ['Febus A1-R', 'Febus A1']:
            acquisition = list(h5_file[f'{group}/Source1/Zone1'].keys())[0]
            dataset = h5_file[f'{group}/Source1/Zone1/{acquisition}']
            transpose = True
            attrs = h5_file[f'{group}/Source1/Zone1'].attrs
            try:
                fs = float(attrs['FreqRes'])
            except KeyError:
                try:
                    fs = float(1000 / attrs['Spacing'][1])
                except KeyError:
                    fs = attrs['PulseRateFreq'][0]
            time = h5_file[f'{group}/Source1/time']
            if len(time.shape) == 2: # Febus A1-R
                start_time = DASDateTime.fromtimestamp(time[0, 0]).utc()
            elif len(time.shape) == 1: # Febus A1
                start_time = DASDateTime.fromtimestamp(time[0]).utc()
            metadata = {'dx': attrs['Spacing'][0], 'fs': fs,
                        'start_channel': int(attrs['Extent'][0]),
                        'start_distance': attrs['Origin'][0],
                        'start_time': start_time,
                        'gauge_length': attrs['GaugeLength'][0]}

        elif file_format == 'OptaSense ODH3':
            dataset = h5_file['data']
            dx = (h5_file['x_axis'][-1] - h5_file['x_axis'][0]) / \
                (len(h5_file['x_axis']) - 1)
            fs = (len(h5_file['t_axis']) - 1) / (h5_file['t_axis'][-1] -
                                                 h5_file['t_axis'][0])
            metadata = {'dx': dx, 'fs': fs, 'start_time': h5_file['t_axis'][0]}

        elif file_format == 'OptaSense ODH4':
            dataset = h5_file['raw_data']
            attrs = h5_file.attrs
            dx = attrs['channel spacing m']
            start_channel = attrs['channel_start']
            metadata = {'dx': dx, 'fs': attrs['sampling rate Hz'],
                        'start_channel': start_channel,
                        'start_distance': start_channel * dx,
                        'start_time': DASDateTime.fromisoformat(
                            attrs['starttime']) ,
                        'data_type': attrs['raw_data_units'],
                        'scale': attrs['scale factor to strain']}

        elif file_format in ['OptaSense ODH4+', 'OptaSense QuantX',
                             'Silixa iDAS-MG', 'Sintela Onyx v1.0',
                             'Smart Earth ZD-DAS', 'Unknown']:
            dataset = h5_file['Acquisition/Raw[0]/RawData/']
            attrs = h5_file['Acquisition'].attrs
            if 'NumberOfLoci' in attrs.keys():
                if dataset.shape[0] != attrs['NumberOfLoci']:
                    transpose = True

            try:
                fs = h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate']
            except KeyError:
                time_arr = h5_file['Acquisition/Raw[0]/RawDataTime/']
                fs = 1 / (np.diff(time_arr).mean() / 1e6)

            try:
                stime = dataset.attrs['PartStartTime']
            except KeyError:
                try:
                    stime = attrs['MeasurementStartTime']
                except KeyError:
                    try:
                        stime = h5_file['Acquisition/Raw[0]/RawDataTime/'][0]
                    except KeyError:
                        stime = 0

            if isinstance(stime, bytes):
                stime = stime.decode('ascii')

            if isinstance(stime, str):
                stime = DASDateTime.fromisoformat(stime)
            else:
                stime = DASDateTime.fromtimestamp(stime / 1e6).astimezone(utc)

            metadata = {'dx': attrs['SpatialSamplingInterval'], 'fs': fs,
                        'start_time': stime,
                        'gauge_length': attrs['GaugeLength']}

        elif file_format == 'Puniu Tech HiFi-DAS':
            dataset = h5_file['default']
            if 'time,channel' in attrs.get('row_major_order', 'time, channel')\
                .replace(' ', '').lower():
                transpose = True

            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v
                     in dataset.attrs.items()}
            step = int(attrs['step'])
            dx = step * attrs.get('spatial_sampling_rate', 1.0)
            start_channel = int(attrs['start_channel'])
            if step != 1:
                if chmin:
                    chmin = (chmin - start_channel) / step + start_channel
                if chmax:
                    chmax = (chmin - start_channel) / step + start_channel
            t0 = int(attrs.get('epoch', 0)) + int(attrs.get('ns', 0)) * 1e-9
            data_type = 'strain rate' if attrs.get('format', '') == \
                'differential' else 'strain',
            metadata = {'dx': dx, 'fs': float(attrs['sampling_rate']),
                        'start_channel': start_channel,
                        'start_distance': start_channel * dx,
                        'start_time': DASDateTime.fromtimestamp(t0, tz=utc),
                        'scale': 110.37, 'data_type': data_type,
                        'cid': attrs.get('cid', '')}
            if 'spatial_resolution' in attrs.keys():
                metadata['gauge_length'] = float(attrs['spatial_resolution'])

        elif file_format == 'Silixa iDAS':
            dataset = h5_file['Acquisition/Raw[0]/RawData/']
            attrs = h5_file['Acquisition/Raw[0]'].attrs
            if dataset.shape[0] != attrs['NumberOfLoci']:
                transpose = True

            dx = np.mean(h5_file['Mapping/MeasuredSpatialResolution'])
            start_distance = h5_file['Acquisition/Custom/UserSettings'].\
                    attrs['StartDistance']
            start_time = DASDateTime.fromisoformat(
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'].
                decode('ascii'))

            metadata = {'dx': dx, 'fs': attrs['OutputDataRate'],
                        'start_distance': start_distance,
                        'start_time':start_time,
                        'gauge_length': h5_file['Acquisition'].
                            attrs['GaugeLength'],
                        'geometry':  np.vstack((h5_file['Mapping/Lon'],
                                                h5_file['Mapping/Lat'])).T,
                        'scale': attrs['AmpScaling']}

        elif file_format == 'T8 Sensor':
            ds_name = list(h5_file['/ProcessedData'].keys())[0]
            dpath = f'/ProcessedData/{ds_name}'
            dataset = h5_file[dpath]
            transpose = True
            attrs = h5_file[dpath].attrs
            dx = float(attrs['record/channel_spacing_m']) * float(
                attrs['downsampling/decimation_in_length'])
            fs = float(attrs['record/pulse_repetition_rate_hz']) / float(
                attrs['downsampling/decimation_in_time'])
            metadata = {'dx': dx, 'fs':fs,
                        'start_distance': float(attrs['record/line_offset_m']),
                        'start_time': DASDateTime.strptime(
                            attrs['record/start_time'], '%Y%m%dT%H%M%S%f'),
                        'gauge_length': float(attrs['record/gauge_length_m'])}
            if 'event/time' in attrs.keys():
                metadata['origin_time'] = DASDateTime.strptime(
                    attrs['event/time'], '%Y%m%dT%H%M%S%f')

        elif file_format == 'INGV':
            dataset = h5_file['Fiber']
            transpose = True
            # dx = h5_file.attrs['MeasureLength[m]'][0] / \
            #     len(h5_file['ChannelMap'])
            dx = (h5_file.attrs['Stop Distance (m)'][0] -
                  h5_file.attrs['Start Distance (m)'][0]) / \
                    len(h5_file['ChannelMap'])
            start_channel = int(np.argmax(h5_file['ChannelMap'][()] >= 0))
            gauge_length = h5_file.attrs['GaugeLength'][0]
            scale = 116e-9 * h5_file.attrs['SamplingFrequency[Hz]'] / \
                gauge_length / 8192 / h5_file.attrs['FilterGain']
            metadata = {'dx': dx, 'fs': h5_file.attrs['Samplerate'][0],
                        'start_channel': start_channel,
                        'start_distance': h5_file.attrs['Start Distance (m)']
                            [0] + start_channel * dx,
                        'start_time': DASDateTime.fromtimestamp(h5_file.attrs
                            ['StartTime'][0] / 1e6).utc(),
                        'gauge_length': gauge_length, 'scale': scale}

        elif file_format in 'JAMSTEC':
            dataset = h5_file['DAS_record']
            metadata = {'dx': h5_file['Sampling_interval_in_space'][0],
                        'fs': 1 / h5_file['Sampling_interval_in_time'][0]}

        elif file_format == 'NEC':
            dataset = h5_file['data']
            dx = dataset.attrs['Interval of monitor point']
            fs = 1.0 / (dataset.attrs['Interval time of data'] / 1000.0) # Hz
            if dataset.shape[0] != \
                dataset.attrs['Number of requested location points']:
                transpose = True
            try:
                scale = dataset.attrs['Radians peer digital value']
            except KeyError:
                try:
                    scale = dataset.attrs['Radians per digital value']
                except KeyError:
                    scale = 1
            # start_time = datetime(1970, 1, 1) + \
            # timedelta(milliseconds=start_unix_epoch_in_ms)
            start_time = DASDateTime.fromtimestamp(
                np.float64(dataset.attrs['Time of sending request']) / 1e3
                ).utc()
            metadata = {'fs': fs, 'dx': dx, 'start_time': start_time,
                        'gauge_length': dataset.attrs['Gauge length'],
                        'scale': scale, 'data_type':'strain rate'}

        elif file_format == 'FORESEE':
            dataset = h5_file['raw']
            fs = round(1 / np.diff(h5_file['timestamp']).mean())
            start_time = DASDateTime.fromtimestamp(
                h5_file['timestamp'][0]).astimezone(utc)
            warnings.warn('This data format doesn\'t include channel interval.'
                          ' Please set manually')
            metadata = {'dx': None, 'fs': fs, 'start_time': start_time}

        elif file_format == 'AI4EPS': # https://ai4eps.github.io/homepage/ml4earth/seismic_event_format_das/
            dataset = h5_file['data']
            attr = h5_file['data'].attrs
            dx = attr['dx_m']
            metadata = {'dx': dx, 'fs': 1 / attr['dt_s'],
                        'start_time': DASDateTime.fromisoformat(
                            attr['begin_time']),
                        'data_type': attr['unit']}
            if 'event_time' in attr.keys():
                metadata['origin_time'] = DASDateTime.fromisoformat(
                    attr['event_time'])

        elif file_format == 'Unknown0':
            dataset = h5_file['data_product/data']
            nch = h5_file.attrs['nx']
            if h5_file['data_product/data'].shape[0] != nch:
                transpose = True

            if h5_file.attrs['saving_start_gps_time'] > 0:
                start_time = DASDateTime.fromtimestamp(
                    h5_file.attrs['file_start_gps_time'])
            else:
                start_time = DASDateTime.fromtimestamp(
                    h5_file.attrs['file_start_computer_time'])
            metadata = {'dx': h5_file.attrs['dx'],
                        'fs': 1 / h5_file.attrs['dt_computer'],
                        'start_time': start_time.astimezone(utc),
                        'gauge_length': h5_file.attrs['gauge_length'],
                        'data_type': h5_file.attrs['data_product']}

        metadata['file_format'] = file_format
        metadata['headers'] = _read_h5_headers(h5_file)
        shape = dataset.shape
        if len(shape) == 3:
            if headonly:
                fs = int(metadata['fs'])
                fs_b = attrs.get('BlockRate', [1000])[0] / 1e3
                nsp_b = round(fs/fs_b)
                shape = (shape[0] * nsp_b, shape[2])
            else:
                shape = (shape[0] * shape[1], shape[2])
        if transpose:
            shape = shape[::-1]
        si, sj, metadata = _trimming_slice_metadata(shape, metadata=metadata,
            chmin=chmin, chmax=chmax, dch=dch, xmin=xmin, xmax=xmax, tmin=tmin,
            tmax=tmax, spmin=spmin, spmax=spmax)
        if headonly:
            data = np.zeros(shape, dtype=dataset.dtype)[si, sj]
        elif transpose:
            if len(dataset.shape) == 3:
                fs = int(metadata['fs'])
                fs_b = attrs.get('BlockRate', [1000])[0] / 1e3
                nsp_b = round(fs/fs_b)
                half_ol = round((dataset.shape[1] - nsp_b) / 2)
                j0, k0 = divmod(sj.start, nsp_b)
                j1, k1 = divmod(sj.stop, nsp_b)
                k1 = (j1 - j0) * nsp_b + k1
                j1 += 1
                data = dataset[j0:j1, half_ol:half_ol+nsp_b, si]
                data = data.reshape((-1, data.shape[-1]))[k0:k1, :].T
            else:
                data = dataset[sj, si].T
        else:
            data = dataset[si, sj]
    return data, metadata


def _read_tdms(fname, headonly=False, file_format='auto', chmin=None,
               chmax=None, dch=1, xmin=None, xmax=None, tmin=None, tmax=None,
               spmin=None, spmax=None):
    """
    Read data and metadata from a TDMS file. see
    https://nptdms.readthedocs.io/en/stable/quickstart.html.
    """
    with TdmsFile.read(fname) as tdms_file:
        if file_format == 'auto':
            group_name = [group.name for group in tdms_file.groups()]
            if group_name == ['Measurement']:
                key = 'Measurement'
                properties = tdms_file.properties
                version = float(properties['iDASVersion'][:3])
                if version < 2.3:
                    file_format = 'Silixa iDAS'
                elif 2.3 <= version < 2.7:
                    file_format = 'Silixa iDAS-v2'
                elif version >= 2.7:
                    file_format = 'Silixa iDAS-v3'
            elif group_name == ['DAS']:
                key = 'DAS'
                properties = tdms_file[key].properties
                file_format = 'Institute of Semiconductors, CAS'
            else:
                key = group_name[0]
                file_format = 'Unknown'
        else:
            file_format = _device_standardized_name(file_format)

        if file_format in ['Silixa iDAS', 'Silixa iDAS-v2', 'Silixa iDAS-v3',
                           'Unknown']:
            start_channel = min([int(channel.name) for channel in
                                 tdms_file[key].channels()])
            shape = (len(tdms_file[key]),
                     len(tdms_file[key][str(start_channel)]))
            metadata = {'dx': properties['SpatialResolution[m]'],
                        'fs': properties['SamplingFrequency[Hz]'],
                        'start_channel': start_channel,
                        'headers': {**properties}}
            try:
                metadata['start_distance'] = properties['Start Distance (m)']
            except KeyError:
                metadata['start_distance'] = properties['StartPosition[m]']
            
            try:
                metadata['start_time'] = DASDateTime.fromisoformat(
                    properties['ISO8601 Timestamp'])
            except KeyError:
                start_time = 0
                for time_key in ['GPSTimeStamp', 'CPUTimeStamp', 'Trigger Time']:
                    if time_key in properties.keys():
                        if isinstance(properties[time_key], str):
                            start_time = DASDateTime.fromisoformat(
                                properties[time_key])
                            break
                        elif isinstance(properties[time_key], np.datetime64):
                            start_time = DASDateTime.from_datetime(
                                properties[time_key].item())
                            break
                metadata['start_time'] = start_time
            if 'GaugeLength' in properties.keys():
                metadata['gauge_length'] = properties['GaugeLength']

            si, sj, metadata = _trimming_slice_metadata(shape,
                metadata=metadata, chmin=chmin, chmax=chmax, dch=dch, xmin=xmin,
                xmax=xmax, tmin=tmin, tmax=tmax, spmin=spmin, spmax=spmax)
            if headonly:
                data = np.zeros(shape,
                    dtype=tdms_file[key][str(start_channel)].dtype)[si, sj]
            else:
                data = np.asarray([tdms_file[key][str(ch)][sj] for ch in
                                   range(si.start, si.stop, si.step)])
        elif file_format == 'Institute of Semiconductors, CAS':
            try:
                start_channel = int(properties['Initial Channel'])
            except KeyError:
                start_channel = 0
            nch = int(properties['Total Channels'])
            nsp = len(tdms_file[key].channels()[0]) // nch
            metadata = {'dx': properties['Spatial Resolution'],
                        'fs': 1 / properties['Time Base'],
                        'start_channel': start_channel,
                        'start_time': DASDateTime.from_datetime(
                            properties['Trigger Time'].item()),
                        'headers': {**tdms_file.properties, **properties}}

            si, sj, metadata = _trimming_slice_metadata((nch, nsp),
                metadata=metadata, chmin=chmin, chmax=chmax, dch=dch, xmin=xmin,
                xmax=xmax, tmin=tmin, tmax=tmax, spmin=spmin, spmax=spmax)
            if headonly:
                data = np.zeros((nch, nsp),
                    dtype=tdms_file[key].channels()[0].dtype)[si, sj]
            else:
                data = np.asarray(tdms_file[key].channels()[0][sj.start * nch,
                    sj.stop * nch]).reshape((-1, nch)).T[si]

        metadata['file_format'] = file_format
    return data, metadata


def _read_segy(fname, headonly=False, file_format='auto', chmin=None,
               chmax=None, dch=1, xmin=None, xmax=None, tmin=None, tmax=None,
               spmin=None, spmax=None):
    """
    Read data and metadata from a SEG-Y file. See
    https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb.
    """
    with segyio.open(fname, ignore_geometry=True) as segy_file:
        if file_format == 'auto':
            file_format = 'Unknown'
        else:
            file_format = _device_standardized_name(file_format)
        
        metadata = {'fs': 1 / (segyio.tools.dt(segy_file) / 1e6),
                    'file_format': file_format}
        warnings.warn('This data format doesn\'t include channel interval.'
                    'Please set manually')
        
        nch = segy_file.tracecount
        nsp = segy_file.trace.raw.shape
        si, sj, metadata = _trimming_slice_metadata((nch, nsp),
            metadata=metadata, chmin=chmin, chmax=chmax, dch=dch, tmin=tmin,
            tmax=tmax, spmin=spmin, spmax=spmax)

        if headonly:
            data = np.zeros((nch, nsp), dtype=segy_file.trace.raw.dtype)[si, sj]
        else:
            data = segy_file.trace.raw[si][:, sj]
        return data, metadata


def _read_npy(fname, headonly=False, chmin=None, chmax=None, dch=1, xmin=None,
              xmax=None, tmin=None, tmax=None, spmin=None, spmax=None):
    """
    Read data from a NumPy .npy file.
    """
    data = np.load(fname)
    if headonly:
        return np.zeros_like(data), {'dx': None, 'fs': None}
    else:
        si, sj, metadata = _trimming_slice_metadata(data.shape, chmin=chmin,
            chmax=chmax, spmin=spmin, spmax=spmax)
        warnings.warn('This data format doesn\'t include channel interval and '
                    'sampling rate. Please set manually')
        return data[si, sj], metadata


def read_json(fname, output_type='dict'):
    """
    Read .json metadata file.

    See :cite:`Lai2024` for format details.

    :param fname: Path of json file.
    :type fname: str or pathlib.PosixPath
    :param output_type: Output type, 'dict' for dictionary, 'Section' for empty
        daspy.Section instance.
    :type output_type: str
    :return: Metadata dictionary or daspy.Section instance without data.
    :rtype: dict or Section
    """
    with open(fname, 'r') as fcc_file:
        headers = json.load(fcc_file)
    if output_type.lower() == 'dict':
        return headers
    elif output_type.lower() in ['section', 'sec']:
        if len(headers['Overview']['Interrogator']) > 1:
            case_type = 'Multiple interrogators, single cable'
            sec_num = len(headers['Overview']['Interrogator'])
            sec = []
            for interrogator in headers['Overview']['Interrogator']:
                nch = interrogator['Acquisition'][0]['Attributes']\
                    ['number_of_channels']
                data = np.zeros((nch, 0))
                dx = interrogator['Acquisition'][0]['Attributes']\
                    ['spatial_sampling_interval']
                fs = interrogator['Acquisition'][0]['Attributes']\
                    ['acquisition_sample_rate']
                gauge_length = interrogator['Acquisition'][0]['Attributes']\
                    ['gauge_length']
                sec.append(Section(data, dx, fs, gauge_length=gauge_length,
                                   headers=headers))
        elif len(headers['Overview']['Interrogator'][0]['Acquisition']) > 1:
            case_type = 'Active survey'
            sec_num = len(
                headers['Overview']['Interrogator'][0]['Acquisition'])
            sec = []
            for acquisition in headers['Overview']['Interrogator'][0]\
                ['Acquisition']:
                nch = acquisition['Attributes']['number_of_channels']
                data = np.zeros((nch, 0))
                dx = acquisition['Attributes']['spatial_sampling_interval']
                fs = acquisition['Attributes']['acquisition_sample_rate']
                gauge_length = acquisition['Attributes']['gauge_length']
                sec.append(Section(data, dx, fs, gauge_length=gauge_length,
                                   headers=headers))
        else:
            sec_num = 1
            if len(headers['Overview']['Cable']) > 1:
                case_type = 'Single interrogators, multiple cable'
            else:
                env = headers['Overview']['Cable'][0]['Attributes']\
                    ['cable_environment']
                if env == 'trench':
                    case_type = 'Direct buried'
                elif env == 'conduit':
                    case_type = 'Dark fiber'
                elif env in ['wireline', 'outside borehole casing']:
                    case_type = 'Borehole cable'
            nch = headers['Overview']['Interrogator'][0]['Acquisition'][0]\
                ['Attributes']['number_of_channels']
            dx = headers['Overview']['Interrogator'][0]['Acquisition'][0]\
                ['Attributes']['spatial_sampling_interval']
            fs = headers['Overview']['Interrogator'][0]['Acquisition'][0]\
                ['Attributes']['acquisition_sample_rate']
            gauge_length = headers['Overview']['Interrogator'][0]\
                ['Acquisition'][0]['Attributes']['gauge_length']
            data = np.zeros((nch, 0))
            sec = Section(data, dx, fs, gauge_length=gauge_length,
                          headers=headers)

        print(f'For case of {case_type}, create {sec_num} empty daspy.Section '
              'instance(s)')
        return sec
