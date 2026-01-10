# Purpose: Module for writing DAS data.
# Author: Minzhe Hu
# Date: 2025.11.19
# Email: hmz2018@mail.ustc.edu.cn
import os
import warnings
import pickle
import numpy as np
import h5py
import segyio
from shutil import copyfile
from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject
from datetime import datetime
from daspy.core.util import _device_standardized_name, _h5_file_format
from daspy.core.dasdatetime import DASDateTime


def write(sec, fname, ftype=None, raw_fname=None, dtype=None,
          file_format='auto'):
    fun_map = {'tdms': _write_tdms, 'h5': _write_h5, 'sgy': _write_segy}
    if ftype is None:
        ftype = str(fname).lower().split('.')[-1]
    for rtp in [('pickle', 'pkl'), ('hdf5', 'h5'), ('segy', 'sgy')]:
        ftype = ftype.replace(*rtp)
    if dtype is not None:
        sec = sec.copy()
        sec.data = sec.data.astype(dtype)
    if ftype == 'npy':
        np.save(fname, sec.data)
    elif ftype == 'pkl':
        write_pkl(sec, fname)
    else:
        fun_map[ftype](sec, fname, raw_fname=raw_fname, file_format=file_format)
    return None


def write_pkl(sec, fname):
    save_dict = sec.__dict__
    save_dict.pop('source', None)
    with open(fname, 'wb') as f:
        pickle.dump(save_dict, f)
    return None


def _update_h5_dataset(h5_file, path, name, data):
    attrs = h5_file[path + name].attrs
    del h5_file[path + name]
    h5_file.get(path).create_dataset(name, data=data)
    for key, value in attrs.items():
        h5_file[path + name].attrs[key] = value
    return None


def _write_h5(sec, fname, raw_fname=None, file_format='auto'):
    """
    Write data to HDF5 file with specified format.
    
    :param sec: DAS data section.
    :type sec: daspy.Section  
    :param fname: Output file path.
    :type fname: str
    :param raw_fname: Template file path to copy from.
    :type raw_fname: str
    :param file_format: Target file format.
    :type file_format: str
    """
    if raw_fname is None: # Create new file with specified format
        if file_format == 'auto':
            file_format = 'OptaSense QuantX'
        else:
            file_format = _device_standardized_name(file_format)
        with h5py.File(fname, 'w') as h5_file:
            if file_format == 'AP Sensing':
                h5_file.create_dataset('strain', data=sec.data.T)
                h5_file.create_dataset('spatialsampling', data=sec.dx)
                h5_file.create_dataset('RepetitionFrequency', data=sec.fs)
                if hasattr(sec, 'gauge_length'):
                    h5_file.create_dataset('GaugeLength', data=sec.gauge_length)
                h5_file.create_dataset('Fiberlength', data=sec.nch * sec.dx)

            elif file_format == 'Aragón Photonics HDAS':
                h5_file.create_dataset('data', data=sec.data)
                header = np.zeros(200)
                header[1] = sec.dx
                header[6] = sec.fs
                header[15] = 1
                header[98] = 1
                header[11] = sec.start_distance
                if isinstance(sec.start_time, datetime):
                    h5_file['data'].attrs['start_time'] = \
                        sec.start_time.isoformat()
                    header[100] = sec.start_time.timestamp()
                else:
                    header[100] = sec.start_time
                h5_file.create_dataset('hdas_header', data=header)

            elif file_format == 'ASN OptoDAS':
                h5_file.create_dataset('data', data=sec.data)
                for key in ['cableSpec',  'header', 'instrumentOptions',
                            'monitoring', 'processingChain', 'timing',
                            'versions']:
                    h5_file.create_group(key)
                for key in ['fileGenerator', 'fileVersion',]:
                    h5_file.create_dataset(key, data=[])

                h5_file['header'].create_dataset('dx', data=sec.dx)
                h5_file['header'].create_dataset('dt', data=1/sec.fs)
                h5_file['header'].create_dataset('time',
                    data=sec.start_time.timestamp() if
                    isinstance(sec.start_time, datetime) else sec.start_time)
                h5_file['header'].create_dataset('gaugeLength',
                    data=sec.gauge_length if hasattr(sec, 'gauge_length')
                    else -1)
                h5_file['header'].create_dataset('dataScale',
                    data=sec.scale if hasattr(sec, 'scale') else 1)
                h5_file['header'].create_dataset('dimensionNames', 
                    data=[b'distance', b'time'])

            elif file_format in ['Febus A1-R', 'Febus A1']:
                h5_file.create_group('fa1-00000/Source1/Zone1')
                attrs = h5_file['fa1-00000/Source1/Zone1'].attrs
                attrs.create('FreqRes', np.bytes_(str(sec.fs)))
                attrs.create('PulseRateFreq', [sec.fs * 1000])
                attrs.create('SamplingRes', [1])
                attrs.create('SamplingRate', [sec.fs])
                attrs.create('FiberLength', [sec.distance])
                attrs.create('Origin', [sec.start_distance, 0, 0])
                attrs.create('Extent',
                    [sec.start_channel, sec.end_channel-1, 0, 999, 0, 0])
                attrs.create('Spacing',
                    [sec.dx, 1000.0/sec.fs, 1.])
                if hasattr(sec, 'gauge_length'):
                    attrs.create('GaugeLength', [sec.gauge_length])
                else:
                    attrs.create('GaugeLength', -1)

                if file_format == 'Febus A1':
                    seconds = np.ceil(sec.nsp / sec.fs).astype(int)
                    DataTime = sec.start_time.timestamp() + np.arange(0,
                                                                      seconds)
                    h5_file.create_dataset(
                        'fa1-00000/Source1/Zone1/acquisition', data=sec.data.T)
                else:
                    data = sec.data
                    fs = int(sec.fs)
                    seconds = sec.nsp // fs
                    mod = sec.nsp % fs
                    if mod:
                        data = np.hstack((data, np.zeros((sec.nch, fs - mod))))
                        seconds += 1

                    data = data.reshape(sec.nch, fs, seconds).T
                    h5_file.create_dataset(
                        'fa1-00000/Source1/Zone1/acquisition', data=data)
                    DataTime = (sec.start_time.timestamp() +
                                np.arange(0, seconds)).reshape((1, -1))
                h5_file.create_dataset(f'fa1-00000/Source1/time', data=DataTime)

            elif file_format == 'OptaSense ODH3':
                h5_file.create_dataset('data', data=sec.data)
                x_axis = np.arange(sec.nch) * sec.dx + sec.start_distance
                t_axis = np.arange(sec.nsp) / sec.fs
                h5_file.create_dataset('x_axis', data=x_axis)
                h5_file.create_dataset('t_axis', data=t_axis)

            elif file_format in ['OptaSense ODH4+', 'OptaSense QuantX',
                                 'Silixa iDAS-MG', 'Sintela Onyx v1.0',
                                 'Smart Earth ZD-DAS', 'Unknown']:
                h5_file.create_group('Acquisition/Raw[0]')
                start_time = sec.start_time.utc() if \
                    isinstance(sec.start_time, datetime) else \
                    DASDateTime.fromtimestamp(sec.start_time)
                if file_format in ['OptaSense ODH4+', 'OptaSense QuantX',
                                   'Unknown']:
                    h5_file.get('Acquisition/Raw[0]/').create_dataset('RawData',
                        data=sec.data)
                    h5_file['Acquisition/Raw[0]/RawData'].\
                        attrs['PartStartTime'] = np.bytes_(
                            start_time.strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
                else:
                    h5_file.get('Acquisition/Raw[0]/').create_dataset('RawData',
                        data=sec.data.T)
                    stime_str = start_time.isoformat()
                    if file_format in ['Silixa iDAS-MG', 'Smart Earth ZD-DAS']:
                        stime_str = np.bytes_(stime_str)
                    h5_file['Acquisition/Raw[0]/RawData'].\
                        attrs['PartStartTime'] = stime_str
                    h5_file['Acquisition'].attrs['MeasurementStartTime'] = \
                        stime_str
                h5_file['Acquisition'].attrs['NumberOfLoci'] = sec.nch
                stimestamp = start_time.timestamp()
                datatime = (np.arange(stimestamp, stimestamp + sec.nsp /
                    sec.fs, 1 / sec.fs) * 1e6).astype(int)
                h5_file.get('Acquisition/Raw[0]/').\
                    create_dataset('RawDataTime', data=datatime)
                h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
                h5_file['Acquisition'].attrs['SpatialSamplingInterval'] = \
                    sec.dx
                h5_file['Acquisition'].attrs['GaugeLength'] = \
                    sec.gauge_length if hasattr(sec, 'gauge_length') else -1

            elif file_format == 'Puniu Tech HiFi-DAS':
                h5_file.create_dataset('default', data=sec.data)
                h5_file['default'].attrs['row_major_order'] = 'channel, time'
                h5_file['default'].attrs['spatial_sampling_rate'] = sec.dx
                h5_file['default'].attrs['step'] = 1
                h5_file['default'].attrs['sampling_rate'] = sec.fs
                h5_file['default'].attrs['start_channel'] = sec.start_channel
                if isinstance(sec.start_time, datetime):
                    t0 = sec.start_time.timestamp()
                else:
                    t0 = sec.start_time
                epoch = int(t0)
                h5_file['default'].attrs['epoch'] = epoch
                h5_file['default'].attrs['ns'] = int(round((t0 - epoch) * 1e9))
                if hasattr(sec, 'data_type'):
                    h5_file['default'].attrs['format'] = 'differential' if \
                        sec.data_type == 'strain rate' else sec.data_type
                else:
                    h5_file['default'].attrs['format'] = 'unknown'
                
                if hasattr(sec, 'gauge_length'):
                    h5_file['default'].attrs['spatial_resolution'] = \
                        sec.gauge_length
                try:
                    h5_file['default'].attrs['cid'] = \
                        sec.headers['default']['attrs']['cid']
                except KeyError:
                    h5_file['default'].attrs['cid'] = 'unknown'

            elif file_format == 'Silixa iDAS':
                h5_file.create_group('Acquisition/Raw[0]')
                h5_file.create_group('Mapping')
                h5_file.create_group('Acquisition/Custom/UserSettings')
                
                h5_file.create_dataset('Acquisition/Raw[0]/RawData',
                                       data=sec.data)
                h5_file['Acquisition/Raw[0]'].attrs['NumberOfLoci'] = sec.nch
                h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
                h5_file['Acquisition/Raw[0]'].attrs['AmpScaling'] = sec.scale \
                    if hasattr(sec, 'scale') else 1.0
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                    np.bytes_(sec.start_time.isoformat())
                h5_file['Acquisition/Custom/UserSettings'].\
                    attrs['StartDistance'] = sec.start_distance
                if hasattr(sec, 'gauge_length'):
                    h5_file['Acquisition'].attrs['GaugeLength'] = \
                        sec.gauge_length
                else:
                    h5_file['Acquisition'].attrs['GaugeLength'] = None

                h5_file.create_dataset('Mapping/MeasuredSpatialResolution',
                                       data=np.full(sec.nch, sec.dx))

                if hasattr(sec, 'geometry'):
                    h5_file.create_dataset('Mapping/Lon',
                                           data=sec.geometry[:,0])
                    h5_file.create_dataset('Mapping/Lat',
                                           data=sec.geometry[:,1])
                else:
                    h5_file.create_dataset('Mapping/Lon',
                                           data=np.zeros(sec.nch))
                    h5_file.create_dataset('Mapping/Lat',
                                           data=np.zeros(sec.nch))

            elif file_format == 'T8 Sensor':
                h5_file.create_group('ProcessedData')
                h5_file.create_dataset('/ProcessedData/DAS', data=sec.data.T)
                
                attrs = h5_file['/ProcessedData/DAS'].attrs
                attrs['record/channel_spacing_m'] = str(sec.dx)
                attrs['record/pulse_repetition_rate_hz'] = str(sec.fs)
                attrs['downsampling/decimation_in_length'] = '1'
                attrs['downsampling/decimation_in_time'] = '1'
                attrs['record/line_offset_m'] = str(sec.start_distance)
                attrs['record/start_time'] = sec.start_time.\
                    strftime('%Y%m%dT%H%M%S%f')
                attrs['record/gauge_length_m'] = str(sec.gauge_length) if \
                    hasattr(sec, 'gauge_length') else '-1'

                if hasattr(sec, 'origin_time'):
                    attrs['event/time'] = sec.origin_time.\
                        strftime('%Y%m%dT%H%M%S%f')

            elif file_format == 'INGV':
                h5_file.create_dataset('ChannelMap',
                    data=[-1] * sec.start_channel + list(range(sec.nch)))
                h5_file.create_dataset('Fiber', data=sec.data.T)
                h5_file.create_dataset('cm', data=sec.end_channel * [0])
                h5_file.create_dataset('t', data=[])
                h5_file.create_dataset('x', data=sec.nch * [0])

                h5_file.attrs['Start Distance (m)'] = \
                    [sec.start_distance - sec.start_channel * sec.dx]
                h5_file.attrs['Stop Distance (m)'] = [sec.end_distance]
                h5_file.attrs['Samplerate'] = [sec.fs]
                h5_file.attrs['SamplingFrequency[Hz]'] = [sec.fs]
                h5_file.attrs['StartTime'] = [int(sec.start_time.timestamp() *
                                                  1e6)]

                gauge_length = sec.gauge_length if \
                    hasattr(sec, 'gauge_length') else -1
                scale = sec.scale if hasattr(sec, 'scale') else 1
                h5_file.attrs['GaugeLength'] = [gauge_length]
                h5_file.attrs['FilterGain'] = 116e-9 * sec.fs / gauge_length / \
                    8192 / scale

            elif file_format in 'JAMSTEC':
                h5_file.create_dataset('DAS_record', data=sec.data)
                h5_file.create_dataset('Sampling_interval_in_space',
                                       data=[sec.dx])
                h5_file.create_dataset('Sampling_interval_in_time',
                                       data=[1/sec.fs])

            elif file_format == 'NEC':
                h5_file.create_dataset('data', data=sec.data)
                h5_file['data'].attrs['Interval of monitor point'] = sec.dx
                h5_file['data'].attrs['Number of requested location points'] \
                    = sec.nch
                h5_file['data'].attrs['Interval time of data'] = \
                    1 / sec.fs * 1e3
                h5_file['data'].attrs['Radians per digital value'] = \
                    sec.scale if hasattr(sec, 'scale') else 1
                if isinstance(sec.start_time, datatime):
                    t0 = sec.start_time.timestamp()
                else:
                    t0 = sec.start_time
                h5_file['data'].attrs['Time of sending request'] = t0 * 1e3
                h5_file['data'].attrs['Gauge length'] = sec.gauge_length if \
                    hasattr(sec, 'gauge_length') else -1

            elif file_format == 'FORESEE':
                h5_file.create_dataset('raw', data=sec.data)
                start_time = sec.start_time.timestamp() if \
                    isinstance(sec.start_time, datetime) else sec.start_time
                timestamp = start_time + np.arange(0, sec.nsp/sec.fs, 1/sec.fs)
                h5_file.create_dataset('timestamp', data=timestamp)

            elif file_format == 'AI4EPS':
                h5_file.create_dataset('data', data=sec.data)
                h5_file['data'].attrs['dx_m'] = sec.dx
                h5_file['data'].attrs['dt_s'] = 1 / sec.fs
                start_time = sec.start_time if \
                    isinstance(sec.start_time, datetime) else \
                    DASDateTime.fromtimestamp(sec.start_time)
                h5_file['data'].attrs['begin_time'] = start_time.isoformat()
                h5_file['data'].attrs['unit'] = sec.data_type if \
                    hasattr(sec, 'data_type') else 'unknown'
                if hasattr(sec, 'origin_time') and \
                    isinstance(sec.origin_time, datetime):
                    h5_file['data'].attrs['event_time'] = \
                        sec.origin_time.isoformat()

            elif file_format == 'Unknown0':
                h5_file.create_dataset('data_product/data', data=sec.data)
                h5_file.attrs['nx'] = sec.nch
                h5_file.attrs['dx'] = sec.dx 
                h5_file.attrs['dt_computer'] = 1 / sec.fs
                h5_file.attrs['gauge_length'] = sec.gauge_length if \
                    hasattr(sec, 'gauge_length') else -1

                timestamp = sec.start_time.timestamp() if \
                    isinstance(sec.start_time, datetime) else sec.start_time
                h5_file.attrs['saving_start_gps_time'] = timestamp
                h5_file.attrs['file_start_computer_time'] = timestamp
                h5_file.attrs['data_product']= sec.data_type if \
                    hasattr(sec, 'data_type') else 'unknown'
    else:
        # Copy template file and update data
        if not os.path.exists(fname) or not os.path.samefile(raw_fname, fname):
            copyfile(raw_fname, fname)
        with h5py.File(fname, 'r+') as h5_file:
            file_format = _h5_file_format(h5_file)
            keys = h5_file.keys()
            group = list(keys)[0]
            if file_format == 'AP Sensing':
                _update_h5_dataset(h5_file, '/', 'strain', sec.data.T)
                _update_h5_dataset(h5_file, '/', 'spatialsampling', sec.dx)
                _update_h5_dataset(h5_file, '/', 'RepetitionFrequency', sec.fs)
                _update_h5_dataset(h5_file, '/', 'GaugeLength',
                    sec.gauge_length if hasattr(sec, 'gauge_length') else -1)

            elif file_format == 'Aragón Photonics HDAS':
                for key in keys:
                    if 'data' in key.lower():
                        data_key = key
                    elif 'header' in key.lower():
                        header_key = key

                header_raw = h5_file[header_key][()]
                if header_raw.ndim == 2:
                    header = header_raw[0]
                else:
                    header = header_raw

                if 'start_time' in h5_file[data_key].attrs.keys():
                    h5_file[data_key].attrs['start_time'] = \
                        sec.start_time.isoformat()
                    _update_h5_dataset(h5_file, '/', data_key, sec.data)
                else:
                    header[100] = sec.start_time.timestamp()
                    _update_h5_dataset(h5_file, '/', data_key, sec.data.T)

                header[1] = sec.dx 
                header[98] = header[6] / header[15] / sec.fs
                header[11] = sec.start_distance
                if header_raw.ndim == 2:
                    header = header.reshape((1, -1))
                _update_h5_dataset(h5_file, '/', header_key, header)

            elif file_format == 'ASN OptoDAS':
                if h5_file['header/dimensionNames'][0] == b'time':
                    _update_h5_dataset(h5_file, '/', 'data', sec.data.T)
                elif h5_file['header/dimensionNames'][0] == b'distance':
                    _update_h5_dataset(h5_file, '/', 'data', sec.data)

                _update_h5_dataset(h5_file, 'header', 'dx', sec.dx)
                _update_h5_dataset(h5_file, 'header', 'dt', 1 / sec.fs)
                _update_h5_dataset(h5_file, 'header', 'time',
                    sec.start_time.timestamp() if
                    isinstance(sec.start_time, datetime) else sec.start_time)
                _update_h5_dataset(h5_file, '/', 'gaugeLength',
                    sec.gauge_length if hasattr(sec, 'gauge_length') else -1)
                _update_h5_dataset(h5_file, '/', 'dataScale',
                                   sec.scale if hasattr(sec, 'scale') else 1)

            elif file_format in ['Febus A1-R', 'Febus A1']:
                acquisition = list(h5_file[f'{group}/Source1/Zone1'].keys())[0]
                data = sec.data
                fs = int(sec.fs)
                d = len(h5_file[f'{group}/Source1/Zone1/{acquisition}'].shape)
                if d == 3:
                    mod = sec.nsp % fs
                    seconds = sec.nsp // fs
                    if mod:
                        data = np.hstack((data, np.zeros((sec.nch, fs - mod))))
                        seconds += 1
                    data = data.reshape((sec.nch, fs, seconds)).T
                elif d == 2:
                    data = data.T
                    seconds = np.ceil(sec.nsp / sec.fs).astype(int)
                _update_h5_dataset(h5_file, f'{group}/Source1/Zone1/',
                                    acquisition, data)
                attrs = h5_file[f'{group}/Source1/Zone1'].attrs
                attrs['Spacing'][0] = sec.dx
                attrs['Spacing'][1] = 1000 / sec.fs
                attrs['FreqRes'] = np.bytes_(str(sec.fs))
                attrs['SamplingRate'][0] = sec.fs
                attrs['SamplingRes'][0] = attrs['PulseRateFreq'][0] / 1000 / \
                    sec.fs
                attrs['Extent'][0] = sec.start_channel
                attrs['Origin'][0] = sec.start_distance
                if hasattr(sec, 'gauge_length'):
                    attrs['GaugeLength'][0] = sec.gauge_length
                else:
                    attrs['GaugeLength'] = -1
                DataTime = sec.start_time.timestamp() + np.arange(0, seconds)
                if len(h5_file[f'{group}/Source1/time'].shape) == 2:
                    DataTime = DataTime.reshape((1, -1))
                _update_h5_dataset(h5_file, f'{group}/Source1/', 'time',
                                   DataTime)

            elif file_format == 'OptaSense ODH3':
                _update_h5_dataset(h5_file, '/', 'data', sec.data)
                _update_h5_dataset(h5_file, '/', 'x_axis',
                    sec.start_distance + np.arange(sec.nch) * sec.dx)
                _update_h5_dataset(h5_file, '/', 't_axis',
                                   sec.start_time + np.arange(sec.nsp) * sec.dt)

            elif file_format == 'OptaSense ODH4':
                _update_h5_dataset(h5_file, '/', 'raw_data', sec.data)
                h5_file.attrs['channel spacing m'] = sec.dx
                h5_file.attrs['sampling rate Hz'] = sec.fs
                h5_file.attrs['channel_start'] = sec.start_channel
                h5_file.attrs['starttime'] = sec.start_time.isoformat()
                h5_file.attrs['raw_data_units'] = sec.data_type if \
                    hasattr(sec, 'data_type') else 'unknown'

                h5_file.attrs['scale factor to strain'] = sec.scale if \
                    hasattr(sec, 'scale') else 1

            elif file_format in ['OptaSense ODH4+', 'OptaSense QuantX',
                                'Silixa iDAS-MG', 'Sintela Onyx v1.0',
                                'Smart Earth ZD-DAS', 'Unknown']:
                if file_format in ['Silixa iDAS-MG', 'Sintela Onyx v1.0',
                                   'Smart Earth ZD-DAS']:
                    _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/',
                                       'RawData', sec.data.T)
                else:
                    _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/',
                                       'RawData', sec.data)
                h5_file['Acquisition'].attrs['NumberOfLoci'] = sec.nch
                if isinstance(sec.start_time, datetime):
                    if isinstance(h5_file['Acquisition/Raw[0]/RawData'].
                                    attrs['PartStartTime'], bytes):
                        h5_file['Acquisition/Raw[0]/RawData'].\
                            attrs['PartStartTime'] = np.bytes_(
                            sec.start_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z'))
                    else:
                        h5_file['Acquisition/Raw[0]/RawData'].\
                            attrs['PartStartTime'] = sec.start_time.strftime(
                                '%Y-%m-%dT%H:%M:%S.%f%z')
                    stime = sec.start_time.timestamp() * 1e6
                    DataTime = np.arange(
                        stime, stime + sec.nsp / sec.fs, 1 / sec.fs)
                else:
                    h5_file['Acquisition/Raw[0]/RawData'].\
                        attrs['PartStartTime'] = np.bytes_(str(sec.start_time))
                    DataTime = sec.start_time + np.arange(0, sec.nsp / sec.fs,
                                                        1 / sec.fs)
                _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/',
                                    'RawDataTime', DataTime)
                h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
                h5_file['Acquisition'].attrs['SpatialSamplingInterval'] = \
                    sec.dx
                h5_file['Acquisition'].attrs['GaugeLength'] = \
                    sec.gauge_length if hasattr(sec, 'gauge_length') else -1

            elif file_format == 'Puniu Tech HiFi-DAS':
                attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k,
                         v in h5_file['default'].attrs.items()}
                if 'time,channel' in attrs.get('row_major_order',
                    'time, channel').replace(' ', '').lower():
                    _update_h5_dataset(h5_file, '/', 'default', sec.data.T)
                else:
                    _update_h5_dataset(h5_file, '/', 'default', sec.data)

                h5_file['default'].attrs['spatial_sampling_rate'] = sec.dx / \
                    attrs['step']
                h5_file['default'].attrs['sampling_rate'] = sec.fs
                h5_file['default'].attrs['start_channel'] = sec.start_channel
                if isinstance(sec.start_time, datetime):
                    t0 = sec.start_time.timestamp()
                else:
                    t0 = sec.start_time
                epoch = int(t0)
                h5_file['default'].attrs['epoch'] = epoch
                h5_file['default'].attrs['ns'] = int(round((t0 - epoch) * 1e9))
                if hasattr(sec, 'data_type'):
                    h5_file['default'].attrs['format'] = 'differential' if \
                        sec.data_type == 'strain rate' else sec.data_type
                if hasattr(sec, 'gauge_length'):
                    h5_file['default'].attrs['spatial_resolution'] = \
                        sec.gauge_length
                try:
                    h5_file['default'].attrs['cid'] = \
                        sec.headers['default']['attrs']['cid']
                except KeyError:
                    pass

            elif file_format == 'Silixa iDAS':
                if h5_file['Acquisition/Raw[0]/RawData/'].shape[0] != \
                    h5_file['Acquisition/Raw[0]'].attrs['NumberOfLoci']:
                    _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/',
                                       'RawData', sec.data.T)
                else:
                    _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/',
                                       'RawData', sec.data)
                h5_file['Acquisition/Raw[0]'].attrs['NumberOfLoci'] = sec.nch
                h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
                h5_file['Acquisition/Raw[0]'].attrs['AmpScaling'] = sec.scale \
                    if hasattr(sec, 'scale') else 1
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                    np.bytes_(sec.start_time.isoformat())
                h5_file['Acquisition/Custom/UserSettings'].\
                    attrs['StartDistance'] = sec.start_distance
                h5_file['Acquisition'].attrs['GaugeLength'] = sec.gauge_length \
                    if hasattr(sec, 'gauge_length') else -1
                _update_h5_dataset(h5_file, 'Mapping/',
                                   'MeasuredSpatialResolution',
                                   np.full(sec.nch, sec.dx))
                if hasattr(sec, 'geometry'):
                    _update_h5_dataset(h5_file, 'Mapping/', 'Lon',
                                       sec.geometry[:,0])
                    _update_h5_dataset(h5_file, 'Mapping/', 'Lat',
                                       sec.geometry[:,1])

            elif file_format == 'T8 Sensor':
                ds_name = list(h5_file['/ProcessedData'].keys())[0]
                dpath = f'/ProcessedData/{ds_name}'
                _update_h5_dataset(h5_file, '/ProcessedData/', ds_name,
                                   sec.data.T)
                attrs = h5_file[dpath].attrs
                attrs['downsampling/decimation_in_length'] = str(
                    float(attrs['record/channel_spacing_m']) / sec.dx)
                attrs['downsampling/decimation_in_time'] = str(
                    float(attrs['record/pulse_repetition_rate_hz']) / sec.fs)
                attrs['record/line_offset_m'] = str(sec.start_distance)
                attrs['record/start_time'] = \
                    sec.start_time.strftime('%Y%m%dT%H%M%S%f')
                attrs['record/gauge_length_m'] = str(sec.gauge_length) if \
                    hasattr(sec, 'gauge_length') else -1
                if hasattr(sec, 'origin_time'):
                    attrs['event/time'] = \
                        sec.origin_time.strftime('%Y%m%dT%H%M%S%f')

            elif file_format == 'INGV':
                if sec.nch != h5_file['Fiber'].shape[1]:
                    _update_h5_dataset(h5_file, '/', 'x', sec.nch * [0])
                    channelmap = [-1] * len(h5_file['ChannelMap'])
                    channelmap[sec.start_channel:sec.end_channel] = \
                        range(sec.nch)

                _update_h5_dataset(h5_file, '/', 'Fiber', sec.data.T)
                h5_file.attrs['Samplerate'] = [sec.fs]
                h5_file.attrs['StartTime'] = [int(sec.start_time.timestamp() *
                                                  1e6)]
                if hasattr(sec, 'gauge_length'):
                    h5_file.attrs['GaugeLength'] = [sec.gauge_length]
                if hasattr(sec, 'scale'):
                    h5_file.attrs['FilterGain'] = 116e-9 * \
                        h5_file.attrs['SamplingFrequency[Hz]'] / \
                        h5_file.attrs['GaugeLength'][0] / 8192 / sec.scale

            elif file_format in 'JAMSTEC':
                _update_h5_dataset(h5_file, '/', 'DAS_record', sec.data)
                _update_h5_dataset(h5_file, '/', 'Sampling_interval_in_space',
                                   [sec.dx])
                _update_h5_dataset(h5_file, '/', 'Sampling_interval_in_time',
                                   [1/sec.fs])

            elif file_format == 'NEC':
                _update_h5_dataset(h5_file, '/', 'data', sec.data)
                h5_file['data'].attrs['Interval of monitor point'] = sec.dx
                h5_file['data'].attrs['Number of requested location points'] \
                    = sec.nch
                h5_file['data'].attrs['Interval time of data'] = \
                    1 / sec.fs * 1e3
                if hasattr(sec, scale):
                    h5_file['data'].attrs['Radians per digital value'] = \
                        sec.scale
                if isinstance(sec.start_time, datatime):
                    t0 = sec.start_time.timestamp()
                else:
                    t0 = sec.start_time
                h5_file['data'].attrs['Time of sending request'] = t0 * 1e3
                if hasattr(sec, 'gauge_length'):
                    h5_file['data'].attrs['Gauge length'] = sec.gauge_length

            elif file_format == 'FORESEE':
                _update_h5_dataset(h5_file, '/', 'raw', sec.data)
                DataTime = sec.start_time.timestamp() + \
                    np.arange(0, sec.nsp / sec.fs, 1 / sec.fs)
                _update_h5_dataset(h5_file, '/', 'timestamp', DataTime)

            elif file_format == 'AI4EPS':
                # https://ai4eps.github.io/homepage/ml4earth/seismic_event_format_das/
                _update_h5_dataset(h5_file, '/', 'data', sec.data)
                h5_file['data'].attrs['dx_m'] = sec.dx
                h5_file['data'].attrs['dt_s'] = 1 / sec.fs
                h5_file['data'].attrs['begin_time'] = \
                    datetime.strftime(sec.start_time, '%Y-%m-%dT%H:%M:%S.%f%z')
                h5_file['data'].attrs['unit'] = sec.data_type

            elif file_format == 'Unknown0':
                _update_h5_dataset(h5_file, 'data_product/', 'data', sec.data)
                h5_file.attrs['dt_computer'] = 1 / sec.fs
                h5_file.attrs['dx'] = sec.dx
                h5_file.attrs['gauge_length'] = sec.gauge_length
                DataTime = sec.start_time.timestamp() + \
                    np.arange(0, sec.nsp / sec.fs, 1 / sec.fs)
                if h5_file.attrs['saving_start_gps_time'] > 0:
                    h5_file.attrs['file_start_gps_time'] = \
                        sec.start_time.timestamp()
                    _update_h5_dataset(h5_file, 'data_product/', 'gps_time',
                                        DataTime)
                    del h5_file['data_product/posix_time']
                else:
                    h5_file.attrs['file_start_computer_time'] = \
                        sec.start_time.timestamp()
                    _update_h5_dataset(h5_file, 'data_product/', 'posix_time',
                                        DataTime)
                    del h5_file['data_product/gps_time']
                h5_file.attrs['data_product'] = sec.data_type
    return None


def _write_tdms(sec, fname, raw_fname=None, file_format='auto'):
    if raw_fname is None:
        key = 'Measurement'
        file_prop = {}
        group_prop = {}
    else:
        original_file = TdmsFile(raw_fname)
        group_name = [group.name for group in original_file.groups()]
        if 'Measurement' in group_name:
            key = 'Measurement'
        elif 'DAS' in group_name:
            key = 'DAS'
        else:
            key = group_name[0]
        file_prop = original_file.properties
        group_prop = original_file[key].properties

    if 'Spatial Resolution' in group_prop.keys():
        group_prop['Spatial Resolution'] = sec.dx
    else:
        file_prop['SpatialResolution[m]'] = sec.dx

    if 'Time Base' in group_prop.keys():
        group_prop['Time Base'] = 1. / sec.fs
    else:
        file_prop['SamplingFrequency[Hz]'] = sec.fs

    if 'Total Channels' in group_prop.keys():
        group_prop['Total Channels'] = sec.nch

    if 'Initial Channel' in group_prop.keys():
        group_prop['Initial Channel'] = sec.start_channel

    file_prop['Start Distance (m)'] = sec.start_distance
    if isinstance(sec.start_time, datetime):
        start_time = sec.start_time
    else:
        start_time = datetime.fromtimestamp(sec.start_time)

    if raw_fname is None:
        file_prop['ISO8601 Timestamp'] = start_time.strftime(
            '%Y-%m-%dT%H:%M:%S.%f%z')
        group_prop['Trigger Time'] = np.datetime64(start_time.remove_tz())
    else:
        if 'ISO8601 Timestamp' in file_prop.keys():
            file_prop['ISO8601 Timestamp'] = start_time.strftime(
                '%Y-%m-%dT%H:%M:%S.%f%z')
        else:
            for s in ['GPSTimeStamp', 'CPUTimeStamp', 'Trigger Time']:
                if s in group_prop.keys():
                    group_prop[s] = np.datetime64(start_time.remove_tz())
                    break

    if hasattr(sec, 'gauge_length'):
        file_prop['GaugeLength'] = sec.gauge_length

    with TdmsWriter(fname) as tdms_file:
        root_object = RootObject(file_prop)
        group_object = GroupObject(key, properties=group_prop)
        if raw_fname and len(original_file[key]) == 1:
            channel = ChannelObject(key, original_file[key].channels()[0].name,
                                    sec.data.T.flatten(), properties={})
            tdms_file.write_segment([root_object, group_object, channel])
        else:
            channel_list = []
            for ch, d in enumerate(sec.data):
                channel_list.append(ChannelObject(key,
                                                  str(ch + sec.start_channel),
                                                  d, properties={}))

            tdms_file.write_segment([root_object, group_object] + channel_list)
    return None


def _write_segy(sec, fname, raw_fname=None, file_format='auto'):
    spec = segyio.spec()
    spec.samples = np.arange(sec.nsp) / sec.fs * 1e3
    spec.tracecount = sec.nch
    if raw_fname is None:
        spec.format = 1
        with segyio.create(fname, spec) as new_file:
            new_file.header.length = sec.nch
            new_file.header.segy._filename = fname
            new_file.trace = sec.data # .astype(np.float32)
    else:
        with segyio.open(raw_fname, ignore_geometry=True) as raw_file:
            spec.sorting = raw_file.sorting
            spec.format = raw_file.format
            raw_file.header.length = sec.nch
            raw_file.header.segy._filename = fname
            with segyio.create(fname, spec) as new_file:
                new_file.text[0] = raw_file.text[0]
                new_file.header = raw_file.header
                new_file.trace = sec.data.astype(raw_file.trace.dtype)

    warnings.warn('This data format doesn\'t include channel interval.')
    return None