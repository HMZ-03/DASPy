# Purpose: Module for writing DAS data.
# Author: Minzhe Hu
# Date: 2024.11.20
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


def write(sec, fname, ftype=None, raw_fname=None):
    fun_map = {'tdms': _write_tdms, 'h5': _write_h5, 'sgy': _write_segy}
    if ftype is None:
        ftype = str(fname).lower().split('.')[-1]
    ftype.replace('hdf5', 'h5')
    ftype.replace('segy', 'sgy')
    if ftype == 'pkl':
        write_pkl(sec, fname)
    elif ftype == 'npy':
        np.save(fname, sec.data)
    else:
        fun_map[ftype](sec, fname, raw_fname=raw_fname)
    return None


def write_pkl(sec, fname):
    with open(fname, 'wb') as f:
        pickle.dump(sec.__dict__, f)
    return None


def _write_tdms(sec, fname, raw_fname=None):
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


def _update_h5_dataset(h5_file, path, name, data):
    attrs = h5_file[path + name].attrs
    del h5_file[path + name]
    h5_file.get(path).create_dataset(name, data=data)
    for key, value in attrs.items():
        h5_file[path + name].attrs[key] = value
    return None


def _write_h5(sec, fname, raw_fname=None):
    if raw_fname is None:
        with h5py.File(fname, 'w') as h5_file:
            h5_file.create_group('Acquisition/Raw[0]')
            h5_file.get('Acquisition/Raw[0]/').\
                create_dataset('RawData', data=sec.data)
            if isinstance(sec.start_time, datetime):
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                    np.bytes_(
                    sec.start_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z'))
                stime = sec.start_time.timestamp() * 1e6
                DataTime = np.arange(
                    stime, stime + sec.nt / sec.fs, 1 / sec.fs)
            else:
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                    np.bytes_(str(sec.start_time))
                DataTime = sec.start_time + np.arange(0, sec.nt / sec.fs,
                                                      1 / sec.fs)

            h5_file.get('Acquisition/Raw[0]/').\
                create_dataset('RawDataTime', data=DataTime)
            h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
            h5_file['Acquisition'].attrs['SpatialSamplingInterval'] = sec.dx
            if hasattr(sec, 'gauge_length'):
                h5_file['Acquisition'].attrs['GaugeLength'] = sec.gauge_length
            else:
                h5_file['Acquisition'].attrs['GaugeLength'] = np.nan
    else:
        if not os.path.exists(fname) or not os.path.samefile(raw_fname, fname):
            copyfile(raw_fname, fname)
        with h5py.File(fname, 'r+') as h5_file:
            group = list(h5_file.keys())[0]
            if len(h5_file.keys()) == 10:
                if h5_file['header/dimensionNames'][0] == b'time':
                    _update_h5_dataset(h5_file, '/', 'data', sec.data.T)
                elif h5_file['header/dimensionNames'][0] == b'distance':
                    _update_h5_dataset(h5_file, '/', 'data', sec.data)

                _update_h5_dataset(h5_file, 'header', 'dx', sec.dx)
                _update_h5_dataset(h5_file, 'header', 'dt', 1 / sec.fs)
                if isinstance(sec.start_time, datetime):
                    _update_h5_dataset(h5_file, 'header', 'time',
                                       sec.start_time.timestamp())
                else:
                    _update_h5_dataset(h5_file, 'header', 'time',
                                       sec.start_time)
                if hasattr(sec, 'gauge_length'):
                    _update_h5_dataset(h5_file, '/', 'gaugeLength',
                                       sec.gauge_length)
                if hasattr(sec, 'scale'):
                    _update_h5_dataset(h5_file, '/', 'dataScale', sec.scale)
            elif len(h5_file.keys()) == 5:
                _update_h5_dataset(h5_file, '/', 'strain', sec.data.T)
                _update_h5_dataset(h5_file, '/', 'spatialsampling', sec.dx)
                _update_h5_dataset(h5_file, '/', 'RepetitionFrequency', sec.fs)
                if hasattr(sec, 'gauge_length'):
                    _update_h5_dataset(h5_file, '/', 'GaugeLength',
                                       sec.gauge_length)
            elif group == 'Acquisition':
                h5_file['Acquisition'].attrs['NumberOfLoci'] = sec.nch
                _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/', 'RawData',
                                   sec.data)
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
                        stime, stime + sec.nt / sec.fs, 1 / sec.fs)
                else:
                    h5_file['Acquisition/Raw[0]/RawData'].\
                        attrs['PartStartTime'] = np.bytes_(str(sec.start_time))
                    DataTime = sec.start_time + np.arange(0, sec.nt / sec.fs,
                                                        1 / sec.fs)
                _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/',
                                   'RawDataTime', DataTime)
                h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
                h5_file['Acquisition'].attrs['SpatialSamplingInterval'] = sec.dx
                if hasattr(sec, 'gauge_length'):
                    h5_file['Acquisition'].attrs['GaugeLength'] = \
                        sec.gauge_length
            elif group == 'raw':
                _update_h5_dataset(h5_file, '/', 'raw', sec.data)
                DataTime = sec.start_time.timestamp() + \
                    np.arange(0, sec.nt / sec.fs, 1 / sec.fs)
                _update_h5_dataset(h5_file, '/', 'timestamp', DataTime)
            elif group == 'data': # https://ai4eps.github.io/homepage/ml4earth/seismic_event_format_das/
                _update_h5_dataset(h5_file, '/', 'data', sec.data)
                h5_file['data'].attrs['dx_m'] = sec.dx
                h5_file['data'].attrs['dt_s'] = 1 / sec.fs
                h5_file['data'].attrs['begin_time'] = \
                    datetime.strftime(sec.start_time, '%Y-%m-%dT%H:%M:%S.%f%z')
                h5_file['data'].attrs['unit'] = sec.data_type
            elif group == 'data_product':
                _update_h5_dataset(h5_file, 'data_product/', 'data', sec.data)
                h5_file.attrs['dt_computer'] = 1 / sec.fs
                h5_file.attrs['dx'] = sec.dx
                h5_file.attrs['gauge_length'] = sec.gauge_length
                DataTime = sec.start_time.timestamp() + \
                    np.arange(0, sec.nt / sec.fs, 1 / sec.fs)
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
            else:
                acquisition = list(h5_file[f'{group}/Source1/Zone1'].keys())[0]
                data = sec.data
                fs = int(sec.fs)
                d = len(h5_file[f'{group}/Source1/Zone1/{acquisition}'].shape)
                if d == 3:
                    mod = sec.nt % fs
                    if mod:
                        data = np.hstack((data, np.zeros((sec.nch, fs - mod))))
                    data = data.reshape((sec.nch, fs, sec.nt//fs)).T
                elif d == 2:
                    data = data.T
                _update_h5_dataset(h5_file, f'{group}/Source1/Zone1/',
                                   acquisition, data)

                h5_file[f'{group}/Source1/Zone1'].attrs['Spacing'][0] = sec.dx
                h5_file[f'{group}/Source1/Zone1'].attrs['FreqRes'] = \
                    np.bytes_(sec.fs)
                h5_file[f'{group}/Source1/Zone1'].attrs['SamplingRate'][0] = \
                    sec.fs
                h5_file[f'{group}/Source1/Zone1'].attrs['Extent'][0] = \
                    sec.start_channel
                h5_file[f'{group}/Source1/Zone1'].attrs['Origin'][0] = \
                    sec.start_distance
                h5_file[f'{group}/Source1/Zone1'].attrs['GaugeLength'][0] = \
                    sec.gauge_length
                DataTime = sec.start_time.timestamp() + \
                    np.arange(0, sec.nt / sec.fs, 1 / sec.fs)
                _update_h5_dataset(h5_file, f'{group}/Source1/',
                                   'time', DataTime.reshape((1, -1)))

    return None


def _write_segy(sec, fname, raw_fname=None):
    spec = segyio.spec()
    spec.samples = np.arange(sec.nt) / sec.fs * 1e3
    spec.tracecount = sec.nch
    if raw_fname is None:
        spec.format = 1
        with segyio.create(fname, spec) as new_file:
            new_file.header.length = sec.nch
            new_file.header.segy._filename = fname
            new_file.trace = sec.data.astype(np.float32)
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
