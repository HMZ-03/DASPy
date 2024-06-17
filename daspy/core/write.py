# Purpose: Module for writing DAS data.
# Author: Minzhe Hu
# Date: 2024.6.17
# Email: hmz2018@mail.ustc.edu.cn
import warnings
import pickle
import numpy as np
import h5py
import segyio
from shutil import copyfile
from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject
from datetime import datetime


def write(sec, fname, raw_fname=None):
    fun_map = {'tdms': _write_tdms, 'h5': _write_h5, 'hdf5': _write_h5,
               'segy': _write_segy, 'sgy': _write_segy}
    ftype = fname.lower().split('.')[-1]
    if ftype == 'pkl':
        write_pkl(sec, fname)
    else:
        if raw_fname is not None:
            if raw_fname.lower().split('.')[-1] != ftype:
                raise KeyError('Format of new_fname and raw_fname should be '
                               'same.')
        fun_map[ftype](sec, fname, raw_fname=raw_fname)
    return None


def write_pkl(sec, fname):
    with open(fname, 'wb') as f:
        pickle.dump(sec.__dict__, f)
    return None


def _write_tdms(sec, fname, raw_fname=None):
    if raw_fname is None:
        original_prop = {}
    else:
        original_file = TdmsFile(raw_fname)
        original_prop = original_file.properties

    original_prop['SpatialResolution[m]'] = sec.dx
    original_prop['SamplingFrequency[Hz]'] = sec.fs
    original_prop['Start Distance (m)'] = sec.start_distance
    if isinstance(sec.start_time, datetime):
        original_prop['ISO8601 Timestamp'] = \
            sec.start_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z')
    else:
        original_prop['ISO8601 Timestamp'] = \
            datetime.fromtimestamp(sec.start_time)
    if hasattr(sec, 'gauge_length'):
        original_prop['GaugeLength'] = sec.gauge_length

    with TdmsWriter(fname) as tdms_file:
        root_object = RootObject(original_prop)
        group_object = GroupObject('Measurement', properties={})
        channel_list = []
        for ch, d in enumerate(sec.data):
            channel_list.append(ChannelObject('Measurement',
                                              str(ch + sec.start_channel), d,
                                              properties={}))

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
                # stime = .encode('ascii')
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                    np.bytes_(str(sec.start_time))
                DataTime = sec.start_time + np.arange(0, sec.nt / sec.fs,
                                                      1 / sec.fs)

            h5_file.get('Acquisition/Raw[0]/').\
                create_dataset('RawDataTime', data=DataTime)
            h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
            h5_file['Acquisition'].attrs['SpatialSamplingInterval'] = sec.dx
            h5_file['Acquisition'].attrs['GaugeLength'] = sec.gauge_length
    else:
        copyfile(raw_fname, fname)
        with h5py.File(fname, 'r+') as h5_file:
            h5_file['Acquisition'].attrs['NumberOfLoci'] = sec.nch
            _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/', 'RawData',
                               sec.data)
            if isinstance(sec.start_time, datetime):
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                    np.bytes_(
                    sec.start_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z'))
                stime = sec.start_time.timestamp() * 1e6
                DataTime = np.arange(
                    stime, stime + sec.nt / sec.fs, 1 / sec.fs)
            else:
                # stime = .encode('ascii')
                h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                    np.bytes_(str(sec.start_time))
                DataTime = sec.start_time + np.arange(0, sec.nt / sec.fs,
                                                      1 / sec.fs)
            _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/', 'RawDataTime',
                               DataTime)

            h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
            h5_file['Acquisition'].attrs['SpatialSamplingInterval'] = sec.dx
            h5_file['Acquisition'].attrs['GaugeLength'] = sec.gauge_length
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
