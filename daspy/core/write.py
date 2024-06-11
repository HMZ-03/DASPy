# Purpose: Module for writing DAS data.
# Author: Minzhe Hu
# Date: 2024.6.11
# Email: hmz2018@mail.ustc.edu.cn
import warnings
import pickle
import numpy as np
import h5py
import segyio
from shutil import copyfile
from nptdms import TdmsFile, TdmsWriter, RootObject, GroupObject, ChannelObject
from datetime import datetime


def write_pkl(fname, sec):
    with open(fname, 'wb') as f:
        pickle.dump(sec.__dict__, f)
    return None


def update(raw_fname, new_fname, sec):
    fun_map = {'tdms': _update_tdms, 'h5': _update_h5,
                        'hdf5': _update_h5, 'segy': _update_segy,
                        'sgy': _update_segy}
    ftype = raw_fname.lower().split('.')[-1]
    if new_fname.lower().split('.')[-1] != ftype:
        raise KeyError('Format of new_fname and raw_fname should be same.')
    fun_map[ftype](raw_fname, new_fname, sec)
    return None


def _update_tdms(raw_fname, new_fname, sec):
    original_file = TdmsFile(raw_fname)
    original_prop = original_file.properties
    original_prop['SpatialResolution[m]'] = sec.dx
    original_prop['SamplingFrequency[Hz]'] = sec.fs
    original_prop['GaugeLength'] = sec.gauge_length
    original_prop['Start Distance (m)'] = sec.start_distance
    original_prop['ISO8601 Timestamp'] = \
        sec.start_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z')

    with TdmsWriter(new_fname) as tdms_file:
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


def _update_h5(raw_fname, new_fname, sec):
    copyfile(raw_fname, new_fname)
    with h5py.File(new_fname, 'r+') as h5_file:
        h5_file['Acquisition'].attrs['NumberOfLoci'] = sec.nch
        _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/', 'RawData', sec.data)
        if isinstance(sec.start_time, datetime):
            h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                np.bytes_(sec.start_time.strftime('%Y-%m-%dT%H:%M:%S.%f%z'))
            stime = sec.start_time.timestamp() * 1e6
            _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/', 'RawDataTime',
                               np.arange(stime, stime + sec.nt / sec.fs,
                                         1 / sec.fs))
        else:
            # stime = .encode('ascii')
            h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime'] = \
                np.bytes_(str(sec.start_time))
            _update_h5_dataset(h5_file, 'Acquisition/Raw[0]/', 'RawDataTime',
                               sec.start_time + np.arange(0, sec.nt / sec.fs,
                                                          1 / sec.fs))

        h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate'] = sec.fs
        h5_file['Acquisition'].attrs['SpatialSamplingInterval'] = sec.dx
        h5_file['Acquisition'].attrs['GaugeLength'] = sec.gauge_length
    return None


def _update_segy(raw_fname, new_fname, sec):
    with segyio.open(raw_fname, ignore_geometry=True) as raw_file:
        spec = segyio.spec()
        spec.sorting = raw_file.sorting
        spec.format = raw_file.format
        spec.samples = (np.arange(sec.nt) / sec.fs)
        spec.tracecount = sec.nch
        raw_file.header.length = sec.nch
        raw_file.header.segy._filename = new_fname
        with segyio.create(new_fname, spec) as new_file:
            warnings.warn('This data format doesn\'t include channel interval.')
            new_file.text[0] = raw_file.text[0]
            new_file.header = raw_file.header
            new_file.trace = sec.data.astype(raw_file.trace.dtype)

    return None
