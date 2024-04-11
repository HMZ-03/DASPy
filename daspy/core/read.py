# Purpose: Module for reading DAS data.
# Author: Minzhe Hu
# Date: 2024.4.11
# Email: hmz2018@mail.ustc.edu.cn
# Modified from
# https://github.com/RobbinLuo/das-toolkit/blob/main/DasTools/DasPrep.py
import warnings
import pickle
import numpy as np
import h5py
import segyio
from pathlib import Path
from nptdms import TdmsFile
from datetime import timedelta, timezone
from daspy.core.section import Section
from daspy.core.dasdatetime import DASDateTime


def read(fname=None, output_type='Section', **kwargs):
    """
    :param fname: Path of DAS data file.
    :param output_type: str. 'Section' means output an instances of
        daspy.Section, 'array' means output numpy.array for data and a
        dictionary for metadata
    :param ch1: int. The first channel required.
    :param ch2: int. The last channel required (not included).
    :return: An instances of daspy.Section, or numpy.array for data and a
        dictionary for metadata.
    """
    fun_map = {'pkl': _read_pkl, 'tdms': _read_tdms, 'h5': _read_h5,
               'segy': _read_segy, 'sgy': _read_segy}
    if fname is None:
        data, metadata = _read_pkl(Path(__file__).parent / 'example.pkl')
    else:
        ftype = fname.lower().split('.')[-1]
        data, metadata = fun_map[ftype](fname, **kwargs)

    if output_type == 'Section':
        if metadata['dx'] is None:
            print('Please set Section.dx manually.')
        if metadata['fs'] is None:
            print('Please set Section.fs manually.')
        return Section(data.astype(float), **metadata)
    elif output_type == 'array':
        return data, metadata


def _read_pkl(fname, **kwargs):
    with open(fname, 'rb') as f:
        sec_dict = pickle.load(f)

    data = sec_dict.pop('data')
    if 'ch1' in kwargs.keys() or 'ch2' in kwargs.keys():
        if 'start_channel' in sec_dict.keys():
            s_chn = sec_dict['start_channel']
            print(f'Data is start with channel {s_chn}.')
        else:
            s_chn = 0
        ch1 = kwargs.pop('ch1', s_chn)
        ch2 = kwargs.pop('ch2', s_chn + len(data))
        data = data[ch1 - s_chn:ch2 - s_chn, :]
        sec_dict['start_channel'] = ch1

    return data, sec_dict


def _read_h5(fname, **kwargs):
    with h5py.File(fname, 'r') as h5_file:
        nch = h5_file['Acquisition'].attrs['NumberOfLoci']
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', nch)

        # read data
        array_shape = h5_file['Acquisition/Raw[0]/RawData/'].shape
        if array_shape[0] == nch:
            data = h5_file['Acquisition/Raw[0]/RawData/'][ch1:ch2, :]
        else:
            data = h5_file['Acquisition/Raw[0]/RawData/'][:, ch1:ch2].T

        # read metadata
        try:
            time_arr = h5_file['Acquisition/Raw[0]/RawDataTime/']
            if len(time_arr) == 0:
                warnings.warn('This data doesn\'t include Data time.')
                fs = None
            else:
                fs = 1 / (np.diff(time_arr).mean() / 1e6)
        except KeyError:
            warnings.warn('This data doesn\'t include Data time.')
            fs = None
        headers = dict(h5_file['Acquisition'].attrs)
        dx = headers['SpatialSamplingInterval']
        gauge_length = headers['GaugeLength']
        metadata = {'fs': fs, 'dx': dx, 'start_channel': ch1,
                    'start_distance': ch1 * dx, 'gauge_length': gauge_length,
                    'headers': headers}
        if 'MeasurementStartTime' in headers.keys():
            stime_str = headers['MeasurementStartTime'].decode('ascii')
            stime = DASDateTime.strptime(
                stime_str[:-6], '%Y-%m-%dT%H:%M:%S.%f')
            tz_hours, tz_minutes = int(stime_str[-6:-3]), int(stime_str[-2:])
            tz = timezone(timedelta(hours=tz_hours, minutes=tz_minutes))
            stime.replace(tzinfo=tz)
            metadata['start_time'] = stime

    return data, metadata


def _read_tdms(fname, **kwargs):
    # https://nptdms.readthedocs.io/en/stable/quickstart.html
    with TdmsFile.read(fname) as tdms_file:
        nch = len(tdms_file['Measurement'])
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', nch)

        # read data
        data = np.asarray([tdms_file['Measurement'][str(i)]
                           for i in range(ch1, ch2)])

        # read metadata
        fs = tdms_file.properties['SamplingFrequency[Hz]']
        dx = tdms_file.properties['SpatialResolution[m]']
        gauge_length = tdms_file.properties['GaugeLength']
        headers = tdms_file.properties

    metadata = {'fs': fs, 'dx': dx, 'start_channel': ch1,
                'start_distance': ch1 * dx, 'gauge_length': gauge_length,
                'headers': headers}

    return data, metadata


def _read_segy(fname, **kwargs):
    # https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb

    with segyio.open(fname, ignore_geometry=True) as segy_file:
        warnings.warn('This data format doesn\'t include channel interval.')
        nch = segy_file.tracecount
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', nch)

        # read data
        data = segy_file.trace.raw[ch1:ch2]

        # read metadata:
        fs = 1 / (segyio.tools.dt(segy_file) / 1e6)
        metadata = {'fs': fs, 'dx': None, 'start_channel': ch1}

        return data, metadata
