# Purpose: Module for reading DAS data.
# Author: Robbin Luo, Minzhe Hu
# Date: 2024.3.27
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


def read(fname=None, output_type='Section', ch1=0, ch2=None) -> Section:
    """
    :param fname: Path of DAS data file.
    :param output_type: str. 'Section' means output as Section class, 'array'
        means output numpy.array for data and a dictionary for metadata
    """
    if fname is None:
        with open(Path(__file__).parent / 'example.pkl', 'rb') as f:
            sec_dict = pickle.load(f)
        return Section(**sec_dict)
    elif fname.lower().endswith('.pkl'):
        with open(fname, 'rb') as f:
            sec_dict = pickle.load(f)
        return Section(**sec_dict)
    elif fname.lower().endswith('.tdms'):
        data, metadata = _read_tdms(fname, ch1=ch1, ch2=ch2)
    elif fname.lower().endswith('.h5'):
        data, metadata = _read_h5(fname, ch1=ch1, ch2=ch2)
    elif fname.lower().endswith(('.segy', 'sgy')):
        data, metadata = _read_segy(fname, ch1=ch1, ch2=ch2)
        warnings.warn('This data format doesn\'t include channel interval.')
    else:
        raise ValueError('DAS data format not supported.')

    if output_type == 'array':
        dx = metadata.pop('dx', None)
        fs = metadata.pop('fs')
        return data, dx, fs, metadata
    elif output_type == 'Section':
        if metadata['dx'] is None:
            print('Please set Section.dx manually.')
        if metadata['fs'] is None:
            print('Please set Section.fs manually.')

        return Section(data.astype(float), **metadata)


def _read_h5(fname, ch1=0, ch2=None):
    with h5py.File(fname, 'r') as h5_file:
        nch = h5_file['Acquisition'].attrs['NumberOfLoci']
        ch2 = (ch2, nch)[ch2 is None]

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
            stime = DASDateTime.strptime(stime_str[:-6], '%Y-%m-%dT%H:%M:%S.%f')
            tz_hours, tz_minutes = int(stime_str[-6:-3]), int(stime_str[-2:])
            tz = timezone(timedelta(hours=tz_hours, minutes=tz_minutes))
            stime.replace(tzinfo=tz)
            metadata['start_time'] = stime

    return data, metadata


def _read_tdms(fname, ch1=0, ch2=None):
    # https://nptdms.readthedocs.io/en/stable/quickstart.html
    with TdmsFile.read(fname) as tdms_file:
        nch = len(tdms_file['Measurement'])
        ch2 = (ch2, nch)[ch2 is None]

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


def _read_segy(fname, ch1=0, ch2=None):
    # https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb

    with segyio.open(fname, ignore_geometry=True) as segy_file:
        nch = segy_file.tracecount
        ch2 = (ch2, nch)[ch2 is None]

        # read data
        data = segy_file.trace.raw[ch1:ch2]

        # read metadata:
        fs = 1 / (segyio.tools.dt(segy_file) / 1e6)
        metadata = {'fs': fs, 'dx':None, 'start_channel': ch1}

        return data, metadata
