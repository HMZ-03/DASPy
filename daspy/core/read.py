# Purpose: Module for reading DAS data.
# Author: Minzhe Hu
# Date: 2024.6.18
# Email: hmz2018@mail.ustc.edu.cn
# Modified from
# https://github.com/RobbinLuo/das-toolkit/blob/main/DasTools/DasPrep.py
import warnings
import json
import pickle
import numpy as np
import h5py
import segyio
from pathlib import Path
from nptdms import TdmsFile
from daspy.core.section import Section
from daspy.core.dasdatetime import DASDateTime


def read(fname=None, output_type='section', **kwargs):
    """
    Read a .pkl/.pickle, .tdms, .h5/.hdf5, .segy/.sgy file.

    :param fname: str or pathlib.PosixPath. Path of DAS data file.
    :param output_type: str. 'Section' means return an instance of
        daspy.Section, 'array' means return numpy.array for data and a
        dictionary for metadata.
    :param ch1: int. The first channel required.
    :param ch2: int. The last channel required (not included).
    :return: An instance of daspy.Section, or numpy.array for data and a
        dictionary for metadata.
    """
    fun_map = {'pkl': _read_pkl, 'pickle': _read_pkl, 'tdms': _read_tdms,
               'h5': _read_h5, 'hdf5': _read_h5, 'segy': _read_segy,
               'sgy': _read_segy}
    if fname is None:
        data, metadata = _read_pkl(Path(__file__).parent / 'example.pkl')
    else:
        ftype = str(fname).lower().split('.')[-1]
        data, metadata = fun_map[ftype](fname, **kwargs)

    if output_type.lower() == 'section':
        metadata['source'] = Path(fname)
        return Section(data.astype(float), **metadata)
    elif output_type.lower() == 'array':
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


def _read_h5_headers(group):
    headers = {}
    if len(group.attrs) != 0:
        headers['attrs'] = dict(group.attrs)
    for k in group.keys():
        gp = group[k]
        if isinstance(gp, h5py._hl.dataset.Dataset):
            continue
        elif isinstance(gp, h5py._hl.group.Group):
            gp_headers = _read_h5_headers(group[k])
            if len(gp_headers):
                headers[k] = gp_headers
        else:
            headers[k] = gp

    return headers


def _read_h5_starttime(h5_file):
    try:
        stime = h5_file['Acquisition/Raw[0]/RawData'].\
            attrs['PartStartTime'].decode('ascii')
    except KeyError:
        try:
            stime = h5_file['Acquisition'].\
                attrs['MeasurementStartTime'].decode('ascii')
        except KeyError:
            try:
                stime = h5_file['Acquisition/Raw[0]/RawDataTime/'][0]
            except KeyError:
                return 0

    if isinstance(stime, str):
        if len(stime) > 26:
            stime = DASDateTime.strptime(stime, '%Y-%m-%dT%H:%M:%S.%f%z')
        else:
            stime = DASDateTime.strptime(stime, '%Y-%m-%dT%H:%M:%S.%f')
    else:
        stime = DASDateTime.fromtimestamp(stime / 1e6)

    return stime


def _read_h5(fname, **kwargs):
    with h5py.File(fname, 'r') as h5_file:
        # read headers

        try:
            nch = h5_file['Acquisition'].attrs['NumberOfLoci']
        except KeyError:
            nch = len(h5_file['Acquisition/Raw[0]/RawData/'])

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
            fs = h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate']
        except KeyError:
            time_arr = h5_file['Acquisition/Raw[0]/RawDataTime/']
            fs = 1 / (np.diff(time_arr).mean() / 1e6)

        dx = h5_file['Acquisition'].attrs['SpatialSamplingInterval']
        gauge_length = h5_file['Acquisition'].attrs['GaugeLength']
        metadata = {'fs': fs, 'dx': dx, 'start_channel': ch1,
                    'start_distance': ch1 * dx, 'gauge_length': gauge_length,
                    }

        metadata['start_time'] = _read_h5_starttime(h5_file)
        metadata['headers'] = _read_h5_headers(h5_file)

    return data, metadata


def _read_tdms(fname, **kwargs):
    # https://nptdms.readthedocs.io/en/stable/quickstart.html
    with TdmsFile.read(fname) as tdms_file:
        nch = len(tdms_file['Measurement'])
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', nch)

        # read data
        data = np.asarray([tdms_file['Measurement'][str(ch)]
                           for ch in range(ch1, ch2)])

        # read metadata
        headers = tdms_file.properties
        dx = headers['SpatialResolution[m]']
        fs = headers['SamplingFrequency[Hz]']
        gauge_length = headers['GaugeLength']
        min_ch = min(int(channel.name) for channel in
                     tdms_file['Measurement'].channels())
        start_distance = headers['Start Distance (m)'] + dx * (ch1 - min_ch)
        start_time = DASDateTime.strptime(
            tdms_file.properties['ISO8601 Timestamp'], '%Y-%m-%dT%H:%M:%S.%f%z')

    metadata = {'fs': fs, 'dx': dx, 'start_channel': ch1,
                'start_distance': start_distance, 'start_time': start_time,
                'gauge_length': gauge_length, 'headers': headers}

    return data, metadata


def _read_segy(fname, **kwargs):
    # https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb

    with segyio.open(fname, ignore_geometry=True) as segy_file:
        warnings.warn('This data format doesn\'t include channel interval. Set '
                      'dx to 1')
        nch = segy_file.tracecount
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', nch)

        # read data
        data = segy_file.trace.raw[ch1:ch2]

        # read metadata:
        fs = 1 / (segyio.tools.dt(segy_file) / 1e6)
        metadata = {'fs': fs, 'dx': 1, 'start_channel': ch1}

        return data, metadata


def read_json(fname, output_type='dict'):
    """
    Read .json metadata file. See {Lai et al. , 2024, Seismol. Res. Lett.}

    :param fname: str or pathlib.PosixPath. Path of json file.
    :param output_type: str. 'dict' means return a dictionary, and 'Section'
        means return a empty daspy.Section instance with metadata.
    :return: A dictionary of metadata or an instance of daspy.Section without
        data.
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
                nch = interrogator['Acquisition'][0]['Attributes']['number_of_channels']
                data = np.zeros((nch, 0))
                dx = interrogator['Acquisition'][0]['Attributes']['spatial_sampling_interval']
                fs = interrogator['Acquisition'][0]['Attributes']['acquisition_sample_rate']
                gauge_length = interrogator['Acquisition'][0]['Attributes']['gauge_length']
                sec.append(Section(data, dx, fs, gauge_length=gauge_length,
                                   headers=headers))
        elif len(headers['Overview']['Interrogator'][0]['Acquisition']) > 1:
            case_type = 'Active survey'
            sec_num = len(
                headers['Overview']['Interrogator'][0]['Acquisition'])
            sec = []
            for acquisition in headers['Overview']['Interrogator'][0]['Acquisition']:
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
                env = headers['Overview']['Cable'][0]['Attributes']['cable_environment']
                if env == 'trench':
                    case_type = 'Direct buried'
                elif env == 'conduit':
                    case_type = 'Dark fiber'
                elif env in ['wireline', 'outside borehole casing']:
                    case_type = 'Borehole cable'
            nch = headers['Overview']['Interrogator'][0]['Acquisition'][0]['Attributes']['number_of_channels']
            dx = headers['Overview']['Interrogator'][0]['Acquisition'][0]['Attributes']['spatial_sampling_interval']
            fs = headers['Overview']['Interrogator'][0]['Acquisition'][0]['Attributes']['acquisition_sample_rate']
            gauge_length = headers['Overview']['Interrogator'][0]['Acquisition'][0]['Attributes']['gauge_length']
            data = np.zeros((nch, 0))
            sec = Section(data, dx, fs, gauge_length=gauge_length,
                          headers=headers)

        print(f'For case of {case_type}, create {sec_num} empty daspy.Section '
              'instance(s)')
        return sec
