# Purpose: Module for reading DAS data.
# Author: Minzhe Hu
# Date: 2024.11.20
# Email: hmz2018@mail.ustc.edu.cn
# Partially modified from
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
from daspy.core.dasdatetime import DASDateTime, utc


def read(fname=None, output_type='section', ftype=None, headonly=False,
         **kwargs):
    """
    Read a .pkl/.pickle, .tdms, .h5/.hdf5, .segy/.sgy file.

    :param fname: str or pathlib.PosixPath. Path of DAS data file.
    :param output_type: str. 'Section' means return an instance of
        daspy.Section, 'array' means return numpy.array for data and a
        dictionary for metadata.
    :param ftype: None, str or function. None for automatic detection, or str to
        specify a type of 'pkl', 'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy',
        or 'npy', or a function for read data and metadata.
    :param headonly. bool. If True, only metadata will be read, the returned
        data will be an array of all zeros of the same size as the original
        data.
    :param ch1: int. The first channel required.
    :param ch2: int. The last channel required (not included).
    :param dch: int. Channel step.
    :return: An instance of daspy.Section, or numpy.array for data and a
        dictionary for metadata.
    """
    fun_map = {'pkl': _read_pkl, 'tdms': _read_tdms, 'h5': _read_h5,
               'sgy': _read_segy, 'npy': _read_npy}
    if fname is None:
        fname = Path(__file__).parent / 'example.pkl'
        ftype = 'pkl'
    if ftype is None:
        ftype = str(fname).split('.')[-1].lower()

    if callable(ftype):
        try:
            data, metadata = ftype(fname, headonly=headonly, **kwargs)
        except TypeError:
            data, metadata = ftype(fname)
    else:
        for rtp in [('pickle', 'pkl'), ('hdf5', 'h5'), ('segy', 'sgy')]:
            ftype = ftype.replace(*rtp)
        data, metadata = fun_map[ftype](fname, headonly=headonly, **kwargs)

    if output_type.lower() == 'section':
        metadata['source'] = Path(fname)
        metadata['source_type'] = ftype
        return Section(data.astype(float), **metadata)
    elif output_type.lower() == 'array':
        return data, metadata


def _read_pkl(fname, headonly=False, **kwargs):
    dch = kwargs.pop('dch', 1)
    with open(fname, 'rb') as f:
        pkl_data = pickle.load(f)
        if isinstance(pkl_data, np.ndarray):
            warnings.warn('This data format doesn\'t include channel interval'
                          'and sampling rate. Please set manually')
            if headonly:
                return np.zeros_like(pkl_data), {'dx': None, 'fs': None}
            else:
                ch1 = kwargs.pop('ch1', 0)
                ch2 = kwargs.pop('ch2', len(pkl_data))
                return pkl_data[ch1:ch2:dch], {'dx': None, 'fs': None}
        elif isinstance(pkl_data, dict):
            data = pkl_data.pop('data')
            if headonly:
                data = np.zeros_like(data)
            else:
                if 'ch1' in kwargs.keys() or 'ch2' in kwargs.keys():
                    if 'start_channel' in pkl_data.keys():
                        s_chn = pkl_data['start_channel']
                        print(f'Data is start with channel {s_chn}.')
                    else:
                        s_chn = 0
                    ch1 = kwargs.pop('ch1', s_chn)
                    ch2 = kwargs.pop('ch2', s_chn + len(data))
                    data = data[ch1 - s_chn:ch2 - s_chn, :]
                    pkl_data['start_channel'] = ch1
            return data, pkl_data
        else:
            raise TypeError('Unknown data type.')


def _read_h5_headers(group):
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


def _read_h5_starttime(h5_file):
    try:
        stime = h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime']
    except KeyError:
        try:
            stime = h5_file['Acquisition'].attrs['MeasurementStartTime']
        except KeyError:
            try:
                stime = h5_file['Acquisition/Raw[0]/RawDataTime/'][0]
            except KeyError:
                return 0
    if isinstance(stime, bytes):
        stime = stime.decode('ascii')

    if isinstance(stime, str):
        if len(stime) > 26:
            stime = DASDateTime.strptime(stime, '%Y-%m-%dT%H:%M:%S.%f%z')
        else:
            stime = DASDateTime.strptime(stime, '%Y-%m-%dT%H:%M:%S.%f').\
                astimezone(utc)
    else:
        stime = DASDateTime.fromtimestamp(stime / 1e6).astimezone(utc)

    return stime


def _read_h5(fname, headonly=False, **kwargs):
    with h5py.File(fname, 'r') as h5_file:
        dch = kwargs.pop('dch', 1)
        group = list(h5_file.keys())[0]
        if len(h5_file.keys()) >= 10: # ASN/OptoDAS https://github.com/ASN-Norway/simpleDAS
            ch1 = kwargs.pop('ch1', 0)
            if h5_file['header/dimensionNames'][0] == b'time':
                nch = h5_file['data'].shape[1]
                if headonly:
                    data = np.zeros_like(h5_file['data']).T
                else:
                    ch2 = kwargs.pop('ch2', nch)
                    data = h5_file['data'][:, ch1:ch2:dch].T
            elif h5_file['header/dimensionNames'][0] == b'distance':
                nch = h5_file['data'].shape[1]
                if headonly:
                    data = np.zeros_like(h5_file['data'])
                else:
                    ch2 = kwargs.pop('ch2', nch)
                    data = h5_file['data'][ch1:ch2:dch, :]
            dx = h5_file['header/dx'][()]
            start_time = DASDateTime.fromtimestamp(
                h5_file['header/time'][()]).utc()
            metadata = {'dx': dx * dch, 'fs': 1 / h5_file['header/dt'][()],
                        'start_time': start_time, 'start_channel': ch1,
                        'start_distance': ch1 * dx,
                        'scale': h5_file['header/dataScale'][()]}
            if h5_file['header/gaugeLength'][()] != np.nan:
                metadata['guage_length'] = h5_file['header/gaugeLength'][()]
        elif len(h5_file.keys()) == 5: # AP Sensing
            # read data
            nch = h5_file['strain'].shape[1]
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            if headonly:
                data = np.zeros_like(h5_file['strain']).T
            else:
                data = h5_file['strain'][:, ch1:ch2:dch].T

            # read metadata
            dx = h5_file['spatialsampling'][()]
            metadata = {'fs': h5_file['RepetitionFrequency'][()],
                        'dx': dx * dch, 'start_channel': ch1,
                        'start_distance': ch1 * dx,
                        'gauge_length': h5_file.get('GaugeLength')[()]}
        elif set(h5_file.keys()) == {'Mapping', 'Acquisition'}: # Silixa/iDAS
            nch = h5_file['Acquisition/Raw[0]'].attrs['NumberOfLoci']
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            if h5_file['Acquisition/Raw[0]/RawData/'].shape[0] == nch:
                if headonly:
                    data = np.zeros_like(h5_file['Acquisition/Raw[0]/RawData/'])
                else:
                    data = h5_file['Acquisition/Raw[0]/RawData/']\
                        [ch1:ch2:dch, :]
            else:
                if headonly:
                    data = np.zeros_like(
                        h5_file['Acquisition/Raw[0]/RawData/']).T
                else:
                    data = h5_file['Acquisition/Raw[0]/RawData/']\
                        [:, ch1:ch2:dch].T

            dx = np.mean(h5_file['Mapping/MeasuredSpatialResolution'])
            start_distance = h5_file['Acquisition/Custom/UserSettings'].\
                    attrs['StartDistance'] + ch1 * dx
            h5_file['Acquisition/Raw[0]/RawData'].attrs['PartStartTime']
            fs = h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate']
            gauge_length = h5_file['Acquisition'].attrs['GaugeLength']
            scale = h5_file['Acquisition/Raw[0]'].attrs['AmpScaling']
            geometry = np.vstack((h5_file['Mapping/Lon'],
                                  h5_file['Mapping/Lat'])).T
            metadata = {'dx': dx * dch, 'fs': fs, 'start_channel': ch1,
                        'start_distance': ch1 * dx,
                        'gauge_length': gauge_length, 'geometry': geometry,
                        'scale': scale}
            metadata['start_time'] = _read_h5_starttime(h5_file)
        elif group == 'Acquisition':
            # OptaSens/ODH, Silixa/iDAS, Sintela/Onyx, Smart Sensing/ZD DAS
            # read data
            try:
                nch = h5_file['Acquisition'].attrs['NumberOfLoci']
            except KeyError:
                nch = len(h5_file['Acquisition/Raw[0]/RawData/'])
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            if h5_file['Acquisition/Raw[0]/RawData/'].shape[0] == nch:
                if headonly:
                    data = np.zeros_like(h5_file['Acquisition/Raw[0]/RawData/'])
                else:
                    data = h5_file['Acquisition/Raw[0]/RawData/']\
                        [ch1:ch2:dch, :]
            else:
                if headonly:
                    data = np.zeros_like(
                        h5_file['Acquisition/Raw[0]/RawData/']).T
                else:
                    data = h5_file['Acquisition/Raw[0]/RawData/']\
                        [:, ch1:ch2:dch].T

            # read metadata
            try:
                fs = h5_file['Acquisition/Raw[0]'].attrs['OutputDataRate']
            except KeyError:
                time_arr = h5_file['Acquisition/Raw[0]/RawDataTime/']
                fs = 1 / (np.diff(time_arr).mean() / 1e6)

            dx = h5_file['Acquisition'].attrs['SpatialSamplingInterval']
            gauge_length = h5_file['Acquisition'].attrs['GaugeLength']
            metadata = {'dx': dx * dch, 'fs': fs, 'start_channel': ch1,
                        'start_distance': ch1 * dx,
                        'gauge_length': gauge_length}

            metadata['start_time'] = _read_h5_starttime(h5_file)
        elif group == 'raw':
            nch = len(h5_file['raw'])
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            if headonly:
                data = np.zeros_like(h5_file['raw'])
            else:
                data = h5_file['raw'][ch1:ch2:dch, :]
            fs = round(1 / np.diff(h5_file['timestamp']).mean())
            start_time = DASDateTime.fromtimestamp(
                h5_file['timestamp'][0]).astimezone(utc)
            warnings.warn('This data format doesn\'t include channel interval. '
                          'Please set manually')
            metadata = {'dx': None, 'fs': fs, 'start_channel': ch1,
                        'start_time': start_time}
        elif group == 'data': # https://ai4eps.github.io/homepage/ml4earth/seismic_event_format_das/
            nch = h5_file['data'].shape[1]
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            dch = kwargs.pop('dch', 1)
            if headonly:
                data = np.zeros_like(h5_file['raw_data'])
            else:
                data = h5_file['data'][ch1:ch2:dch, :]
            attr = h5_file['data'].attrs
            dx = attr['dx_m']
            metadata = {'dx': dx, 'fs': 1 / attr['dt_s'], 'start_channel': ch1,
                        'start_distance': ch1 * dx,
                        'start_time': DASDateTime.strptime(
                            attr['begin_time'], '%Y-%m-%dT%H:%M:%S.%f%z'),
                        'data_type': attr['unit']}
            if 'event_time' in attr.keys():
                try:
                    origin_time = DASDateTime.strptime(
                        attr['event_time'], '%Y-%m-%dT%H:%M:%S.%f%z')
                except ValueError:
                    origin_time = DASDateTime.strptime(
                        attr['event_time'], '%Y-%m-%dT%H:%M:%S.%f')
                metadata['origin_time'] = origin_time

        elif group == 'data_product':
            # read data
            nch = h5_file.attrs['nx']
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            array_shape = h5_file['data_product/data'].shape
            if array_shape[0] == nch:
                if headonly:
                    data = np.zeros_like(h5_file['data_product/data'])
                else:
                    data = h5_file['data_product/data'][ch1:ch2:dch, :]
            else:
                if headonly:
                    data = np.zeros_like(h5_file['data_product/data']).T
                else:
                    data = h5_file['data_product/data'][:, ch1:ch2:dch].T

            # read metadata
            fs = 1 / h5_file.attrs['dt_computer']
            dx = h5_file.attrs['dx']
            gauge_length = h5_file.attrs['gauge_length']
            if h5_file.attrs['saving_start_gps_time'] > 0:
                start_time = DASDateTime.fromtimestamp(
                    h5_file.attrs['file_start_gps_time'])
            else:
                start_time = DASDateTime.fromtimestamp(
                    h5_file.attrs['file_start_computer_time'])
            data_type = h5_file.attrs['data_product']

            metadata = {'dx': dx * dch, 'fs': fs, 'start_channel': ch1,
                        'start_distance': ch1 * dx,
                        'start_time': start_time.astimezone(utc),
                        'gauge_length': gauge_length, 'data_type': data_type}
        else: # Febus
            acquisition = list(h5_file[f'{group}/Source1/Zone1'].keys())[0]
            # read data
            start_channel = int(h5_file[f'{group}/Source1/Zone1'].
                                attrs['Extent'][0])
            dataset = h5_file[f'{group}/Source1/Zone1/{acquisition}']
            nch = dataset.shape[-1]
            ch1 = kwargs.pop('ch1', start_channel)
            ch2 = kwargs.pop('ch2', start_channel + nch)
            if headonly:
                data = np.zeros_like(dataset).T.reshape((nch, -1))
            else:
                if len(dataset.shape) == 3: # Febus A1-R
                    data = dataset[:, :, ch1 - start_channel:ch2 - start_channel
                                   :dch].T.reshape(((ch2 - ch1) // dch, -1))
                elif len(dataset.shape) == 2: # Febus A1
                    data = dataset[:, ch1 - start_channel:ch2 - start_channel:
                                   dch].T
            # read metadata
            attrs = h5_file[f'{group}/Source1/Zone1'].attrs
            dx = attrs['Spacing'][0]
            try:
                fs = float(attrs['FreqRes'])
            except KeyError:
                try:
                    fs = (attrs['PulseRateFreq'][0] /
                          attrs['SamplingRes'][0]) / 1000
                except KeyError:
                    fs = attrs['SamplingRate'][0]
            start_distance = attrs['Origin'][0]
            time = h5_file[f'{group}/Source1/time']
            if len(time.shape) == 2: # Febus A1-R
                start_time = DASDateTime.fromtimestamp(time[0, 0]).\
                    astimezone(utc)
            elif len(time.shape) == 1: # Febus A1
                start_time = DASDateTime.fromtimestamp(time[0]).astimezone(utc)
            gauge_length = attrs['GaugeLength'][0]
            metadata = {'dx': dx * dch, 'fs': fs, 'start_channel': ch1,
                        'start_distance': start_distance +
                                            (ch1 - start_channel) * dx,
                        'start_time': start_time, 'gauge_length': gauge_length}

        metadata['headers'] = _read_h5_headers(h5_file)

    return data, metadata


def _read_tdms(fname, headonly=False, **kwargs):
    # https://nptdms.readthedocs.io/en/stable/quickstart.html
    with TdmsFile.read(fname) as tdms_file:
        group_name = [group.name for group in tdms_file.groups()]
        if 'Measurement' in group_name:
            key = 'Measurement'
        elif 'DAS' in group_name:
            key = 'DAS'
        else:
            key = group_name[0]

        headers = {**tdms_file.properties, **tdms_file[key].properties}
        nch = len(tdms_file[key])
        dch = kwargs.pop('dch', 1)
        # read data
        if nch > 1:
            start_channel = min(int(channel.name) for channel in
                                tdms_file[key].channels())
            ch1 = max(kwargs.pop('ch1', start_channel), start_channel)
            ch2 = min(kwargs.pop('ch2', start_channel + nch),
                      start_channel + nch)
            if headonly:
                nt = len(tdms_file[key][str(start_channel)])
                data = np.zeros((nch, nt))
            else:
                data = np.asarray([tdms_file[key][str(ch)]
                                   for ch in range(ch1, ch2, dch)])
        elif nch == 1:
            try:
                start_channel = int(headers['Initial Channel'])
            except KeyError:
                start_channel = 0

            ch1 = max(kwargs.pop('ch1', start_channel), start_channel)
            nch = int(headers['Total Channels'])
            ch2 = min(kwargs.pop('ch2', start_channel + nch),
                      start_channel + nch)
            if headonly:
                data = np.zeros(len(tdms_file[key].channels()[0])).\
                    reshape((nch, -1))
            else:
                data = np.asarray(tdms_file[key].channels()[0]).\
                    reshape((-1, nch)).T
                data = data[ch1 - start_channel:ch2 - start_channel:dch]

        # read metadata
        try:
            dx = headers['SpatialResolution[m]']
        except KeyError:
            try:
                dx = headers['Spatial Resolution']
            except KeyError:
                dx = None

        try:
            fs = headers['SamplingFrequency[Hz]']
        except KeyError:
            try:
                fs = 1 / headers['Time Base']
            except KeyError:
                fs = None

        try:
            start_distance = headers['Start Distance (m)'] + \
                dx * (ch1 - start_channel)
        except KeyError:
            start_distance = dx * ch1

        try:
            start_time = DASDateTime.strptime(headers['ISO8601 Timestamp'],
                                              '%Y-%m-%dT%H:%M:%S.%f%z')
        except ValueError:
            start_time = DASDateTime.strptime(headers['ISO8601 Timestamp'],
                                              '%Y-%m-%dT%H:%M:%S.%f')
        except KeyError:
            start_time = 0
            for key in ['GPSTimeStamp', 'CPUTimeStamp', 'Trigger Time']:
                if key in headers.keys():
                    if headers[key]:
                        start_time = DASDateTime.from_datetime(headers[key].
                                                               item())
                        break

        if dx is not None:
            dx *= dch
        metadata = {'dx': dx, 'fs': fs, 'start_channel': ch1,
                    'start_distance': start_distance, 'start_time': start_time,
                    'headers': headers}

        if 'GaugeLength' in headers.keys():
            metadata['gauge_length'] = headers['GaugeLength']

    return data, metadata


def _read_segy(fname, headonly=False, **kwargs):
    # https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb
    with segyio.open(fname, ignore_geometry=True) as segy_file:
        nch = segy_file.tracecount
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', nch)
        dch = kwargs.pop('dch', 1)

        # read data
        if headonly:
            data = np.zeros_like(segy_file.trace.raw[:])
        else:
            data = segy_file.trace.raw[ch1:ch2:dch]

        # read metadata:
        fs = 1 / (segyio.tools.dt(segy_file) / 1e6)
        metadata = {'dx': None, 'fs': fs, 'start_channel': ch1}
        warnings.warn('This data format doesn\'t include channel interval.'
                      'Please set manually')

        return data, metadata


def _read_npy(fname, headonly=False, **kwargs):
    data = np.load(fname)
    if headonly:
        return np.zeros_like(data), {'dx': None, 'fs': None}
    else:
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', len(data))
        dch = kwargs.pop('dch', 1)
        warnings.warn('This data format doesn\'t include channel interval and '
                    'sampling rate. Please set manually')
        return data[ch1:ch2:dch], {'dx': None, 'fs': None}


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
