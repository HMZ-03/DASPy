from daspy.basic_tools.preprocessing import _trimming_index


def _device_standardized_name(file_format: str) -> str:
    """
    Standardize device or file format name.
    """
    file_format = file_format.lower()
    file_format = file_format.replace('-', '').replace(' ', '').\
        replace('(', '').replace(')', '').replace(',', '')
    allowed_format = {
        'AP Sensing': ['apsensing', 'aps'],
        'Aragón Photonics HDAS': ['aragónphotonics', 'aragonphotonics',
                                  'aragónphotonicshdas', 'aragonphotonicshdas',
                                  'hdas'],
        'ASN OptoDAS': ['asnoptodas', 'asn', 'optodas'],
        'Febus A1-R': ['febusa1r', 'febus', 'a1r'],
        'Febus A1': ['febusa1', 'a1'],
        'OptaSense ODH3': ['optasenseodh3', 'odh3'],
        'OptaSense ODH4': ['optasenseodh4', 'odh4'],
        'OptaSense ODH4+': ['optasenseodh4+', 'odh4+', 'optasenseodh4plus',
                            'odh4plus'],
        'OptaSense QuantX': ['optasensequantx', 'quantx'],
        'Puniu Tech HiFi-DAS': ['puniutechhifidas, puniu, puniutech, hifidas',
                                'puniuhifidas', 'puniudas'],
        'Silixa iDAS': ['silixaidas', 'silixaidasv1', 'idasv1', 'idas'],
        'Silixa iDAS-v2': ['silixaidasv2', 'idasv2'],
        'Silixa iDAS-v3': ['silixaidasv3', 'idasv3'],
        'Silixa iDAS-MG': ['silixaidasmg', 'idasmg'],
        'Silixa Carina': ['silixacarina', 'carina'],
        'Sintela Onyx v1.0': ['sintelaonyxv1.0', 'sintelaonyxv1',
                              'sintalaonyx', 'sintela', 'onyxv1.0', 'onyxv1',
                              'onyx'],
        'T8 Sensor': ['t8sensor', 't8'],
        'Smart Earth ZD-DAS': ['smartearthzddas', 'smartearth', 'zddas',
                               'smartearthsensingzddas', 'smartearthsensing',
                               'zhidisensing', 'zhidi', 'zhididas'],
        'Institute of Semiconductors, CAS': ['iscas', 'cas',
                                             'instituteofsemiconductors'
                                             'instituteofsemiconductorscas'],
        'AI4EPS': ['ai4eps', 'daseventdata'],
        'INGV': ['ingv', 'istitutonazionaledigeofisicaevulcanologia'],
        'JAMSTEC': ['jamstec',
                    'japanagencyformarineearthscienceandtechnology'],
        'NEC': ['nec', 'nipponelectriccompany'],
        'FORESEE': ['forsee', 'fiberopticforenvironmentsenseing'],
        'Unknown0': ['unknown0'],
        'Unknown': ['unknown', 'other']
        }
    for standardized_format_name, allowed_name in allowed_format.items():
        if file_format in allowed_name:
            return standardized_format_name
    return 'Unknown'


def _h5_file_format(h5_file):
    """
    Detect HDF5 file format based on keys and structure.
    """
    keys = h5_file.keys()
    group = list(keys)[0]
    if set(keys) == {'Fiberlength', 'GaugeLength', 'RepetitionFrequency',
                     'spatialsampling', 'strain'}:
        file_format = 'AP Sensing'
    elif (set(keys) == {'File_Header', 'HDAS_DATA'}) or \
        (set(keys) == {'hdas_header', 'data'}):
        file_format = 'Aragón Photonics HDAS'
    elif all([key in keys for key in ['cableSpec', 'data',
        'fileGenerator', 'fileVersion', 'header', 'instrumentOptions',
        'monitoring', 'processingChain', 'timing', 'versions']]):
        file_format = 'ASN OptoDAS'
    elif (len(keys) == 1) and group.startswith('fa1-'):
        time = h5_file[f'{group}/Source1/time']
        if len(time.shape) == 2:
            file_format = 'Febus A1-R'
        elif len(time.shape) == 1:
            file_format = 'Febus A1'
    elif set(keys) == {'data', 't_axis', 'x_axis'}:
        file_format = 'OptaSense ODH3'
    elif list(keys) == ['raw_data']:
        file_format = 'OptaSense ODH4'
    elif list(keys) == ['Acquisition']:
        file_format = 'OptaSense QuantX' # 'OptaSense ODH4+'
        try:
            nch = h5_file['Acquisition'].attrs['NumberOfLoci']
            if h5_file['Acquisition/Raw[0]/RawData/'].shape[0] != nch:
                file_format = 'Silixa iDAS-MG' # 'Sintela Onyx v1.0', 'Smart Earth ZD-DAS'
        except KeyError:
            pass    
        try:
            if not isinstance(h5_file['Acquisition/Raw[0]/RawData/']
                                .attrs['PartStartTime'], bytes):
                file_format = 'Sintela Onyx v1.0'
        except KeyError:
            pass
    elif 'default' in keys:
        file_format = 'Puniu Tech HiFi-DAS'
    elif set(keys) == {'Mapping', 'Acquisition'}:
        file_format = 'Silixa iDAS'
    elif set(keys) == {'ChannelMap', 'Fiber', 'cm', 't', 'x'}:
        file_format = 'INGV'
    elif set(keys) == {'DAS_record', 'Sampling_interval_in_space',
                       'Sampling_interval_in_time', 'Sampling_points_in_space',
                       'Sampling_points_in_time'}:
        file_format = 'JAMSTEC'
    elif list(keys) == ['data']:
        if 'Interval of monitor point' in \
            list(h5_file['data'].attrs['Interval of monitor point'].keys()):
            file_format = 'NEC'
        else:
            file_format = 'AI4EPS'
    elif set(keys) == {'raw', 'timestamp'}:
        file_format = 'FORESEE'
    elif list(keys) == ['ProcessedData']:
        file_format = 'T8 Sensor'
    elif group == 'data_product':
        file_format = 'Unknown0'
    else:
        file_format = 'Unknown'
    return file_format


def _trimming_slice_metadata(shape, metadata={'dx': None, 'fs': None},
                             chmin=None, chmax=None, dch=1, xmin=None,
                             xmax=None, tmin=None, tmax=None, spmin=None,
                             spmax=None):
    """
    Calculate slicing indices and update metadata for trimming.
    """
    nch, nsp = shape
    metadata.setdefault('dx', None)
    metadata.setdefault('fs', None)
    metadata.setdefault('start_channel', 0)
    metadata.setdefault('start_distance', 0)
    metadata.setdefault('start_time', 0)
    try:
        i0, i1, j0, j1 = _trimming_index(nch, nsp, dx=metadata['dx'],
            fs=metadata['fs'], start_channel=metadata['start_channel'],
            start_distance=metadata['start_distance'],
            start_time=metadata['start_time'],
            xmin=xmin, xmax=xmax, chmin=chmin, chmax=chmax, tmin=tmin, tmax=tmax,
            spmin=spmin, spmax=spmax)
        metadata['start_channel'] += i0
        if metadata['dx'] is not None:
            metadata['start_distance'] += i0 * metadata['dx']
            metadata['dx'] *= dch
        if metadata['fs'] is not None:
            metadata['start_time'] += j0 / metadata['fs']
        return slice(i0, i1, dch), slice(j0, j1), metadata
    except ValueError:
        return slice(0, 0, 1), slice(0, 0), metadata