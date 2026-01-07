# Purpose: Module for handling Collection objects.
# Author: Minzhe Hu
# Date: 2025.11.26
# Email: hmz2018@mail.ustc.edu.cn
import os
import warnings
import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from glob import glob
from datetime import datetime
from daspy.core.read import read
from daspy.core.dasdatetime import DASDateTime


class Collection(object):
    def __init__(self, fpath, ftype=None, file_format='auto', flength=None,
                 meta_from_file=True, timeinfo_slice=slice(None),
                 timeinfo_format=None, timeinfo_tz=None,
                 timeinfo_from_basename=True, **kwargs):
        """
        :param fpath: str or Sequence of str. File path(s) containing data.
        :param ftype: None or str. None for automatic detection, or 'pkl',
            'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy', 'npy'.
        :param file_format: Format in which the file is saved. Function is
            allowed to extract dataset and metadata.
        :type file_format: str or function
        :param flength: float. The duration of a single file in senconds.
        :param meta_from_file: bool or 'all'. False for manually set dt, dx, fs
            and gauge_length. True for extracting dt, dx, fs and gauge_length
            from first 2 file. 'all' for exracting and checking these metadata
            from all file.
        :param timeinfo_slice: slice. Slice for extracting start time from file
            name.
        :param timeinfo_format: str. Format for extracting start time from file
            name.
        :param timeinfo_tz: datetime.timezone. Time zone for extracting start
            time from file name.
        :param timeinfo_from_basename: bool. If True, timeinfo_format will use
            DASDateTime.strptime to basename of fpath.
        :param nch: int. Channel number.
        :param nsp: int. Sampling points of each file.
        :param dx: number. Channel interval in m.
        :param fs: number. Sampling rate in Hz.
        :param gauge_length: number. Gauge length in m.
        """
        if isinstance(fpath, (list, tuple)):
            self.flist = []
            for fp in fpath:
                self.flist.extend(glob(fp))
        else:
            self.flist = glob(fpath)
        if not len(self.flist):
            raise ValueError('No file input.')
        self.flist.sort()
        self.ftype = ftype
        self.file_format = file_format
        for key in ['nch', 'nsp', 'dx', 'fs', 'gauge_length']:
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
        if timeinfo_format is None and not meta_from_file:
            meta_from_file = True

        if meta_from_file == 'all':
            ftime = []
            metadata_list = []
            for f in self.flist:
                sec = read(f, ftype=ftype, file_format=file_format,
                           headonly=True)
                if not hasattr(sec, 'gauge_length'):
                    sec.gauge_length = None
                ftime.append(sec.start_time)
                metadata_list.append((sec.nch, sec.nsp, sec.dx, sec.fs,
                                      sec.gauge_length, sec.duration))

            if len(set(metadata_list)) > 1:
                warnings.warn('More than one kind of setting detected.')
            metadata = max(metadata_list, key=metadata_list.count)
            for i, key in enumerate(['nch', 'nsp', 'dx', 'fs', 'gauge_length']):
                if not hasattr(self, key):
                    setattr(self, key, metadata[i])
                if flength is None:
                    flength = metadata[-1]
            self.ftime = ftime
        elif meta_from_file:
            i = int(len(self.flist) > 1)
            sec = read(self.flist[i], ftype=ftype, file_format=file_format,
                       headonly=True)
            if timeinfo_format is None:
                if flength is None:
                    flength = sec.duration
                self.ftime = [sec.start_time + (j - i) * flength for j in
                            range(len(self))]
            if not hasattr(sec, 'gauge_length'):
                sec.gauge_length = None
            metadata = (sec.nch, sec.nsp, sec.dx, sec.fs, sec.gauge_length)
            for i, key in enumerate(['nch', 'nsp', 'dx', 'fs', 'gauge_length']):
                if not hasattr(self, key):
                    setattr(self, key, metadata[i])

        if not hasattr(self, 'ftime'):
            if timeinfo_from_basename:
                flist_use = [os.path.basename(f) for f in self.flist]
            else:
                flist_use = self.flist
            if timeinfo_tz is None:
                self.ftime = [DASDateTime.strptime(f[timeinfo_slice],
                    timeinfo_format) for f in flist_use]
            else:
                if '%z' in timeinfo_format.lower():
                    self.ftime = [DASDateTime.strptime(f[timeinfo_slice],
                        timeinfo_format).astimezone(timeinfo_tz) for f in
                        flist_use]
                else:
                    self.ftime = [DASDateTime.strptime(f[timeinfo_slice],
                        timeinfo_format).replace(tzinfo=timeinfo_tz) for f in
                        flist_use]

        self._sort()
        if flength is None:
            if len(self.flist) > 2:
                time_diff = np.round(np.diff(self.ftime[1:]).astype(float))
                flength_set, counts = np.unique(time_diff, return_counts=True)
                if len(flength_set) > 1:
                    warnings.warn('File start times are unevenly spaced. Data '
                                  'may not be continuous and self.flength may '
                                  'be incorrectly detected.')
                flength = flength_set[counts.argmax()]
            elif len(self.flist) == 2:
                flength = self.ftime[1] - self.ftime[0]
            else:
                flength = read(self.flist[0], ftype=ftype,
                               headonly=True).duration
        elif flength <= 0:
           raise ValueError('dt must > 0')
        
        self.flength = flength

    def __str__(self):
        if len(self) == 1:
            describe = f'       flist: {self.flist}\n'
        elif len(self) <= 5:
            describe = f'       flist: {len(self)} files\n' + \
                       f'              {self.flist}\n'
        else:
            describe = f'       flist: {len(self)} files\n' + \
                       f'              [{self[0]},\n' + \
                       f'               {self[1]},\n' + \
                       f'               ...,\n' + \
                       f'               {self[-1]}]\n'
            
        describe += f'       ftime: {self.start_time} to {self.end_time}\n' + \
                    f'     flength: {self.flength}\n'
        for key in ['nch', 'nsp', 'dx', 'fs', 'gauge_length']:
            if hasattr(self, key):
                long_key = key.rjust(12)
                value = getattr(self, key)
                describe += f'{long_key}: {value}\n'

        return describe

    __repr__ = __str__

    def __getitem__(self, i):
        return self.flist[i]

    def __len__(self):
        return len(self.flist)

    def _sort(self):
        sort = np.argsort(self.ftime)
        self.ftime = [self.ftime[i] for i in sort]
        self.flist = [self.flist[i] for i in sort]
        return self

    @property
    def start_time(self):
        return self.ftime[0]

    @property
    def end_time(self):
        return self.ftime[-1] + self.flength

    @property
    def duration(self):
        return self.end_time - self.start_time

    @property
    def file_size(self):
        return os.path.getsize(self[1])

    def copy(self):
        return deepcopy(self)

    def file_interruption(self, tolerance=0.5):
        time_diff = np.diff(self.ftime)
        return np.where(abs(time_diff - self.flength) > tolerance)[0]

    def select(self, start=None, end=None, readsec=False, tolerance=1,
               **kwargs):
        """
        Select a period of data.

        :param start, end: DASDateTime or int. Start and end time or index of
            required data.
        :param readsec: bool. If True, read as a instance of daspy.Section and
            return. If False, update self.flist.
        :param tolerance: float. The maximum error allowed between ftime and the
            actual file start time
        :param ftype: None, str or function. None for automatic detection, or
            str to specify a type of 'pkl', 'pickle', 'tdms', 'h5', 'hdf5',
            'segy', 'sgy', or 'npy', or a function for read data and metadata.
            Only used when readsec=True.
        :param file_format: str. The format in which the file is saved. It could
            be name of the device manufacturer (e.g. 'Silixa'), device model
            (e.g. 'OptaSense QuantX'), file format standard (s.g. 'AI4EPS'),
            organization (e.g. 'INGV') or dataset (e.g. 'FORESEE'). Only used
            when readsec=True.
        :param dtype: str. The data type of the returned data. Only used when
            readsec=True.
        :param chmin, chmax, dch: int. Channel number range and step. Only used
            when readsec=True.
        :param xmin, xmax: float. Range of distance. Only used when
            readsec=True.
        :param tmin, tmax: float or DASDateTime. Range of time. Only used when
            readsec=True.
        :param spmin, spmax: int. Sampling point range. Only used when
            readsec=True.
        """
        if start is None:
            start = 0
        if end is None:
            end = len(self.flist)
        if 'stime' in kwargs.keys():
            start = kwargs.pop('stime')
            warnings.warn('In future versions, the parameter \'stime\' will be '
                          'replaced by \'start\'.')
        if 'etime' in kwargs.keys():
            end = kwargs.pop('etime')
            warnings.warn('In future versions, the parameter \'etime\' will be '
                          'replaced by \'end\'.')

        if start is None:
            if 'tmin' in kwargs.keys():
                start = kwargs['tmin']
            else:
                start = 0
        if end is None:
            if 'tmax' in kwargs.keys():
                end = kwargs['tmax']
            else:
                end = len(self)

        if isinstance(start, datetime):
            for i, ftime in enumerate(self.ftime):
                if ftime > start - tolerance:
                    i -= 1
                    break
                elif ftime == start - tolerance:
                    break
            s = max(i, 0)
        else:
            s = int(start)

        if isinstance(end, datetime):
            for i, ftime in enumerate(self.ftime[s:]):
                if ftime >= end + tolerance:
                    i -= 1
                    break
            e = s + i + 1
        else:
            e = int(end)

        flist = self.flist[s:e]
        if len(flist) == 0:
            warnings.warn('No valid data was selected.')
            return None

        if readsec:
            kwargs.setdefault('ftype', self.ftype)
            kwargs.setdefault('file_format', self.file_format)
            tmin = start if isinstance(start, datetime) else None
            tmax = end if isinstance(end, datetime) else None
            if len(flist) == 1:
                sec = read(flist[0], tmin=tmin, tmax=tmax, **kwargs)
            else:
                sec = read(flist[0], tmin=tmin-tolerance, **kwargs)
                for f in flist[1:-1]:
                    sec += read(f, **kwargs)
                sec += read(flist[-1], tmax=tmax+tolerance, **kwargs)
                sec.trimming(tmin=tmin, tmax=tmax)
            return sec
        else:
            self.flist = flist
            self.ftime = self.ftime[s:e]
            return self

    def read(self, **kwargs):
        return self.select(readsec=True, **kwargs)

    def continuous_acquisition(self):
        index = self.file_interruption()
        index = [-1] + index.tolist() + [len(self)-1]
        coll_list = []
        for i in range(len(index) - 1):
            coll = self.copy().select(start=index[i]+1, end=index[i+1]+1)
            coll_list.append(coll)
        return coll_list

    def _optimize_for_continuity(self, operations):
        method_list = []
        kwargs_list = []
        if not isinstance(operations[0], (list, tuple)):
            operations = [operations]
        for opera in operations:
            method, kwargs = opera
            if method == 'downsampling':
                if_filter = ('tint' not in kwargs.keys() and 'fs' not in
                    kwargs.keys()) or ('lowpass_filter' in kwargs.keys() and
                    not kwargs['lowpass_filter'])
                if if_filter:
                    method_list.append('downsampling')
                    kwargs_list.append(kwargs)
                else:
                    method_list.extend(['lowpass_cheby_2', 'downsampling'])
                    kwargs['lowpass_filter'] = False
                    kwargs0 = dict(freq=self.fs/2/kwargs['tint'], zi=0)
                    kwargs_list.extend([kwargs0, kwargs])
            else:
                if method in ['taper', 'cosine_taper']:
                    kwargs.setdefault('side', 'both')
                elif method in ['bandpass', 'bandstop', 'lowpass', 'highpass',
                                'lowpass_cheby_2']:
                    kwargs.setdefault('zi', 0)

                method_list.append(method)
                kwargs_list.append(kwargs)
        return method_list, kwargs_list

    def _kwargs_initialization(self, method_list, kwargs_list):
        for j, method in enumerate(method_list):
            if method == 'time_integration':
                kwargs_list[j]['c'] = 0
            elif method == 'time_differential':
                kwargs_list[j]['prepend'] = 0
            elif method in ['bandpass', 'bandstop', 'lowpass',
                            'highpass', 'lowpass_cheby_2']:
                kwargs_list[j]['zi'] = 0

    def process(self, operations, savepath='./processed', merge=1,
                suffix='_pro', ftype=None, dtype=None, file_format='auto',
                save_operations=False, tolerance=0.5, **read_kwargs):
        """
        :param operations: list or None. Each element of operations list
            should be [str of method name, dict of kwargs]. None for read
            files related to operations in savepath.
        :param savepath: str. Path to save processed files.
        :param merge: int or str. int for merge several processed files into 1.
            'all' for merge all files.
        :param suffix: str. Suffix for processed files.
        :param ftype: None or str. File format for saving. None for automatic
            detection, or 'pkl', 'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy',
            'npy'.
        :param file_format: Format in which the file is saved.
        :type file_format: str or function
        :param dtype: str. The data type of the saved data.
        :parma save_operations: bool. If True, save the operations to
            method_list.pkl and kwargs_list.pkl in savepath.
        :param tolerance: float. Tolerance for checking continuity of data.
        :param read_kwargs: dict. Paramters for read function.
        """
        os.makedirs(savepath, exist_ok=True)
        method_file = os.path.join(savepath, 'method_list.pkl')
        kwargs_file = os.path.join(savepath, 'kwargs_list.pkl')
        if operations is None:
            if (not os.path.exists(method_file)) or \
                (not os.path.exists(kwargs_file)):
                raise ValueError('No operations input and no method_list.pkl '
                                 'and kwargs_list.pkl found in savepath.')
            with open(os.path.join(savepath, 'method_list.pkl'), 'rb') as f:
                method_list = pickle.load(f)
            with open(os.path.join(savepath, 'kwargs_list.pkl'), 'rb') as f:
                kwargs_list = pickle.load(f)
        else:
            method_list, kwargs_list = self._optimize_for_continuity(operations)
        if merge == 'all' or merge > len(self):
            merge = len(self)
        m = 0
        try:
            for i in tqdm(range(len(self))):
                f = self[i]
                if os.path.getsize(f) == 0:
                    warnings.warn(f'{f} is an empty file. Continuous data is '
                                  'interrupted here.')
                    if m > 0:
                        sec_merge.save(filepath, file_format=file_format,
                                       dtype=dtype)
                        m = 0
                    self._kwargs_initialization(method_list, kwargs_list)
                    continue
                try:
                    sec = read(f, ftype=self.ftype, **read_kwargs)
                    if sec.data.size == 0:
                        if m > 0:
                            sec_merge.save(filepath, ftype=ftype,
                                           file_format=file_format, dtype=dtype)
                            m = 0
                        self._kwargs_initialization(method_list, kwargs_list)
                        continue
                except Exception as e:
                    warnings.warn(f'Error reading {f}: {e}. Continuous data is '
                                  'interrupted here.')
                    if m > 0:
                        sec_merge.save(filepath, ftype=ftype,
                                       file_format=file_format, dtype=dtype)
                        m = 0
                    self._kwargs_initialization(method_list, kwargs_list)
                    continue
                for j, method in enumerate(method_list):
                    if method in ['taper', 'cosine_taper']:
                        if not ((i==0 and kwargs_list[j]['side'] != 'right') or
                                (i == len(self) - 1 and kwargs_list[j]['side']
                                 != 'left')):
                            continue
                    out = getattr(sec, method)(**kwargs_list[j])
                    if method == 'time_integration':
                        kwargs_list[j]['c'] = sec.data[:, -1].copy()
                    elif method == 'time_differential':
                        kwargs_list[j]['prepend'] = sec.data[:, -1].copy()
                    elif method in ['bandpass', 'bandstop', 'lowpass',
                                    'highpass', 'lowpass_cheby_2']:
                        kwargs_list[j]['zi'] = out
                
                if m == 0:
                    sec_merge = sec
                    f0, f1 = os.path.splitext(os.path.basename(f))
                    f1 = f1 if ftype is None else ftype
                    filepath = os.path.join(savepath, f0+suffix+f1)
                elif abs(sec_merge.end_time - sec.start_time) <= tolerance:
                    sec_merge += sec
                else:
                    warnings.warn(f'The start time of {f} does not correspond '
                                  'to the end time of the previous file. '
                                  'Continuous data is interrupted here.')
                    sec_merge.save(filepath, ftype=ftype,
                                   file_format=file_format, dtype=dtype)
                    sec_merge = sec
                    f0, f1 = os.path.splitext(os.path.basename(f))
                    f1 = f1 if ftype is None else ftype
                    filepath = os.path.join(savepath, f0+suffix+f1)
                    m = 0
                m += 1
                if m == merge:
                    sec_merge.save(filepath, ftype=ftype,
                                   file_format=file_format, dtype=dtype)
                    m = 0
            if m > 0:
                sec_merge.save(filepath, ftype=ftype, file_format=file_format,
                               dtype=dtype)
        except KeyboardInterrupt as e:
            with open(method_file, 'wb') as f:
                pickle.dump(method_list, f)
            with open(kwargs_file, 'wb') as f:
                pickle.dump(kwargs_list, f)
            print(f'Process interrupted. Saving method_list and kwargs_list.')
            raise e
        else:
            if save_operations:
                with open(method_file, 'wb') as f:
                    pickle.dump(method_list, f)
                with open(kwargs_file, 'wb') as f:
                    pickle.dump(kwargs_list, f)
                print(f'Operations saved to {method_file} and {kwargs_file}.')
            else:
                if os.path.exists(method_file):
                    os.remove(method_file)
                if os.path.exists(kwargs_file):
                    os.remove(kwargs_file)


# Dynamically add methods for cascade_methods
def _create_cascade_method(method_name):
    def cascade_method(self, savepath='./processed', merge=1,
                       suffix=f'_{method_name}', ftype=None, dtype=None,
                       file_format='auto', save_operations=False, tolerance=0.5,
                       **kwargs):
        """
        Automatically generated method for {method_name}.
        Applies the {method_name} operation to the data and saves the result.
        """
        operations = [[method_name, kwargs]]
        self.process(operations, savepath=savepath, merge=merge, suffix=suffix,
                     ftype=ftype, dtype=dtype, file_format=file_format,
                     save_operations=save_operations, tolerance=tolerance)
    return cascade_method


for method in ['time_integration', 'time_differential', 'downsampling',
               'bandpass', 'bandstop', 'lowpass', 'highpass',
               'lowpass_cheby_2']:
    setattr(Collection, method, _create_cascade_method(method))