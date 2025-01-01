# Purpose: Module for handling Collection objects.
# Author: Minzhe Hu
# Date: 2025.1.1
# Email: hmz2018@mail.ustc.edu.cn
import os
import warnings
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from glob import glob
from daspy.core.read import read
from daspy.core.dasdatetime import DASDateTime


cascade_method = ['time_integration', 'time_differential', 'downsampling',
                  'bandpass', 'bandstop', 'lowpass', 'highpass',
                  'lowpass_cheby_2']

class Collection(object):
    def __init__(self, fpath, ftype=None, flength=None, meta_from_file=True,
                 timeinfo_format=None, timeinfo_from_basename=True, **kwargs):
        """
        :param fpath: str or Sequence of str. File path(s) containing data.
        :param ftype: None or str. None for automatic detection, or 'pkl',
            'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy', 'npy'.
        :param flength: float. The duration of a single file in senconds.
        :param meta_from_file: bool or 'all'. False for manually set dt, dx, fs
            and gauge_length. True for extracting dt, dx, fs and gauge_length
            from first 2 file. 'all' for exracting and checking these metadata
            from all file.
        :param timeinfo_format: str or (slice, str). Format for extracting start
            time from file name.
        :param timeinfo_from_basename: bool. If True, timeinfo_format will use
            DASDateTime.strptime to basename of fpath.
        :param nch: int. Channel number.
        :param nt: int. Sampling points of each file.
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
        for key in ['nch', 'nt', 'dx', 'fs', 'gauge_length']:
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
        if timeinfo_format is None and not meta_from_file:
            meta_from_file = True

        if meta_from_file == 'all':
            ftime = []
            metadata_list = []
            for f in self.flist:
                sec = read(f, ftype=ftype, headonly=True)
                if not hasattr(sec, 'gauge_length'):
                    sec.gauge_length = None
                ftime.append(sec.start_time)
                metadata_list.append((sec.nch, sec.nt, sec.dx, sec.fs,
                                      sec.gauge_length))

            if len(set(metadata_list)) > 1:
                warnings.warn('More than one kind of setting detected.')
            metadata = max(metadata_list, key=metadata_list.count)
            for i, key in enumerate(['nch', 'nt', 'dx', 'fs', 'gauge_length']):
                if not hasattr(self, key):
                    setattr(self, key, metadata[i])
            self.ftime = ftime
        elif meta_from_file:
            i = int(len(self.flist) > 1)
            sec = read(self.flist[i], ftype=ftype, headonly=True)
            if timeinfo_format is None:
                if flength is None:
                    flength = sec.duration
                self.ftime = [sec.start_time + (j - i) * flength for j in
                            range(len(self))]
            if not hasattr(sec, 'gauge_length'):
                sec.gauge_length = None
            metadata = (sec.nch, sec.nt, sec.dx, sec.fs, sec.gauge_length)
            for i, key in enumerate(['nch', 'nt', 'dx', 'fs', 'gauge_length']):
                if not hasattr(self, key):
                    setattr(self, key, metadata[i])

        if not hasattr(self, 'ftime'):
            if isinstance(timeinfo_format, tuple):
                timeinfo_slice, timeinfo_format = timeinfo_format
            else:
                timeinfo_slice = slice(None)
            if timeinfo_from_basename:
                self.ftime = [DASDateTime.strptime(
                    os.path.basename(f)[timeinfo_slice], timeinfo_format)
                    for f in self.flist]
            else:
                self.ftime = [DASDateTime.strptime(f[timeinfo_slice],
                    timeinfo_format) for f in self.flist]

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
        for key in ['nch', 'nt', 'dx', 'fs', 'gauge_length']:
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

    def select(self, stime=None, etime=None, readsec=False, **kwargs):
        """
        Select a period of data.

        :param stime, etime: DASDateTime. Start and end time of required data.
        :param readsec: bool. If True, read as a instance of daspy.Section and
            return. If False, update self.flist.
        :param ch1: int. The first channel required. Only works when
            readsec=True.
        :param ch2: int. The last channel required (not included). Only works
            when readsec=True.
        :param dch: int. Channel step. Only works when readsec=True.
        """
        if stime is None:
            stime = self.ftime[0]
        elif stime - self.ftime[0] < 0:
            warnings.warn('stime is earlier than the start time of the first '
                          'file. Set stime to self.ftime[0].')

        if etime is None:
            etime = self.ftime[-1] + self.flength
        elif etime - self.ftime[-1] > self.flength:
            warnings.warn('etime is later than the end time of the last file. '
                          'Set etime to self.ftime[-1] + self.flength.')

        if stime > etime:
            raise ValueError('Start time can\'t be later than end time.')

        flist = []
        ftime = []
        for i in range(len(self)):
            if (stime - self.flength) < self.ftime[i] < etime:
                flist.append(self.flist[i])
                ftime.append(self.ftime[i])

        if readsec:
            sec = read(flist[0], **kwargs)
            for f in flist[1:]:
                sec += read(f, **kwargs)
            sec.trimming(tmin=stime, tmax=etime)
            return sec
        else:
            coll = self.copy()
            coll.flist = flist
            coll.ftime = ftime
            return coll

    def _optimize_for_continuity(self, operations):
        method_list = []
        kwargs_list = []
        if not isinstance(operations[0], (list, tuple)):
            operations = [operations]
        for opera in operations:
            method, kwargs = opera
            if method == 'downsampling':
                if hasattr(kwargs, 'lowpass_filter') and not\
                        kwargs['lowpass_filter']:
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

    def process(self, operations, savepath='./processed', merge=1,
                suffix='_pro', ftype=None, **read_kwargs):
        """
        :param operations: list. Each element of operations list should be [str
            of method name, dict of kwargs].
        :param savepath: str. Path to save processed files.
        :param merge: int or str. int for merge several processed files into 1.
            'all' for merge all files.
        :param suffix: str. Suffix for processed files.
        :param ftype: None or str. None for automatic detection, or 'pkl',
            'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy', 'npy'.
        :param read_kwargs: dict. Paramters for read function.
        """
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        method_list, kwargs_list = self._optimize_for_continuity(operations)
        if merge == 'all' or merge > len(self):
            merge = len(self)
        for i in tqdm(range(0, len(self))):
            f = self[i]
            sec = read(f, ftype=self.ftype, **read_kwargs)
            for j, method in enumerate(method_list):
                if method in ['taper', 'cosine_taper']:
                    if not ((i==0 and kwargs_list[j] != 'right') or
                             (i == len(self) - 1 and kwargs_list[j] != 'left')):
                        continue
                out = eval(f'sec.{method}')(**kwargs_list[j])
                if method == 'time_integration':
                    kwargs_list[j]['c'] = sec.data[:, -1]
                elif method == 'time_differential':
                    kwargs_list[j]['prepend'] = sec.data[:, -1]
                elif method in ['bandpass', 'bandstop', 'lowpass', 'highpass',
                                'lowpass_cheby_2']:
                    kwargs_list[j]['zi'] = out
            
            if i % merge == 0: 
                if i != 0:
                    sec_merge.save(filepath)
                sec_merge = sec
                f0, f1 = os.path.splitext(os.path.basename(f))
                if ftype is not None:
                    f1 = ftype
                filepath = os.path.join(savepath, f0+suffix+f1)
            else:
                sec_merge += sec
        sec_merge.save(filepath)
