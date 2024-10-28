# Purpose: Module for handling Collection objects.
# Author: Minzhe Hu
# Date: 2024.10.28
# Email: hmz2018@mail.ustc.edu.cn
import os
import warnings
import numpy as np
from tqdm import tqdm
from glob import glob
from daspy.core.read import read
from daspy.core.dasdatetime import DASDateTime
from daspy.basic_tools.preprocessing import cosine_taper


cascade_method = ['time_integration', 'time_differential', 'downsampling',
                  'bandpass', 'bandstop', 'lowpass', 'highpass',
                  'lowpass_cheby_2']

class Collection(object):
    def __init__(self, fpath, ftype=None, flength=None, meta_from_file=True,
                 timeinfo_format=None, **kwargs):
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
        self.ftype = ftype
        for key in ['nch', 'nt', 'dx', 'fs', 'gauge_length']:
            if key in kwargs.keys():
                setattr(self, key, kwargs[key])
        if timeinfo_format is None and meta_from_file is None:
            meta_from_file = True

        if meta_from_file:
            time = []
            metadata_list = []
            for f in self.flist:
                sec = read(f, ftype=ftype, read_data=False)
                if not hasattr(sec, 'gauge_length'):
                    sec.gauge_length = None
                time.append(sec.start_time)
                metadata_list.append((sec.nch, sec.nt, sec.dx, sec.fs,
                                      sec.gauge_length))
                if meta_from_file != 'all':
                    break

            if len(set(metadata_list)) > 1:
                warnings.warn('More than one kind of setting detected.')
            metadata = max(metadata_list, key=metadata_list.count)
            for i, key in enumerate(['nch', 'nt', 'dx', 'fs', 'gauge_length']):
                if not hasattr(self, key):
                    setattr(self, key, metadata[i])
            if len(time) == len(self.flist):
                self.time = time

        if not hasattr(self, 'time'):
            if timeinfo_format is None:
                self.flist.sort()
                sec = read(self.flist[0], ftype=ftype, read_data=False)
                if flength is None:
                    flength = sec.duration
                self.time = [sec.start_time + i * flength for i in
                             range(len(self))]
            else:
                if isinstance(timeinfo_format, tuple):
                    timeinfo_slice, timeinfo_format = timeinfo_format
                else:
                    timeinfo_slice = slice(None)
                self.time = [DASDateTime.strptime(
                    os.path.basename(f)[timeinfo_slice], timeinfo_format)
                    for f in self.flist]

        self._sort()
        if flength is None:
            time_diff = np.unique(np.diff(self.time))
            if len(time_diff) > 1:
                warnings.warn('File start times are unevenly spaced and'
                                'self.flength may be incorrectly detected')
            flength = time_diff.min()
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
                       f'              [{self[0]}, {self[1]}, ..., {self[-1]}]\n'

        describe += f'        time: {self.start_time} to {self.end_time}\n' + \
                    f'     flength: {self.flength}\n' + \
                    f'         nch: {self.nch}\n' + \
                    f'          nt: {self.nt}\n' + \
                    f'          dx: {self.dx}\n' + \
                    f'          fs: {self.fs}\n' + \
                    f'gauge_length: {self.gauge_length}\n'

        return describe

    __repr__ = __str__

    def __getitem__(self, i):
        return self.flist[i]

    def __len__(self):
        return len(self.flist)

    def _sort(self):
        sort = np.argsort(self.time)
        self.time = [self.time[i] for i in sort]
        self.flist = [self.flist[i] for i in sort]
        return self

    @property
    def start_time(self):
        return self.time[0]

    @property
    def end_time(self):
        return self.time[-1] + self.flength

    @property
    def duration(self):
        return self.end_time - self.start_time

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
        """
        if stime is None:
            stime = self.time[0]

        if etime is None:
            etime = self.time[-1] + self.flength

        flist = [self.flist[i] for i in range(len(self))
                    if (stime - self.flength) < self.time[i] <= etime]
        if readsec:
            sec = read(flist[0], **kwargs)
            for f in flist[1:]:
                sec += read(f, **kwargs)
            sec.trimming(tmin=stime, tmax=etime)
            return sec
        else:
            self.flist = flist
            return self

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
                if method in cascade_method:
                    kwargs.setdefault('zi', 0)

                method_list.append(method)
                kwargs_list.append(kwargs)
        return method_list, kwargs_list

    def process(self, operations, savepath='./processed',
                suffix='_pro', ftype=None, **read_kwargs):
        """
        :param savepath:
        :param ch1: int. The first channel required.
        :param ch2: int. The last channel required (not included).
        """
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        method_list, kwargs_list = self._optimize_for_continuity(operations)
        new_flist = []
        for i in tqdm(range(len(self))):
            f = self[i]
            sec = read(f, ftype=self.ftype, **read_kwargs)
            for j, method in enumerate(method_list):
                if method == 'taper':
                    if i == 0:
                        win = cosine_taper(np.ones_like(sec.data), **kwargs_list[j])
                        win[:, -sec.nt//2:] = 1
                        sec.data *= win
                    elif i == len(self) - 1:
                        win = cosine_taper(np.ones_like(sec.data), **kwargs_list[j])
                        win[:, :sec.nt//2] = 1
                        sec.data *= win
                else:
                    out = eval(f'sec.{method}')(**kwargs_list[j])
                    if method == 'time_integration':
                        kwargs_list[j]['c'] = sec.data[:, -1]
                    elif method == 'time_differential':
                        kwargs_list[j]['prepend'] = sec.data[:, -1]
                    elif method in cascade_method:
                        kwargs_list[j]['zi'] = out
            f0, f1 = os.path.splitext(os.path.basename(f))
            if ftype is not None:
                f1 = ftype
            filepath = os.path.join(savepath, f0+suffix+f1)
            sec.save(filepath)
            new_flist.append(filepath)

        return Collection(new_flist, ftype=ftype, dt=self.flength)