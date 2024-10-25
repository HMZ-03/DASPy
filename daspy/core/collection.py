# Purpose: Module for handling Collection objects.
# Author: Minzhe Hu
# Date: 2024.10.25
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
    def __init__(self, filepath, ftype=None, dt=None, meta_from_file=True, 
                 dx=None, fs=None, gauge_length=None,
                 timeinfo_format='%Y-%m-%dT%H:%M:%S%z',
                 timeinfo_slice=slice(None)):
        """
        :param filepath: str or Sequence of str. File path(s) containing data.
        :param ftype: None or str. None for automatic detection, or 'pkl',
            'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy', 'npy'.
        :param dt: float. The duration of a single file in senconds.
        :param meta_from_file: bool or 'all'. False for manually set dt, dx, fs
            and gauge_length. True for extracting dt, dx, fs and gauge_length
            from first 2 file. 'all' for exracting and checking these metadata
            from all file.
        :param dx: number. Channel interval in m.
        :param fs: number. Sampling rate in Hz.
        :param gauge_length: number. Gauge length in m.
        :param timeinfo_format: str. Format for extracting start time from file
            name.
        :param timeinfo_slice: slice. Slice of file name that containning time
            information.
        """
        if isinstance(filepath, (list, tuple)):
            self.filelist = []
            for fp in filepath:
                self.filelist.extend(glob(fp))
        else:
            self.filelist = glob(filepath)
        if not len(self.filelist):
            raise ValueError('No file input.')
        self.ftype = ftype
        if meta_from_file:
            time = []
            sl = slice(None) if meta_from_file == 'all' else slice(0,2)
            metadata_list = []
            for f in self.filelist[sl]:
                _, metadata = read(f, ftype=ftype, output_type='array', ch1=0,
                                   ch2=0)
                time.append(metadata.pop('start_time'))
                metadata_list.append((metadata.pop('dx', None),
                                      metadata.pop('fs', None),
                                      metadata.pop('gauge_length', None)))

            if len(set(metadata_list)) > 1:
                warnings.warn('More than one kind of setting detected.')
            metadata = max(metadata_list, key=metadata_list.count)
            self.dx = metadata[0] if dx is None else dx
            self.fs = metadata[1] if fs is None else fs
            self.gauge_length = metadata[2] if gauge_length is None else \
                gauge_length
            if len(time) == len(self.filelist):
                self.time = time
        
        if not hasattr(self, 'time'):
            self.time = []
            for f in self.filelist:
                self.time.append(DASDateTime.strptime(
                    os.path.basename(f)[timeinfo_slice], timeinfo_format))

        self._sort()
        if dt is None:
            if len(self.filelist) == 1:
                dt = read(self.filelist[0], ftype=ftype, ch1=0, ch2=1).duration
            else:
                time_diff = np.unique(np.diff(self.time))
                if len(time_diff) > 1:
                    warnings.warn('File start times are unevenly spaced and'
                                  'self.dt may be incorrectly detected')
                self.dt = time_diff.min()
        elif dt <= 0:
           raise ValueError('dt must > 0')
        
        self.dt = dt

    def __str__(self):
        if len(self) == 1:
            describe = f'    filelist: {self.filelist}\n'
        elif len(self) <= 5:
            describe = f'    filelist: {len(self)} files\n' + \
                       f'              {self.filelist}\n'
        else:
            describe = f'    filelist: {len(self)} files\n' + \
                       f'              [{self[0]}, {self[1]}, ..., {self[-1]}]\n'

        describe += f'        time: {self.start_time} to {self.end_time}\n' + \
                    f'          dt: {self.dt}\n' + \
                    f'          dx: {self.dx}\n' + \
                    f'          fs: {self.fs}\n' + \
                    f'gauge_length: {self.gauge_length}\n'

        return describe

    __repr__ = __str__

    def __getitem__(self, i):
        return self.filelist[i]

    def __len__(self):
        return len(self.filelist)

    def _sort(self):
        sort = np.argsort(self.time)
        self.time = [self.time[i] for i in sort]
        self.filelist = [self.filelist[i] for i in sort]
        return self

    @property
    def start_time(self):
        return self.time[0]

    @property
    def end_time(self):
        return self.time[-1] + self.dt

    @property
    def duration(self):
        return self.end_time - self.start_time

    def select(self, stime=None, etime=None, readsec=False, **kwargs):
        """
        Select a period of data.

        :param stime, etime: DASDateTime. Start and end time of required data.
        :param readsec: bool. If True, read as a instance of daspy.Section and
            return. If False, update self.filelist.
        :param ch1: int. The first channel required. Only works when
            readsec=True.
        :param ch2: int. The last channel required (not included). Only works
            when readsec=True.
        """
        if stime is None:
            stime = self.time[0]

        if etime is None:
            etime = self.time[-1] + self.dt

        filelist = [self.filelist[i] for i in range(len(self))
                    if (stime - self.dt) < self.time[i] <= etime]
        if readsec:
            sec = read(filelist[0], **kwargs)
            for f in filelist[1:]:
                sec += read(f, **kwargs)
            sec.trimming(tmin=stime, tmax=etime)
            return sec
        else:
            self.filelist = filelist
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
        new_filelist = []
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
                elif method == 'time_integration':
                    kwargs_list[j]['c'] = sec.data[:, -1]
                elif method == 'time_differential':
                    kwargs_list[j]['prepend'] = sec.data[:, -1]
                else:
                    out = eval(f'sec.{method}')(**kwargs_list[j])
                    if method in cascade_method:
                        kwargs_list[j]['zi'] = out
            f0, f1 = os.path.splitext(os.path.basename(f))
            if ftype is not None:
                f1 = ftype
            filepath = os.path.join(savepath, f0+suffix+f1)
            sec.save(filepath)
            new_filelist.append(filepath)

        return Collection(new_filelist, ftype=ftype, dt=self.dt)