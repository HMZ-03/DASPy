# Purpose: Module for handling Collection objects.
# Author: Minzhe Hu
# Date: 2024.10.17
# Email: hmz2018@mail.ustc.edu.cn
import os
import warnings
import numpy as np
from tqdm import tqdm
from glob import glob
from daspy.core.read import read
from daspy.core.dasdatetime import DASDateTime
from daspy.basic_tools.preprocessing import cosine_taper


class Collection(object):
    def __init__(self, filepath, ftype=None, dt=None, check_all=False):
        """
        :param filepath: str or Sequence of str. File path(s) containing data.
        :param ftype: None or str. None for automatic detection), or 'pkl',
            'pickle', 'tdms', 'h5', 'hdf5', 'segy', 'sgy', 'npy'.
        :param dt: float. The duration of a single file in senconds.
        :param check_all: bool.
        """
        if isinstance(filepath, (list, tuple)):
            self.filelist = []
            for fp in filepath:
                self.filelist.extend(glob(fp))
        else:
            self.filelist = glob(filepath)
        self.ftype = ftype
        self.time = []
        metadata_list = []
        for f in self.filelist:
            _, metadata = read(f, ftype=ftype, output_type='array', ch1=0,
                               ch2=0)
            self.time.append(metadata.pop('start_time'))
            metadata_list.append((metadata.pop('dx', None),
                                metadata.pop('fs', None),
                                metadata.pop('gauge_length', None)))

        self._sort()
        if len(set(metadata_list)) > 1:
            warnings.warn('More than one kind of setting detected.')

        metadata = max(metadata_list, key=metadata_list.count)
        self.dx, self.fs, self.gauge_length = metadata


        if dt is None:
            time_diff = np.unique(np.diff(self.time))
            self._dt = time_diff.min()
            if len(time_diff) > 1:
                warnings.warn('File start times are unevenly spaced and self.dt'
                              ' may be incorrectly detected')
        elif dt <= 0:
           raise ValueError('dt must > 0')
        else:
            self._dt = dt

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
    def dt(self): 
        return self._dt

    @dt.setter
    def dt(self, x):
        if x <= 0:
           raise ValueError('dt must > 0')
        self._dt = x

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
    
    def process(self, operations, savepath='./processed',
                suffix='_pro', ftype=None, **read_kwargs):
        """
        :param savepath:
        :param ch1: int. The first channel required.
        :param ch2: int. The last channel required (not included).
        """
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        cascade_method = ['bandpass', 'bandstop', 'lowpass', 'highpass',
                          'lowpass_cheby_2']
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
