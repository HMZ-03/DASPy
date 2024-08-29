# Purpose: Module for handling FileSystem objects.
# Author: Minzhe Hu
# Date: 2024.8.27
# Email: hmz2018@mail.ustc.edu.cn
import warnings
import numpy as np
from daspy.core.read import read
from daspy.core.dasdatetime import DASDateTime


class FileSystem(object):
    def __init__(self, filelist, format='%Y-%m-%dT%H:%M:%S%z', dt=None,
                 range=slice(None, None)):
        """
        :param filelist: Sequence of file path or directory containing data.
        :param format: str. Format of filename to represent start time.
        :param dt: float. The duration of a single file in senconds.
        :param range: slice. Slice of the time part of the file name.
        """
        self.filelist = filelist
        self.timelist = \
            np.array([DASDateTime.strptime(f.split('/')[-1][range], format)
                      for f in self.filelist])
        self._sort()
        if dt is None:
            time_diff = np.unique(np.diff(self.timelist))
            self._dt = time_diff.min()
            if len(time_diff) > 1:
                warnings.warn('File start times are unevenly spaced and self.dt'
                              ' may be incorrectly detected')
        elif dt <= 0:
           raise ValueError('dt must > 0')
        else:
            self._dt = dt

    def __len__(self):
        return len(self.filelist)

    def _sort(self):
        sort = np.argsort(self.timelist)
        self.timelist = np.array(self.timelist)[sort]
        self.filelist = np.array(self.filelist)[sort]
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
        return self.timelist[0]

    @property
    def end_time(self): 
        return self.timelist[-1] + self.dt

    @property
    def duration(self):
        return self.end_time - self.start_time

    def extract(self, stime=None, etime=None, readsec=False, **kwargs):
        """
        Extract a period of data.

        :param stime, etime: DASDateTime. Start and end time of required data.
        :param readsec: bool. If True, read as a instance of daspy.Section and
            return. If False, return list of filename.
        :param ch1: int. The first channel required. Only works when
            readsec=True.
        :param ch2: int. The last channel required (not included). Only works
            when readsec=True.
        """
        if stime is None:
            stime = self.timelist[0]

        if etime is None:
            etime = self.timelist[-1] + self.dt

        filelist = [self.filelist[i] for i in range(len(self))
                    if (stime - self.dt) <= self.timelist[i] <= etime]
        if readsec:
            sec = read(filelist[0], **kwargs)
            for f in filelist[1:]:
                sec += read(f, **kwargs)
            sec.trimming(tmin=stime, tmax=etime)
            return sec
        else:
            return filelist