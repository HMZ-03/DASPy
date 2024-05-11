# Purpose: Module for handling Section objects.
# Author: Minzhe Hu
# Date: 2024.5.8
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
        self.timestamplist = \
            np.array([DASDateTime.strptime(f.split('/')[-1][range], format).
                      timestamp() for f in self.filelist])
        self._sort()
        if dt is None:
            time_diff = np.unique(np.diff(self.timestamplist))
            self.dt = time_diff.min()
            if len(time_diff) > 1:
                warnings.warn('File start times are unevenly spaced and self.dt'
                              ' may be incorrectly detected')
        else:
            self.dt = dt
    
    def __len__(self):
        return len(self.filelist)

    def _sort(self):
        sort = np.argsort(self.timestamplist)
        self.timestamplist = np.array(self.timestamplist)[sort]
        self.filelist = np.array(self.filelist)[sort]
        return self

    def extract(self, stime=None, etime=None, readsec=False):
        """
        Extract a period of data.

        :param stime, etime: DASDateTime. Start and end time of required data.
        :param read: bool. If True, read as a instance of daspy.Section and
            return. If False, return list of filename.
        """
        if stime is None:
            stime = self.timestamplist[0]
        else:
            stime = stime.timestamp()
        
        if etime is None:
            etime = self.timestamplist[-1] + self.dt
        else:
            etime = etime.timestamp()
        
        filelist = [self.filelist[i] for i in range(len(self))
                    if (stime - self.dt) <= self.timestamplist[i] <= etime]
        if readsec:
            sec = read(filelist[0])
            for f in filelist[1:]:
                sec += read(f)
            return sec
        else:
            return filelist