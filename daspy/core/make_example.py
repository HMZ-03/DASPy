import h5py
import numpy as np
from daspy import read, Section, DASDateTime
from daspy.core.dasdatetime import utc


origin_time = DASDateTime(2016, 3, 21, 7, 37, 10, 535000, tzinfo=utc)

# read DAS data
dx, fs = 1, 1000
sec = Section(np.zeros((8721, 0)), dx, fs, data_type='Strain rate')
filename = ['PoroTomo_iDAS16043_160321073651.h5',
            'PoroTomo_iDAS16043_160321073721.h5',
            'PoroTomo_iDAS16043_160321073751.h5',
            'PoroTomo_iDAS16043_160321073821.h5']

first = True
for fn in filename:
    with h5py.File(fn,'r') as fp:
        if first:
            sec.start_time = DASDateTime.fromtimestamp(fp['t'][0]).astimezone(utc)
            sec.start_channel = fp['channel'][0]
            first = False
        sec += fp['das'][()].T

sec.origin_time = origin_time
sec.trimming(mode=0, xmin=2500, xmax=3000)
sec.trimming(tmin=origin_time, tmax=origin_time+90)
sec.downsampling(tint=10)
sec.trimming(tmin=origin_time+20, tmax=origin_time+70)

sec.save('example.py')