# Purpose: Module for handling DASDateTime objects.
# Author: Minzhe Hu
# Date: 2024.4.15
# Email: hmz2018@mail.ustc.edu.cn
from typing import Iterable
import time
from datetime import datetime, timedelta, timezone


utc = timezone.utc


class DASDateTime(datetime):

    _tz_str = time.strftime('%z', time.localtime())
    _tz_h = int(_tz_str[:3])
    _tz_m = int(_tz_str[0] + _tz_str[3:5])
    local_tz = timezone(timedelta(hours=_tz_h, minutes=_tz_m))

    def __add__(self, other):
        if isinstance(other, Iterable):
            out = []
            for t in other:
                out.append(self + t)
            return out
        elif not isinstance(other, timedelta):
            other = timedelta(seconds=other)
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, Iterable):
            out = []
            for t in other:
                out.append(self - t)
            return out
        elif isinstance(other, datetime):
            if self.tzinfo and not other.tzinfo:
                subtracted = other.replace(tzinfo=DASDateTime.local_tz)
            elif not self.tzinfo and other.tzinfo:
                return - (other - self)
            else:
                subtracted = other
            return (super().__sub__(subtracted)).total_seconds()
        elif not isinstance(other, timedelta):
            other = timedelta(seconds=other)
        return super().__sub__(other)

    def convert_to_datetime(self):
        return datetime.fromtimestamp(self.timestamp())
