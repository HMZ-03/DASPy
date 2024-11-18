# Purpose: Module for handling DASDateTime objects.
# Author: Minzhe Hu
# Date: 2024.11.18
# Email: hmz2018@mail.ustc.edu.cn
import time
from typing import Iterable
from datetime import datetime, timedelta, timezone


utc = timezone.utc
local_tz = timezone(timedelta(seconds=-time.altzone))


class DASDateTime(datetime):


    def __add__(self, other):
        if isinstance(other, Iterable):
            out = []
            for t in other:
                out.append(self + t)
            return out
        elif not isinstance(other, timedelta):
            other = timedelta(seconds=float(other))
        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, Iterable):
            out = []
            for t in other:
                out.append(self - t)
            return out
        elif isinstance(other, datetime):
            return datetime.__sub__(*self._unify_tz(other)).total_seconds()
        elif not isinstance(other, timedelta):
            other = timedelta(seconds=other)
        return super().__sub__(other)

    def __le__(self, other):
        return datetime.__le__(*self._unify_tz(other))

    def __lt__(self, other):
        return datetime.__lt__(*self._unify_tz(other))

    def __ge__(self, other):
        return datetime.__ge__(*self._unify_tz(other))

    def __gt__(self, other):
        return datetime.__gt__(*self._unify_tz(other))

    def _unify_tz(self, other):
        if self.tzinfo and not other.tzinfo:
            return self, other.replace(tzinfo=self.tzinfo)
        elif not self.tzinfo and other.tzinfo:
            return self.replace(tzinfo=other.tzinfo), other
        return self, other

    def local(self):
        return self.astimezone(tz=local_tz)

    def utc(self):
        return self.astimezone(tz=utc)
    
    def remove_tz(self):
        return self.replace(tzinfo=None)

    @classmethod
    def from_datetime(cls, dt: datetime):
        return cls.fromtimestamp(dt.timestamp(), tz=dt.tzinfo)

    @classmethod
    def from_obspy_UTCDateTime(cls, dt):
        return cls.from_datetime(dt.datetime)

    def to_datetime(self):
        return datetime.fromtimestamp(self.timestamp(), tz=self.tzinfo)

    def to_obspy_UTCDateTime(self):
        from obspy import UTCDateTime
        return UTCDateTime(UTCDateTime(self.to_datetime()))