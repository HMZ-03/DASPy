# Purpose: Module for handling DASDateTime objects.
# Author: Minzhe Hu
# Date: 2024.9.25
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
            if self.tzinfo and not other.tzinfo:
                return super().__sub__(other.replace(tzinfo=self.tzinfo)).\
                    total_seconds()
            elif not self.tzinfo and other.tzinfo:
                return - (other - self)
            return super().__sub__(other).total_seconds()
        elif not isinstance(other, timedelta):
            other = timedelta(seconds=other)
        return super().__sub__(other)

    def local(self):
        return self.astimezone(tz=local_tz)

    def utc(self):
        return self.astimezone(tz=utc)

    @classmethod
    def from_datetime(cls, dt: datetime):
        return cls.fromtimestamp(dt.timestamp(), tz=dt.tzinfo)

    def to_datetime(self):
        return datetime.fromtimestamp(self.timestamp(), tz=self.tzinfo)
