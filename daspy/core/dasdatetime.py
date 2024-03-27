# Purpose: Module for handling DASDateTime objects.
# Author: Minzhe Hu
# Date: 2024.3.14
# Email: hmz2018@mail.ustc.edu.cn
from datetime import datetime, timedelta


class DASDateTime(datetime):
    def __add__(self, other):
        if not isinstance(other, timedelta):
            other = timedelta(seconds=other)

        return super().__add__(other)

    def __sub__(self, other):
        if isinstance(other, datetime):
            return (super().__sub__(other)).total_seconds()
        elif not isinstance(other, timedelta):
            other = timedelta(seconds=other)

        return super().__sub__(other)

    def convert_to_datetime(self):
        return datetime.fromtimestamp(self.timestamp())
