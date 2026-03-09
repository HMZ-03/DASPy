"""Top-level public API for DASPy.

DASPy focuses on Distributed Acoustic Sensing (DAS) waveform processing while
keeping a lightweight user-facing API for common workflows.
"""

from daspy.core.collection import Collection
from daspy.core.dasdatetime import DASDateTime, local_tz, utc
from daspy.core.read import read
from daspy.core.section import Section

__all__ = ["Section", "Collection", "read", "DASDateTime", "local_tz", "utc"]
__version__ = "1.2.3"
