<img src="https://raw.githubusercontent.com/HMZ-03/DASPy/main/website/USTC.svg" height="170" />&emsp;<img src="https://raw.githubusercontent.com/HMZ-03/DASPy/main/website/DAMS.png" height="150" />


# DASPy

DASPy is an open-source project dedicated to provide a python package for DAS (Distributed Acoustic Sensing) data processing.

The goal of the daspy project is to lower the bar of DAS data processing. DASPy includes:
* Classic seismic data processing techniques, including preprocessing, filter, spectrum analysis, and visualization
* Specialized algorithms for DAS applications, including denoising, waveform decomposition, channel attribute analysis, and strain-velocity conversion. 

DASPy is licensed under the MIT License.

# Installation
DASPy is currently running on Linux, Windows and Mac OS.
DASPy runs on Python 3.8 and up. We recommend you use the latest version of python 3 if possible.

```
conda install -c hmz-03 daspy
```
or
```
conda install hmz-03::daspy
```

If an error is reported, please try updating conda:

```
conda update -n base -c conda-forge conda
```

# Getting started
```
from daspy import read
sec = read()  # load example waveform
sec.bandpass(1, 10)
sec.plot()
```
<img src="https://raw.githubusercontent.com/HMZ-03/DASPy/main/website/waveform.svg" height="250" />

# DASPy Coding Style Guide
see [PEP 8](https://peps.python.org/pep-0008/)
