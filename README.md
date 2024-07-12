<img src="./website/USTC.svg" height="170" />&emsp;<img src="./website/DAMS.png" height="150" />


## DASPy

DASPy is an open-source project dedicated to provide a python package for DAS (Distributed Acoustic Sensing) data processing.

The goal of the DASPy project is to lower the bar of DAS data processing. DASPy includes:
* Classic seismic data processing techniques, including preprocessing, filter, spectrum analysis, and visualization
* Specialized algorithms for DAS applications, including denoising, waveform decomposition, channel attribute analysis, and strain-velocity conversion. 

DASPy is licensed under the MIT License. [An English version of DASPy tutorial](https://daspy-tutorial.readthedocs.io/en/latest/), [a Chinese version of DASPy tutorial](https://daspy-tutorial-cn.readthedocs.io/zh-cn/latest/) and [a preprint of DASPy paper](document/Hu_and_Li_DASPy.pdf) is available. If you have any questions, please contact me via <hmz2018@mail.ustc.edu.cn>.

## Installation
DASPy is currently running on Linux, Windows and Mac OS.
DASPy runs on Python 3.9 and up. We recommend you use the latest version of python 3 if possible.

### Pip (recommanded)
```
pip install git+https://github.com/HMZ-03/DASPy.git
```

### Conda
```
conda install -c hmz-03 daspy
```

If an error is reported, please try updating conda:

```
conda update -n base -c conda-forge conda
```

### Manual installation
1. Install dependent packages: numpy, scipy >=1.13, matplotlib, geographiclib, pyproj, h5py, segyio, nptdms

2. Add DASPy into your Python path.

## Getting started
```
from daspy import read
sec = read()  # load example waveform
sec.bandpass(1, 15)
sec.plot()
```
<img src="./website/waveform.png" height="500" />

### Contributing

Please see details on how to contribute to the project [here](CONTRIBUTING.md) and [here](CodingStyleGuide.md)

### References

  * Minzhe Hu, Zefeng Li (2010), [DASPy: A Python Toolbox for DAS Seismology](document/Hu_and_Li_DASPy.pdf), _SRL_, accepted