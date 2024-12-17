<img src="./website/USTC.svg" height="170" />&emsp;<img src="./website/DAMS.png" height="150" />

# DASPy

[![Supported Python versions](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://pypi.org/project/DASPy-toolbox/)
[![License](https://img.shields.io/pypi/l/daspy-toolbox.svg)](https://opensource.org/license/mit)
[![PyPI Version](https://img.shields.io/pypi/v/daspy-toolbox.svg)](https://pypi.org/project/DASPy-toolbox/)
[![DOI](https://img.shields.io/badge/DOI-10.1785/0220240124-blue.svg)](https://doi.org/10.1785/0220240124)
[![PyPI Downloads](https://img.shields.io/pypi/dm/daspy-toolbox.svg?label=pypi)](https://pypi.org/project/DASPy-toolbox/)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/daspy-toolbox?label=conda)](https://anaconda.org/conda-forge/daspy-toolbox)

DASPy is an open-source project dedicated to provide a python package for DAS (Distributed Acoustic Sensing) data processing.

The goal of the DASPy project is to lower the bar of DAS data processing. DASPy includes:
* Classic seismic data processing techniques, including preprocessing, filter, spectrum analysis, and visualization
* Specialized algorithms for DAS applications, including denoising, waveform decomposition, channel attribute analysis, and strain-velocity conversion. 

DASPy is licensed under the MIT License. [An English version of DASPy tutorial](https://daspy-tutorial.readthedocs.io/en/latest/), [a Chinese version of DASPy tutorial](https://daspy-tutorial-cn.readthedocs.io/zh-cn/latest/) and [an example of Jupyter notebook](document/example.ipynb) is available. If you have any questions, please contact me via <hmz2018@mail.ustc.edu.cn>.

## Installation
DASPy runs on Linux, Windows and Mac OS and on Python 3.9 and up.

### Pip
```
pip install daspy-toolbox
```

Install the latest version from GitHub:

```
pip install git+https://github.com/HMZ-03/DASPy.git
```

### Conda

```
conda install daspy-toolbox
```

or

```
conda install conda-forge::daspy-toolbox
```

### Manual installation
1. Install dependent packages: numpy, scipy >=1.13, matplotlib, geographiclib, pyproj, h5py, segyio, nptdms, tqdm

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

Please see details on how to contribute to the project [here](CONTRIBUTING.md) and [here](CodingStyleGuide.md).

### Reference

  * Minzhe Hu and Zefeng Li (2024), [DASPy: A Python Toolbox for DAS Seismology](https://pubs.geoscienceworld.org/ssa/srl/article/95/5/3055/645865/DASPy-A-Python-Toolbox-for-DAS-Seismology), *Seismological Research Letters*, 95(5), 3055–3066, doi: `https://doi.org/10.1785/0220240124`.
