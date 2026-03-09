"""Setup script for DASPy.

This project still uses ``setup.py`` for compatibility with existing release
workflows while metadata is centralized here.
"""

from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README = ROOT / "README.md"


setup(
    name="DASPy-toolbox",
    version="1.2.3",
    description=(
        "DASPy is an open-source Python package for "
        "Distributed Acoustic Sensing (DAS) data processing."
    ),
    long_description=README.read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Minzhe Hu, Zefeng Li",
    author_email="hmz2018@mail.ustc.edu.cn",
    maintainer="Minzhe Hu",
    maintainer_email="hmz2018@mail.ustc.edu.cn",
    license="MIT License",
    url="https://github.com/HMZ-03/DASPy",
    project_urls={
        "Documentation (EN)": "https://daspy-tutorial.readthedocs.io/en/latest/",
        "Documentation (ZH)": "https://daspy-tutorial-cn.readthedocs.io/zh-cn/latest/",
        "Source": "https://github.com/HMZ-03/DASPy",
        "Issue Tracker": "https://github.com/HMZ-03/DASPy/issues",
    },
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "daspy = daspy.main:main",
        ]
    },
    include_package_data=True,
    package_data={"daspy": ["core/example.pkl"]},
    classifiers=[
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "scipy>=1.13",
        "matplotlib",
        "geographiclib",
        "pyproj",
        "h5py",
        "segyio",
        "nptdms",
        "tqdm",
    ],
)
