from setuptools import setup, find_packages


setup(
    name='DASPy-toolbox', version='1.1.1',
    description=(
        'DASPy is an open-source project dedicated to provide a python package '
        'for DAS (Distributed Acoustic Sensing) data processing, which '
        'comprises classic seismic data processing techniques and Specialized '
        'algorithms for DAS applications.'
        ),
    long_description=open('README.md').read(),
    author='Minzhe Hu, Zefeng Li',
    author_email='hmz2018@mail.ustc.edu.cn',
    maintainer='Minzhe Hu',
    maintainer_email='hmz2018@mail.ustc.edu.cn',
    license='MIT License',
    url='https://github.com/HMZ-03/DASPy',
    packages=find_packages(),
    entry_points={  
        'console_scripts': [
            'daspy = daspy.main:main',
            ]
    },
    include_package_data=True,
    package_data={
        'daspy': ['core/example.pkl']
    },
    classifiers=[
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy>=1.13',
        'matplotlib',
        'geographiclib',
        'pyproj',
        'h5py',
        'segyio',
        'nptdms',
        'tqdm'
    ]
)
