#import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name='quantimpy.morphology',
        sources=['quantimpy/morphology.pyx', 'quantimpy/quantimpyc.c', 'quantimpy/morphologyc.c'],
    ),
    Extension(
        name='quantimpy.minkowski',
        sources=['quantimpy/minkowski.pyx', 'quantimpy/quantimpyc.c', 'quantimpy/minkowskic.c'],
    ),
]

setup(
    name='quantimpy',
    version='0.1.0',
    description='This package performs morphological operations and can compute the Minkowski functionals and functions',
    url='https://github.com/boeleman/quantimpy',
    author='Arnout Boelens',
    author_email='boelens@stanford.edu',
    install_requires=[
        'numpy',
    ],
    ext_modules=cythonize(extensions, language_level=3),
    packages=find_packages(),
    include_dirs=[np.get_include()],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Programming Language :: Cython',
    ],
)
