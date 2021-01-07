#import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

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
    version='0.1',
    ext_modules=cythonize(extensions, language_level=3),
    packages=find_packages(),
    include_dirs=[numpy.get_include()],
)
