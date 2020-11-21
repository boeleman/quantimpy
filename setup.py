#import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

#def find_pyx(path='.'):
#    pyx_files = []
#    for root, dirs, filenames in os.walk(path):
#        for fname in filenames:
#            if fname.endswith('.pyx'):
#                pyx_files.append(os.path.join(root, fname))
#    return pyx_files

extensions = [
    Extension(
        name='quantimpy.binary',
        sources=['quantimpy/binary.pyx', 'quantimpy/binaryc.c'],
    ),
]


setup(
    name='quantimpy',
    version='0.1',
#    ext_modules=cythonize(find_pyx(), language_level=3),
    ext_modules=cythonize(extensions, language_level=3),
    packages=find_packages(),
)
