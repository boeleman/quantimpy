import pathlib
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

# The directory containing this file
current_dir = pathlib.Path(__file__).parent

# The text of the README file
README = (current_dir / "README.md").read_text()

extensions = [
    Extension(
        name="quantimpy.minkowski",
        sources=["quantimpy/minkowski.pyx", "quantimpy/quantimpyc.c", "quantimpy/minkowskic.c"],
    ),
]

setup(
    name="quantimpy",
    version="0.2.2",
    description="This package performs morphological operations and can compute the Minkowski functionals and functions",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/boeleman/quantimpy",
    author="Arnout Boelens",
    author_email="boelens@stanford.edu",
    install_requires=[
        "numpy",
        "edt",
    ],
    ext_modules=cythonize(extensions, language_level=3),
    packages=find_packages(exclude=("tests",)),
    include_dirs=[np.get_include()],

    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Programming Language :: Cython",
    ],
)
