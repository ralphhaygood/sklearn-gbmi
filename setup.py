#! /usr/bin/env python


VERSION = '1.0.2'


import os

import numpy as np

from setuptools import Extension, setup


try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None

use_cythonize = cythonize is not None and bool(int(os.getenv('USE_CYTHONIZE', 0)))


extensions = [Extension('sklearn_gbmi._partial_dependence_tree', ['sklearn_gbmi/_partial_dependence_tree'+('.pyx' if use_cythonize else '.c')])]

if use_cythonize:
    extensions = cythonize(extensions)


setup(
    name = 'sklearn-gbmi',
    version = VERSION,
    url = "https://github.com/ralphhaygood/sklearn-gbmi",
    download_url = "https://github.com/ralphhaygood/sklearn-gbmi/tarball/{}".format(VERSION),
    author = "Ralph Haygood",
    author_email = "ralph@ralphhaygood.org",
    classifiers =
        [
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: MIT License',
            'Natural Language :: English',
            "Programming Language :: Python :: 2",
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering'
        ],
    license = 'MIT',
    description =
        "Compute Friedman and Popescu's H statistics, in order to look for interactions among variables in scikit-"+
        "learn gradient-boosting models.",
    long_description = open("README.md").read(),
    keywords =
        [
            "boosted",
            "boosting",
            "data science",
            "Friedman",
            "gradient boosted",
            "gradient boosting",
            "H statistic",
            "interaction",
            "machine learning",
            "Popescu",
            "scikit learn",
            "sklearn"
        ],
    zip_safe = True,
    install_requires = ['numpy', 'scikit-learn'],
    ext_modules = extensions,
    include_dirs = [np.get_include()],
    packages = ['sklearn_gbmi']
)
