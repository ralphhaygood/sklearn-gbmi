#! /usr/bin/env python

from setuptools import setup

VERSION = '1.0.1'

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
    packages = ['sklearn_gbmi']
)
