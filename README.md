# Overview

[![Documentation Status](https://readthedocs.org/projects/laser-generic/badge/?style=flat)](https://readthedocs.org/projects/laser-generic/)
[![GitHub Actions Build Status](https://github.com/InstituteforDiseaseModeling/laser-generic/actions/workflows/github-actions.yml/badge.svg)](https://github.com/InstituteforDiseaseModeling/laser-generic/actions)
[![Coverage Status](https://codecov.io/gh/InstituteforDiseaseModeling/laser-generic/branch/main/graphs/badge.svg?branch=main)](https://app.codecov.io/github/InstituteforDiseaseModeling/laser-generic)
[![PyPI Package latest release](https://img.shields.io/pypi/v/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![PyPI Wheel](https://img.shields.io/pypi/wheel/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![Supported versions](https://img.shields.io/pypi/pyversions/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![Supported implementations](https://img.shields.io/pypi/implementation/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![Commits since latest release](https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser-generic/v0.0.0.svg)](https://github.com/InstituteforDiseaseModeling/laser-generic/compare/v0.0.0...main)

Generic disease models implemented with the LASER framework.

* Free software: MIT license

## Installation

```sh
pip install laser-generic
```

You can also install the in-development version with:

```sh
pip install https://github.com/InstituteforDiseaseModeling/laser-generic/archive/main.zip
```

## Documentation

https://institutefordiseasemodeling.github.io/laser-generic

## Development

To run all the tests run:

```sh
tox
```

Note, to combine the coverage data from all the tox environments run:

**Windows:**
```sh
set PYTEST_ADDOPTS=--cov-append
 tox
```

**Other:**
```sh
PYTEST_ADDOPTS=--cov-append tox
```
