========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - |github-actions| |codecov|
    * - package
      - |version| |wheel| |supported-versions| |supported-implementations| |commits-since|
.. |docs| image:: https://readthedocs.org/projects/laser-generic/badge/?style=flat
    :target: https://readthedocs.org/projects/laser-generic/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/InstituteforDiseaseModeling/laser-generic/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/InstituteforDiseaseModeling/laser-generic/actions

.. |codecov| image:: https://codecov.io/gh/InstituteforDiseaseModeling/laser-generic/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://app.codecov.io/github/InstituteforDiseaseModeling/laser-generic

.. |version| image:: https://img.shields.io/pypi/v/laser-generic.svg
    :alt: PyPI Package latest release
    :target: https://test.pypi.org/project/laser-generic

.. |wheel| image:: https://img.shields.io/pypi/wheel/laser-generic.svg
    :alt: PyPI Wheel
    :target: https://test.pypi.org/project/laser-generic

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/laser-generic.svg
    :alt: Supported versions
    :target: https://test.pypi.org/project/laser-generic

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/laser-generic.svg
    :alt: Supported implementations
    :target: https://test.pypi.org/project/laser-generic

.. |commits-since| image:: https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser-generic/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/InstituteforDiseaseModeling/laser-generic/compare/v0.0.0...main



.. end-badges

Generic disease models implented with the LASER framework.

* Free software: MIT license

Installation
============

::

    pip install laser-generic

You can also install the in-development version with::

    pip install https://github.com/InstituteforDiseaseModeling/laser-generic/archive/main.zip


Documentation
=============


https://laser-generic.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
