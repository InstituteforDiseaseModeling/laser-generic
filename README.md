# Overview

[![Documentation Build Status](https://github.com/laser-base/laser-generic/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/laser-base/laser-generic/actions/workflows/pages/pages-build-deployment)
[![GitHub Actions Build Status](https://github.com/laser-base/laser-generic/actions/workflows/github-actions.yml/badge.svg)](https://github.com/laser-base/laser-generic/actions)
[![Coverage Status](https://codecov.io/gh/laser-base/laser-generic/branch/main/graphs/badge.svg?branch=main)](https://app.codecov.io/github/laser-base/laser-generic)
[![PyPI Package latest release](https://img.shields.io/pypi/v/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![PyPI Wheel](https://img.shields.io/pypi/wheel/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![Supported versions](https://img.shields.io/pypi/pyversions/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![Supported implementations](https://img.shields.io/pypi/implementation/laser-generic.svg)](https://test.pypi.org/project/laser-generic)
[![Commits since latest release](https://img.shields.io/github/commits-since/InstituteforDiseaseModeling/laser-generic/v0.0.0.svg)](https://github.com/InstituteforDiseaseModeling/laser-generic/compare/v0.0.0...main)

LASER (Lightweight Agent Spatial modeling for ERadication) is a framework for building agent-based infectious disease models with an emphasis on spatial modeling and efficient computation at scale.

[`laser-generic`](https://github.com/laser-base/laser-generic) builds on top of [`laser-core`](https://github.com/laser-base/laser-core), offering a set of ready-to-use, generic disease model components (e.g., SI, SIS, SIR dynamics, births, deaths, vaccination).

* Free software: MIT license

## New model components

`laser-generic` adds additional modeling components to those developed for `laser-core`. They include:

### Infection & transmission

- ``Infection()`` / ``Infection_SIS()`` – intrahost progression for SI and SIS models.
- ``Susceptibility()`` – manages agent susceptibility.
- ``Exposure()`` – models exposed (latent) state with timers.
- ``Transmission()`` / ``TransmissionSIR()`` – interhost transmission dynamics.
- ``Infect_Agents_In_Patch()`` / ``Infect_Random_Agents()`` – stochastic infection events.

### Births & demographics

- ``Births()`` – demographic process, assigning DOB and node IDs.
- ``Births_ConstantPop()`` – keeps population constant by matching births to deaths.
- ``Births_ConstantPop_VariableBirthRate()`` – constant population but with variable crude birth rates.

### Immunization

- ``ImmunizationCampaign()`` – age-targeted, periodic campaigns.
- ``RoutineImmunization()`` – ongoing routine immunization at target ages.
- ``immunize_in_age_window()`` – helper to immunize within an age band.

### Initialization & seeding

- ``seed_infections_in_patch()`` / ``seed_infections_randomly()`` / ``seed_infections_randomly_SI()`` – seed infections at start.
- ``set_initial_susceptibility_in_patch()`` / ``set_initial_susceptibility_randomly()`` – initialize susceptibility.

### Utilities

- ``calc_capacity()`` – computes population capacity given births and ticks.
- ``calc_distances()`` – helper for spatial coupling via geocoordinates.
- ``get_default_parameters()`` – returns baseline parameters.


## Installation

```sh
pip install laser-generic
```

You can also install the in-development version with:

```sh
pip install https://github.com/InstituteforDiseaseModeling/laser-generic/archive/main.zip
```

## Documentation

https://laser-base.github.io/laser-generic/

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

-----

## Disclaimer

The code in this repository was developed by IDM and other collaborators to support our joint research on flexible agent-based modeling. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as permitted under the MIT License.
