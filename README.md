# LASER Generic

Generic disease models built with the LASER framework.

- SI/SIS
- SIR/SIRS
- SEIR/SEIRS

## Installation

Currently:

```bash
pip install git+https://github.com/InstituteforDiseaseModeling/laser-generic.git@v0.0.0
```

Once we have published on PyPI:

```bash
pip install laser-generic
```


## Running in Codespace

Check the installed version(s) of NumPy
```bash
pip list | grep numpy
```
Note if there is a 2.x version of NumPy installed. If so, uninstall it (sub your 2.x version for `2.1.1` below).
```bash
pip uninstall numpy==2.1.1
```
Install `laser-generic` in development mode:
```bash
pip install -e .
```