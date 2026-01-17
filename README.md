# BICE

[![CI](https://github.com/simon123h/bice/actions/workflows/ci.yml/badge.svg)](https://github.com/simon123h/bice/actions/workflows/ci.yml)
![PyPI - Version](https://img.shields.io/pypi/v/bice)

A numerical path continuation software written in Python.

## Example

Example of a bifurcation diagram with snaking in the Swift-Hohenberg equation obtained with _bice_, see the [corresponding SHE demo](src/bice/demos/notebooks/she.ipynb).
<img src="src/bice/demos/SHE/sample.svg" alt="Sample bifurcation diagram with snaking" width="900"/>

## Installation

To install the latest published version from PyPI simply execute:

```bash
pip3 install bice
```

If you instead want to install the package locally, e.g. for development purposes, you may download the latest version from this git and install it using:

```bash
git clone https://gitlab.com/simon123h/bice
pip3 install -e bice/
```

in any directory of your choice.

### Requirements

The software depends on Python 3 and the following third-party packages:
`numpy`, `scipy`, `matplotlib`, `findiff`, and `numdifftools`.
All will be installed automatically when installing `bice`.

## Tutorials

We have first tutorials!

- [Heat equation tutorial](src/bice/demos/notebooks/heat_eq.ipynb): a simple tutorial on how to use bice to implement a first partial differential equation and perform a time simulation of it.
- [Swift-Hohenberg equation tutorial](src/bice/demos/notebooks/she.ipynb): a simple tutorial on how to use bice's path continuation capabilities.
- [Predator-prey model tutorial](src/bice/demos/notebooks/lve.ipynb): an introduction into the continuation of periodic orbits.

More will follow soon.

Meanwhile you can check out the less documented [demos](src/bice/demos/).

## Documentation

Click here for the
[online version of the documentation](https://simon123h.gitlab.io/bice).

### Building the documentation locally

The documentation can also be generated locally with the commands

```bash
cd doc
make html
```

The documentation can then be found in the folder `doc`.

You will need to have `Sphinx` and `sphinx_rtd_theme` installed:

```bash
pip3 install Sphinx sphinx_rtd_theme
```
