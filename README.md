# BICE

A numerical path continuation software written in Python.

TODO: find new name!

## Example

Example of a bifurcation diagram with snaking in the Swift-Hohenberg equation obtained with _bice_, see the [corresponding SHE demo](demos/SHE/).
<img src="demos/SHE/sample.svg" alt="Sample bifurcation diagram with snaking" width="900"/>

## Installation

This package is not yet published on PyPI, therefore you need to install it locally.

Download `bice` and install it locally using:

```bash
git clone https://zivgitlab.uni-muenster.de/s_hart20/bice
pip3 install -e bice/
```

in any directory of your choice. The installation is performed using `setup.py`.

### Requirements

The software depends on Python 3 and the following third-party packages:
`numpy`, `scipy`, `matplotlib`, `findiff`, and `numdifftools`.
All will be installed automatically when installing `bice`.

## Tutorials

We have a first tutorial!

- [Heat equation tutorial](demos/notebooks/heat_eq.ipynb)

More will follow soon.

Meanwhile you can check out the less documented [demos](demos/).

## Documentation

Click here for the
[online version of the documentation](https://s_hart20.zivgitlabpages.uni-muenster.de/bice/).

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
