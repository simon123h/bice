# BICE

A continuation software written in Python

TODO: find backronym!

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

## Documentation

The documentation can be created with the commands

```bash
cd doc
sphinx-apidoc -o source ../bice
make html
```

The documentation can then be found in the folder `doc`.

You will need to have `Sphinx` and `sphinx_rtd_theme` installed:

```bash
pip3 install Sphinx sphinx_rtd_theme
```
