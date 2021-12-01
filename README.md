# BICE

A continuation software written in Python

TODO: find backronym!

## Installation

This package is not yet published on PyPI.

Download the package and install it locally using:

```bash
git clone https://zivgitlab.uni-muenster.de/s_hart20/bice
pip3 install -e bice/
```

in any directory of your choice. This will install the bice package using `setup.py`.

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
