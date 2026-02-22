# BICE

A numerical path continuation software for algebraic equations, ODEs, and PDEs.

Implements methods for the parameter continuation (arclength method) in the solution space of
a user-defined equation and provides a toolbox for the stability and bifurcation analysis.

In addition, interfaces to various solvers and time-stepping methods are provided.

## Installation

To install the latest published version from PyPI simply execute:

```bash
pip3 install bice
```

Alternatively, `bice` can be installed locally using the source code from our [git repository](https://gitlab.com/simon123h/bice).

### Requirements

The software depends on Python 3 and the following third-party packages:
`numpy`, `scipy`, `matplotlib`, `findiff`, and `numdifftools`.
All will be installed automatically when installing `bice`.

## Tutorials

We have first tutorials!

- [Heat equation tutorial](https://gitlab.com/simon123h/bice/-/blob/master/src/bice/demos/notebooks/heat_eq.ipynb): a simple tutorial on how to use bice to implement a first partial differential equation and perform a time simulation of it.
- [Swift-Hohenberg equation tutorial](https://gitlab.com/simon123h/bice/-/blob/master/src/bice/demos/notebooks/she.ipynb): a simple tutorial on how to use bice's path continuation capabilities.
- [Predator-prey model tutorial](https://gitlab.com/simon123h/bice/-/blob/master/src/bice/demos/notebooks/lve.ipynb): an introduction into the continuation of periodic orbits.

More will follow soon.

Meanwhile you can check out the less documented [demos](src/bice/demos/).

## Documentation

Click here for the
[online version of the documentation](https://simon123h.gitlab.io/bice).
