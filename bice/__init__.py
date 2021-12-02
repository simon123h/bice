"""
This is bice.
"""
# TODO: we need a short description of bice here

# import core namespace, so we can e.g. use bice.Problem
from .core import *

__all__ = []

# 'from bice import *' should import everything defined in core.__all__
from . import core
__all__.extend(core.__all__)
