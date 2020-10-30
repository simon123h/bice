#!/bin/bash

DISABLED=invalid-name,fixme,too-few-public-methods,too-many-instance-attributes,missing-function-docstring

pylint --disable=$DISABLED --output-format=colorized bice