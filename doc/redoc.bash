#!/bin/bash

rm -r source/
sphinx-apidoc -o source ../bice
make html
