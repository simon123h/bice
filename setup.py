from pathlib import Path

from setuptools import find_packages, setup

cwd = Path(__file__).parent
description_file = cwd / "doc" / "pypi-description.md"
if description_file.is_file():
    long_description = description_file.read_text()
else:
    long_description = ""

setup(
    name="bice",
    version="0.3.6",
    description="Numerical continuation and bifurcation package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Simon Hartmann",
    author_email="s.hartmann@wwu.de",
    license="MIT",
    packages=find_packages(),
    install_requires=["numpy", "scipy", "matplotlib", "findiff", "numdifftools"],
)
