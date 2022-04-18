from setuptools import setup, find_packages

setup(name='bice',
      version='0.3.0',
      description='Numerical continuation and bifurcation package',
      author='Simon Hartmann',
      author_email='s.hartmann@wwu.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'findiff',
          'numdifftools'
      ]
      )
