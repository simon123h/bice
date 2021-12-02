from setuptools import setup

setup(name='bice',
      version='0.2',
      description='Numerical continuation and bifurcation package',
      author='Simon Hartmann',
      author_email='s.hartmann@wwu.de',
      license='MIT',
      packages=['bice'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'findiff',
          'numdifftools'
      ]
      )
