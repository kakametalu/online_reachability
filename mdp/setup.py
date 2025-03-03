from setuptools import setup
from setuptools import find_packages

setup(name='mdp',
      version='0.0.1',
      author='Kene Akametalu',
      author_email='kakametalu@berkeley.edu',
      description='Code for generating MDPs',
      install_requires=['scipy', 'numpy', 'sklearn', 'cvxopt'],
      packages=find_packages(),
      license='Hybrid Systems Lab UC Berkeley'
)
