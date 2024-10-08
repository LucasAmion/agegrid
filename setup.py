from setuptools import find_packages, setup

setup(
    name='agegrid',
    version = "2024.0.0",
    packages=find_packages()
    install_requires=[
      "numpy",
      "xarray",
      "scikit-image",
      "astropy-healpix",
      "pygmt"
    ],
)
