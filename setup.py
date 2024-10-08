from setuptools import setup

setup(
    name='agegrid',
    version = "2024.0.0",
    install_requires=[
      "numpy",
      "xarray",
      "scikit-image",
      "astropy-healpix",
      "pygmt"
    ],
)
