# pyGMS

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/194238991.svg)](https://zenodo.org/badge/latestdoi/194238991)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/50e5df33317949d58e8d7bf7c40a336b)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cmeessen/pyGMS&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/50e5df33317949d58e8d7bf7c40a336b)](https://www.codacy.com?utm_source=github.com&utm_medium=referral&utm_content=cmeessen/pyGMS&utm_campaign=Badge_Coverage)

`pyGMS` is a Python 3 module designed to analyse the rheological behaviour of
lithosphere-scale 3D structural models that were created with the
[GeoModellingSystem](https://www.gfz-potsdam.de/en/section/basin-modeling/infrastructure/gms/)
(GMS, GFZ Potsdam). `pyGMS` was originally written for the
purpose of plotting yield strength envelope cross sections for my PhD thesis.

## Installation

This is a short version of the installation instructions. For a more detailed
version visit the
[documentation](https://cmeessen.github.io/pyGMS/installation.html).

```bash
# Clone the repository
git clone git@github.com:cmeessen/pyGMS.git

# Create an Anaconda environment
cd pyGMS
conda env create -f environment.yml

# Install with pip
conda activate pygms
pip install -e .

# Install some dependencies to be able to see the kernel in Jupyter notebooks
conda install -c conda-forge nb_conda_kernels
```

## Documentation

Please have a look at the
[documentation](https://cmeessen.github.io/pyGMS/index.html) for information
on how to install and use pyGMS.

## Contributing

If you find bugs, have a feature wish or a pull request, please open an
[issue](https://github.com/cmeessen/pyGMS/issues).

### Preparing a pull request

Before preparing a pull request make sure to

- comment the code
- update CHANGELOG
- check code style (`make pycodestyle`)
- add a test if the contribution adds a new feature or fixes a bug
- update the documentation (`cd docs/sphinx && make html && make gh-pages`)
- run `make coverage` (maintainers only)
