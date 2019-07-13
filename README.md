# pyGMS

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/50e5df33317949d58e8d7bf7c40a336b)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=cmeessen/pyGMS&amp;utm_campaign=Badge_Grade)
[![Codacy Badge](https://api.codacy.com/project/badge/Coverage/50e5df33317949d58e8d7bf7c40a336b)](https://www.codacy.com?utm_source=github.com&utm_medium=referral&utm_content=cmeessen/pyGMS&utm_campaign=Badge_Coverage)

`pyGMS` is a Python 3 module designed to analyse the rheological behaviour of
lithosphere-scale 3D structural models that were created with the
[GeoModellingSystem](https://www.gfz-potsdam.de/en/section/basin-modeling/infrastructure/gms/)
(GMS, GFZ Potsdam). `pyGMS` was originally written for the
purpose of plotting yield strength envelope cross sections for my PhD thesis.

## Documentation

Please have a look at the
[documentation](https://cmeessen.github.io/pyGMS/index.html) for information
on how to install and use pyGMS.

## Contributing

If you find bugs, have a feature wish or a pull request, please open an
[issue](https://github.com/cmeessen/pyGMS/issues).

### Preparing a pull request

Before preparing a pull request make sure to

- [ ] comment the code
- [ ] update CHANGELOG
- [ ] check code style (`make pycodestyle`)
- [ ] add a test if the contribution adds a new feature or fixes a bug
- [ ] update the documentation (`cd docs/sphinx && make html && make gh-pages`)
- [ ] run `make coverage` (maintainers only)
