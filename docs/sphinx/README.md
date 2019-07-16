Documentation
=============

Working on the documentation
----------------------------

To generate new documentation:

```bash
make html
```

The html files will be saved in the folder `pyGMS/docs/sphinx/build/html/`.

Updating the documentation for a commit
---------------------------------------

In order to be accessed via github-pages, the documentation has to be in the
`pyGMS/docs/` folder:

```bash
make clean     # Remove all previous builds and files
make html      # Create the html pages
make gh-pages  # Move all files into the docs folder
git add ../
git commit -m "Updated docs"
```
