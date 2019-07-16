# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys


def add_to_path():
    realpath = os.path.dirname(os.path.realpath(__file__))
    partial_path = os.path.join(realpath, '../../..')
    workspace_path = os.path.abspath(partial_path)
    assert os.path.exists(workspace_path)

    projects = []

    for current, dirs, c in os.walk(str(workspace_path)):
        for dir in dirs:

            project_path = os.path.join(workspace_path, dir, 'src')

            if os.path.exists(project_path):
                projects.append(project_path)

    for project_str in projects:
        sys.path.append(project_str)

add_to_path()

sys.path.insert(0, os.path.abspath('../../..'))

# -- Project information -----------------------------------------------------

project = 'pyGMS'
copyright = '2019, Christian Meeßen'
author = 'Christian Meeßen'

# The full version, including alpha/beta/rc tags
from pyGMS import __version__
version = __version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.githubpages',
    'sphinx.ext.todo',
    'sphinxcontrib.bibtex',
    # 'sphinx_copybutton',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'matplotlib.sphinxext.plot_directive',
    # 'nbsphinx',
    'm2r'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'nature'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True


# -- sphinx_rtd_theme --------------------------------------------------------

html_theme_options = {
    'canonical_url': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    # 'vcs_pageview_mode': '',
    # 'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}


# -- sphinx.ext.napoleon -----------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_admonition_for_examples = False
napoleon_use_ivar = True

# matplotlib directive settings
plot_include_source = True

# todo settings
todo_include_todos = True

# sphinx math settings
numfig = True
math_number_all = True


# -- copybutton --------------------------------------------------------------

def setup(app):
    """Adds copy buttons to code blocks, code location in ./_static"""
    app.add_stylesheet('copy_button.css')
    app.add_javascript('copy_button.js')
    app.add_javascript('clipboard.js')
