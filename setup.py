from setuptools import setup
from setuptools import find_packages
from pkg_resources import resource_filename


# METADATA
NAME = 'pygms-cmeessen'
MODULE = 'pyGMS'
VERSION = '0.0.1'
AUTHOR = 'Christian Meeßen'
AUTHOR_EMAIL = 'christian.meessen@gfz-potsdam.de'
MAINTAINER = 'Christian Meeßen'
MAINTAINER_EMAIL = 'christian.meessen@gfz-potsdam.de'
URL = 'https://github.com/cmeessen/pyGMS'
DESCRIPTION = 'A Python module to analyse models created with the GeoModellingSystem'
try:
    with open(resource_filename(MODULE, '../README.md'), 'r') as fh:
        LONG_DESCRIPTION = fh.read()
except ImportError:
    with open('README.md') as fh:
        LONG_DESCRIPTION = fh.read()
LONG_DESCRIPTION_TYPE = 'text/markdown'
PACKAGE_DATA = find_packages()
CLASSIFIERS = [
    'Natural Language :: English',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: GNU GPL-3.0',
    'Operating System :: OS Independent',
    'Topic :: Geophysics',
]

# DEPENDENCIES
INSTALL_REQUIRES = [
    'numpy',
    'matplotlib',
    'pandas',
    'scipy'
]

if __name__ == '__main__':
    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type=LONG_DESCRIPTION_TYPE,
        url=URL,
        packages=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
    )
