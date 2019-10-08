Getting started
===============

This page is an overview of the features within pyGMS. Many of the methods
shown here have plenty of configuration options. Please refer to the
documentation of the individual methods to gain more information. An overview
of the most important classes and their methods is given in :ref:`api`.

For installation instructreions see :ref:`installation`. Generally, I recommend
to use pyGMS together with `Anaconda <https://continuum.io>`_ and `jupyter
<https://jupyter.org>`_.


Loading a model
---------------

.. note:: pyGMS currently only works with models that were saved in ASCII
          fromat.

.. note:: pyGMS recognises layer refinement by checking whether a layer name
          ends with `_` followed by a number, e.g. `_42`. When naming your
          layers make sure not to use names that end with such a pattern.

To get started, import the :class:`~pyGMS.structures.GMS` class of pyGMS and
load a GMS model file.

.. ipython::

    In [1]: from pyGMS import GMS

    In [3]: model = GMS('../../examples/model.fem')

You can display some information about to model using the
:attr:`~pyGMS.structures.GMS.info` property:

.. ipython::

    In [4]: model.info

To check whether it was loaded correctly, we can for example plot a profile:

.. ipython::

    @savefig plot_topography.png
    In [1]: model.plot_topography();

.. note::
    The `;` at the end of the command is only required for technical reasons
    within this documentation of pyGMS.

or list the layers:

.. ipython::

    In [1]: model.layer_dict_unique.items()

.. note::
    The layer references are stored as dictionaries that link the layer id to
    the layer name. The `layer_dict_unique` contains the primary layers,
    whereas the `layer_dict` also contains all refined layers.

Loading variables
-----------------

To accelerate pyGMS, variables are not loaded into the model when initiating
an object. The available variable names are stored in the ``field_dict`` of a
pyGMS model:

.. ipython::

    In [1]: model.field_dict.keys()

In order to add a variable, use the function
:meth:`~pyGMS.structures.GMS.layer_add_var`. In this case we add the
temperature variable ``T``:

.. ipython::

    In [1]: model.layer_add_var('T')
