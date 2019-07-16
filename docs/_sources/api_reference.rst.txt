.. _api:

API
===

This page is a summary of the most important classes and methods of
:mod:`pyGMS`. For a full list of modules please refer to the :ref:`modindex`.

Classes
-------

The classes of pyGMS are:

.. currentmodule:: pyGMS.structures

.. autosummary::

    GMS
    Well
    Layer

Computing methods
-----------------

Thermal field
~~~~~~~~~~~~~

.. autosummary::

    GMS.compute_surface_heat_flow

Rheology
~~~~~~~~

.. autosummary::

    GMS.compute_bd_thickness
    GMS.compute_elastic_thickness
    GMS.compute_integrated_strength
    GMS.compute_yse

Visualisation
-------------

Profiles
~~~~~~~~

.. autosummary::

    GMS.plot_profile
    GMS.plot_layer_bounds
    GMS.plot_strength_profile
    GMS.plot_strength_profile
    GMS.plot_yse

Maps
~~~~

.. autosummary::

    GMS.plot_topography
    GMS.plot_surface_heat_flow

Wells
~~~~~

.. autosummary::

    Well.plot_var
    Well.plot_grad
