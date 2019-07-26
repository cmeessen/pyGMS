.. _installation:

Installation
============

pyGMS must be cloned from the `GitHub repository
<https://github.com/cmeessen/pyGMS>`_:

.. code-block:: bash

    git clone https://github.com/cmeessen/pyGMS.git

pyGMS can now be installed by (optionally) creating an
`Anaconda environment <#creating-an-ananconda-environment>`_, and then running
`pip <#installation-with-pip>`_.

Creating an Anaconda environment
--------------------------------

This step is optional. If you do not wish to use an isolated environment, jump
directly to `Installation with pip`_.

Installing pyGMS with Anaconda allows you to use pyGMS
inside an environment that does not alter the rest of your Python
installations. First, if you do not have an installation, download a version
`here <https://continuum.io>`_.

Now navigate to the directory where you cloned pyGMS into, and execute

.. code-block:: bash

    conda env create -f environment.yml

This will create an environment named `pygms` that includes all required
packages. Activate the environment with

.. code-block:: bash

    conda activate pygms

To be able to use the environment as a kernel in jupyter notebooks, run this
while the `pygms` environment is active

.. code-block:: bash

    conda install -c conda-forge nb_conda_kernels

The environment can be deactivated using:

.. code-block:: bash

    conda deactivate


Installation with pip
---------------------

To install pyGMS navigate to the base folder of pyGMS (containing `setup.py`)
and execute:

.. code-block:: bash

    pip install -e .

The installation with the ``-e`` flag will create a symbolic link to the
package, meaning whenever you change something within the repository (e.g. pull
an update), it will be automatically available.


Testing the installation
------------------------

In order to check whether everything works correctly, execute the tests:

.. code-block:: bash

    make test

The output should look like this:

.. code-block::

    python tests/test.py
    test_call_depth (__main__.TestGMS) ... ok
    test_call_well (__main__.TestGMS) ... ok
    test_get_xlim (__main__.TestGMS) ... ok
    test_get_ylim (__main__.TestGMS) ... ok
    test_get_zlim (__main__.TestGMS) ... ok
    test_getitem_laye_int (__main__.TestGMS) ... ok
    test_getitem_layer_str (__main__.TestGMS) ... ok
    test_getitem_layer_str_fail (__main__.TestGMS) ... ok
    test_info_df (__main__.TestGMS) ... ok
    test_call_fail (__main__.TestLayer) ... ok
    test_plot_profile (__main__.TestPlot) ... ok
    test_plot_strength_profile (__main__.TestPlot) ... ok
    test_plot_topography (__main__.TestPlot) ... ok
    test_plot_yse (__main__.TestPlot) ... ok
    test_compute_bd_thickness_comp (__main__.TestRheology) ... ok
    test_compute_elastic_thickness_comp (__main__.TestRheology) ... ok
    test_integrated_strength_comp (__main__.TestRheology) ... ok
    test_integrated_strength_ext (__main__.TestRheology) ... ok
    test_sigma_byerlee_comp (__main__.TestRheology) ... ok
    test_sigma_byerlee_ext (__main__.TestRheology) ... ok
    test_sigma_byerlee_fail (__main__.TestRheology) ... ok
    test_sigma_d_comp (__main__.TestRheology) ... ok
    test_sigma_d_ext (__main__.TestRheology) ... ok
    test_sigma_d_fail_z (__main__.TestRheology) ... ok
    test_sigma_diffusion (__main__.TestRheology) ... ok
    test_sigma_dislocation (__main__.TestRheology) ... ok
    test_sigma_dorn (__main__.TestRheology) ... ok
    test_sigma_dorn_nan_a_d (__main__.TestRheology) ... ok
    test_sigma_dorn_nan_q_d (__main__.TestRheology) ... ok
    test_sigma_dorn_zero (__main__.TestRheology) ... ok
    test_heat_flow (__main__.TestThermal) ... ok
    test_call (__main__.TestWell) ... ok
    test_getitem (__main__.TestWell) ... ok
    test_plot_grad (__main__.TestWell) ... ok

    ----------------------------------------------------------------------
    Ran 34 tests in 1.684s

    OK


Remove pyGMS
------------

To uninstall pyGMS perform the following steps:

1. If you have used pyGMS inside an Anaconda environment simply run

.. code-block:: bash

    conda env remove -n pygms

2. If you have installed it simply with pip, then run

.. code-block:: bash

    pip remove pygms-cmeessen
