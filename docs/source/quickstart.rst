Quickstart
==========

If you only want to run Giulia, you can quickly get started with the following steps:

Preparation
-----------

1. Install Conda
~~~~~~~~~~~~~~~~

You can take a look at the `detailed Miniconda install steps <https://www.anaconda.com/docs/getting-started/miniconda/install>`_.

2. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    # Clone the repository
    # Tip: replace <your-org> with your organization's name
    git clone https://github.com/<your-org>/Giulia.git

    # Enter the repository
    cd Giulia

**All the following steps will assume that you are inside the repository's folder.**

.. note::

    You will need ``git`` installed in your system. If you don't have it, you can download it from `here <https://git-scm.com/downloads>`_.

3. Import and prepare the conda environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this command, and follow the instructions.

.. code-block:: shell

    ./scripts/installer.sh --install-requirements --install-dev-requirements --env-name giulia

This will create a new conda environment called ``giulia``, and install all the required dependencies.

Feel free to change the environment name to whatever you like.

.. important::

    This command will only work from inside the UPV. If you have no access to the UPV's network (physically or through
    the VPN), add ``--no-shadowing-download`` at the end of the command, and ask a UPV member to share the shadowing
    files with you.

    If you take this approach, please, place the shared shadowing files (``.mat``) into the ``shadowing`` folder of
    Giulia's root (``Giulia/shadowing`` not ``Giulia/giulia/shadowing``).

4. Git LFS (only for the tests)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The shadowing maps are not stored with Git LFS: they are downloaded by the installer of step 3 (or shared by a UPV
member), and Giulia looks for them in ``shadowing/`` or in the folder given by the ``SHADOWING_FILES`` environment
variable. `Git LFS <https://git-lfs.com/>`_ is used only for the precomputed test references under
``tests/references/`` (about 665 MB), which are needed only to run the regression tests:

.. code-block:: shell

    git lfs install
    git lfs pull

5. Done
~~~~~~~

You should be ready to run Giulia. Copy the example configuration and launch it:

.. code-block:: shell

    cp config.example.yml config.yml
    python main.py

``main.py`` reads ``config.yml`` from the repository root by default; pass a path as the first
argument to use a different one. See :doc:`running-giulia` for the configuration reference.
