Quickstart
==========

If you only want to run Giulia, you can quickly get started with the following steps:

Preparation
-----------

1. Install Conda
~~~~~~~~~~~~~~~~

You can take a look at detailed steps `here <https://www.anaconda.com/docs/getting-started/miniconda/install>`_.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

4. Done
~~~~~~~

You should be ready to run Giulia. You can use the following command

.. code-block:: shell

    python examples/giulia_EV.py
