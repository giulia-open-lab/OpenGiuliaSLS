Conda
================

Conda is an open-source package management system and environment management system that runs on Windows, macOS, and
Linux. Conda quickly installs, runs, and updates packages and their dependencies. Conda easily creates, saves, loads,
and switches between environments on your local computer. It was created for Python programs, but it can package and
distribute software for any language.

We use Conda in order to make sure you are using the correct Python version, and that all the required dependencies are
installed, and using the correct versions.

We recommend using a tool called MiniConda, which is a minimal installer for Conda.

1. Install Miniconda
--------------------

To install the latest version of miniconda, follow the instructions here: https://www.anaconda.com/docs/getting-started/miniconda/main

Once you reach this point you have two options.
You can either use the environment we provide, or create a new one.

2a. Creating a Conda environment and installing dependencies
------------------------------------------------------------

2a.1 Create a new environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a new environment, run the following command:

.. code-block:: shell

    conda create --name giulia python=3.11

.. important::

    Python **3.11 or newer** is required: ``pyproject.toml`` sets
    ``requires-python = ">=3.11"``, and Sionna 2 needs it too. The validated reference
    environment is Python 3.11.

2a.2. Activate the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To activate the environment, run the following command:

.. code-block:: shell

    conda activate giulia

.. tip::

    To make sure you have activated the environment, you can check the command prompt. If the environment is activated,
    you should see the environment name in the command prompt (``giulia``).

.. note::

    You can create multiple environments depending on what you want to do with the project.
    For example, if you only want to create documentation, create an environment called ``giulia-doc``.
    Or for development, ``giulia-dev``.

2a.3. Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

The dependency list is kept in the installer script, not in a ``requirements.txt``. With the
environment activated, run:

.. code-block:: shell

    bash scripts/installer_dependencies.sh

This installs the same pinned scientific stack that ``scripts/installer.sh`` would, into the
environment that is currently active. See :doc:`environment-setup` for the full list of what gets
installed and for the flags ``scripts/installer.sh`` accepts.

.. hint::

    You must be inside the project's root directory to run this command.

.. note::

    Even though we are running ``pip``, this only installs into the currently active Conda
    environment.

.. important::

    Do **not** install PyTorch yourself, and do not install a CUDA toolkit through Conda.
    Sionna 2 is PyTorch-based and pulls in both PyTorch and its bundled CUDA runtime
    automatically; installing them separately overwrites the build Sionna resolved and breaks
    the environment.

2a.4. Export the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to export the environment to use it in another machine, you can run the following command:

.. code-block:: shell

    conda env export --name giulia > environment.yml

.. note::

    ``environment.yml`` lives at the **repository root** and is the full bit-for-bit snapshot of
    the validated runtime environment. The ``env/`` folder holds only the auxiliary environments,
    currently ``env/environment-docs.yml`` for building the documentation.

2b. Importing a Conda environment
----------------------------------

2b.1. Import the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of creating an environment and installing into it, you can recreate the validated one
directly from the snapshot committed in the repository:

.. tabs::

   .. tab:: Locally (from cloned repository)

        .. code-block:: shell

            conda env create -n giulia --file environment.yml

   .. tab:: Remotely (from a URL)

        .. code-block:: shell

            conda env create -n giulia --file https://raw.githubusercontent.com/david-lopez-perez/Giulia/refs/heads/main/environment.yml

.. note::

    Two environment files are provided:

    1. ``environment.yml`` (repository **root**): the full runtime environment, pinned bit-for-bit,
       including the exact PyTorch build resolved by Sionna 2.
    2. ``env/environment-docs.yml``: the dependencies needed to build this documentation. It does
       **not** include the simulator's GPU stack.

.. tip::

    If you want to have multiple environments on your machine, suffix them with the type of
    environment you are creating. For example, ``giulia-docs`` for the documentation environment:

    .. code-block:: shell

        conda env create -n giulia-docs --file env/environment-docs.yml

2b.2. Activate the environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To activate the environment, run the following command:

.. code-block:: shell

    conda activate giulia

3. Using the environment from your IDE
--------------------------------------

You can select the Python interpreter to use in your IDE.

3.1. PyCharm
~~~~~~~~~~~~

On the bottom right corner of the IDE, you will see the current Python interpreter.
Click on it, and a dropup will appear.

.. figure:: _static/conda/pycharm-environment-1.png
    :alt: Current Python interpreter
    :align: center

    Current: Python 3.11

On the drop-up, click on ``Add New Interpreter`` and ``Add Local Interpreter``.

.. figure:: _static/conda/pycharm-environment-2.png
    :alt: Add local interpreter
    :align: center

Now, select Conda on the left side, make sure your "Conda Executable" is pointing to the correct path, and select the
environment you want to use ("Use existing environment").

.. figure:: _static/conda/pycharm-environment-3.png
    :alt: Select Conda environment
    :align: center

Finally click "Ok". The IDE will switch to that environment automatically, but you can make sure that it's correct by
looking at the bottom right corner.

.. figure:: _static/conda/pycharm-environment-4.png
    :alt: Correct Conda environment
    :align: center

3.2. Visual Studio Code
~~~~~~~~~~~~~~~~~~~~~~~

To change the Python interpreter in Visual Studio Code, you will first have to open a Python file. If you have clonned
the repository, you can open whatever file ends with ``.py``.

Once the file is open, you will see new options in the right side of the bottom bar of the IDE.

.. figure:: _static/conda/code-environment-1.png
    :alt: Current Python interpreter
    :align: center

    Current: Python 3.11

Click on the Python version, and a popup window will appear. Select the interpreter you want to use.

.. tip::

    Environments are grouped by types. So you will see all your Conda environments grouped together.

.. figure:: _static/conda/code-environment-2.png
    :alt: Select Conda environment
    :align: center

You can make sure you are using the correct environment by looking at the bottom right corner.

.. figure:: _static/conda/code-environment-3.png
    :alt: Correct Conda environment
    :align: center
