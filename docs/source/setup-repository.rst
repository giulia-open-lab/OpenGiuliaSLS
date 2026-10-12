Getting started
===============

In order to begin to contribute to the project, you must first set up your environment.
This includes forking the repository, cloning it, and setting up the development environment.


Repository Fork
---------------

To begin contributing to the project, you must first fork the repository.
This will create a copy of the repository in your GitHub account.

To fork the repository, navigate to the repository's GitHub page and click the |gh-fork| button in the upper right corner
of the page. This will create a copy of the repository in your GitHub account.

.. attention::

    By cloning and using the repository, you agree with the `license <https://github.com/david-lopez-perez/Giulia/blob/main/LICENSE>`_.

    Legal actions will be taken if you use the repository without agreeing with the license.

Clone the Repository
--------------------

After forking the repository, you must clone the repository to your local machine.
To do so, navigate to the repository's GitHub page and click the green |gh-code| button,
and copy the URL provided.
On a terminal in your system, run the following command:

.. code-block:: bash

    git clone <repository-url>

.. note::

    You will need to have ``git`` installed on your system to clone the repository.
    `Homepage for Git <https://git-scm.com/>`_.

Fetch the Git LFS files
-----------------------

The precomputed test references under ``tests/references/`` (about 665 MB of ``*.parquet`` files) are stored with
`Git Large File Storage (LFS) <https://git-lfs.com/>`_ instead of being committed directly. They are needed only to
run the regression tests. The shadowing maps under ``shadowing/`` are **not** stored with Git LFS: they are
downloaded separately (see the quickstart).

First, install the Git LFS extension. This is a one-time setup per machine:

.. code-block:: bash

    git lfs install

With Git LFS installed, a fresh ``git clone`` automatically downloads the LFS files for you, so no extra
step is needed.

If you cloned the repository **before** installing Git LFS (or you only see the small pointer files),
download the real content from inside the repository with:

.. code-block:: bash

    git lfs pull

.. note::

    You can check that the LFS files were downloaded correctly by running ``git lfs ls-files``,
    which lists all files tracked by Git LFS.

Once you have this ready, you have to prepare your computer to develop with the project.

You can take a look at :doc:`our guide <setup-environment>` in order to get started with the
environment installation.

Then, choose one of the options below depending on your scope:

1. :doc:`Contributing to the public project <dev-public>`

    Use this option if you want your changes to be public and used by the community.

    Your changes will be reviewed by the maintainers and, if approved, merged into the main repository.

    The code style will be checked by the CI/CD pipeline, and the maintainers will help you to fix any issues.

2. :doc:`Developing your own simulator <dev-own>`

    Use this option if you are researching and developing your own simulator from the base we provide.

    You will be able to use the project as a base and develop your own features without the need to share them with the
    community.

.. |gh-fork| image:: _static/github/fork.png
   :height: 3ex
   :class: no-scaled-link

.. |gh-code| image:: _static/github/code.png
   :height: 3ex
   :class: no-scaled-link
