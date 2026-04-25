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

Once you have this ready, you have to prepare your computer to develop with the project.

You can take a look at `our guide <conda>`_ in order to get started with the environment installation.

Then, choose one of the options below depending on your scope:

1. `Contributing to the public project <dev-public>`_

    Use this option if you want your changes to be public and used by the community.

    Your changes will be reviewed by the maintainers and, if approved, merged into the main repository.

    The code style will be checked by the CI/CD pipeline, and the maintainers will help you to fix any issues.

2. `Developing your own simulator <dev-own>`_

    Use this option if you are researching and developing your own simulator from the base we provide.

    You will be able to use the project as a base and develop your own features without the need to share them with the
    community.

.. |gh-fork| image:: _static/github/fork.png
   :height: 3ex
   :class: no-scaled-link

.. |gh-code| image:: _static/github/code.png
   :height: 3ex
   :class: no-scaled-link
