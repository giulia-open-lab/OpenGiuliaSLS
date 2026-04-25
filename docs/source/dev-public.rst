Contributing to the project
===========================

Code Style
----------

All code contributions are welcome. All contributions must follow the code style conventions defined by the Python team.
Those conventions are defined in the `PEP-8`_, `PEP-257`_ and `PEP-287`_.

*In the future*, there will be automatic checks to ensure that the code is following the conventions.

To summary, the most important guidelines regarding code style are:

1. **Annotate all the types** of variables and function return types. For example:

   .. code-block:: python

      def my_function(a: int, b: str) -> float:
          return a + float(b)

   In this case, ``a`` is an integer, ``b`` is a string, and the function returns a float.

2. **Document** new functions and variables, when necessary (`PEP-257`_ and `PEP-287`_). For example:

   .. code-block:: python

      def calculate_area_square(length: int) -> int:
          """
          Calculates the area of a square of side ``length``.

          :param length: the length of the square's side
          :returns: the area of the square.
          :raises ValueError: if ``length`` is a negative number.
          """
          if length < 0:
              raise ValueError("Length must be a positive number")
          return length * length

   .. code-block:: python

      var_w_lng_nm: int = 10
      """A variable with a long and hard to understand name."""

   It's not necessary to document every single function or variable, but it is important to document the ones that are
   not self-explanatory. The documentation should be clear and concise. For example this doesn't need a comment:

   .. code-block:: python

      area_squared: int = area ** 2

3. **Use tensors instead of numpy arrays**. The project is based on PyTorch, so it is important to use tensors instead
   of numpy arrays. Even though we are still using numpy in some parts of the code, we are migrating to PyTorch tensors, so
   all new code should use tensors.


Forks and Branches
------------------

Everyone should have a fork of the repository. The fork is a copy of the repository in your GitHub account. You can
know more about how to create a fork in the `Getting Started <setup-repository.html#repository-fork>`_ guide.

**No one** should work on the ``main`` branch, not even in their fork.

To contribute to the ``main`` branch, first create a new branch with the scope of the modifications you want to make.

For example, let's say you want to update the dependencies of the project. You would have to follow those steps:

1. Create the ``update-dependencies`` branch (in your fork).
2. Make the changes locally.
3. Commit the changes using meaningful commit messages.
   As a general guideline, it's a good idea to keep the messages small, on a single line and focused on a single change.
4. Push the changes to your fork.
5. Create a pull request from your branch to the ``main`` branch of the main repository.
6. Add a meaningful description to the pull request.

After all of these steps have been followed correctly, the checks will be run automatically, and a maintainer will
review the changes. If everything is correct, the changes will be merged into the ``main`` branch, otherwise, the
changes required will be requested.

.. tip::

    Github automatically suggests creating a pull request when you push a new branch to your fork.

    .. image:: _static/github/pr-auto.png
       :alt: Pull request suggestion
       :align: center


.. _PEP-8: https://peps.python.org/pep-0008/
.. _PEP-257: https://peps.python.org/pep-0257/
.. _PEP-287: https://peps.python.org/pep-0287/
