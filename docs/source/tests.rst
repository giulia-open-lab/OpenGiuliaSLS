Regression tests
================

In order to keep Giulia running and not breaking, we have a set of regression tests that are run every time a pull request is made.
These tests are located in the ``tests`` folder.

The tests require some files that are not included in the repository, since they are too large.
They provide a way to check that the code is working as expected, and that the changes made do not break the code.

The way the tests are set up is by running Giulia, and then comparing the results to the expected results, if they are in range, the test passes.

Precomputing files
------------------

In order to precompute this values, first you have to run the tests once, they will all fail, but some files will be generated.
To run the tests execute the following command in the root of the repository:

.. code-block:: shell

    python tests/test_main_program.py

This will run all the tests available, and generate some folders in ``<giulia>/outputs/results``.
Note that tests only use the ``results-mean.npz`` files, so we have to get rid of the rest in order to save up some space.
To get rid of those files, use the following commands:

.. code-block:: shell

    cd outputs
    find . -type f ! -name 'results-mean.npz' -exec rm -f {} +

You will now have the same folder structure, but only ``results-mean.npz`` will be retained.
Now the final thing to do is to create the tests source directory, and copy those directories into it:

.. code-block:: shell

    cd ..
    mv outputs/results/ tests/regression_test_dlp/

Take into account that even after the optimizations, this folder will use ~200MB of space.
There's an open issue to optimize this: https://github.com/david-lopez-perez/Giulia/issues/35

Running tests
-------------

Now that the precomputed files are ready, you can safely run the tests using the command from before, and the results will be compared:

.. code-block:: shell

    python tests/test_main_program.py

If you don't want to run all the tests, you can use the ``TESTS`` environment variable.
When not set, all tests will run, but if set, you can specify which tests to run, separated by spaces, so for example,
if you only want to run ``3GPPTR36_777_UMa_AV_uniform`` and ``3GPPTR36_777_UMi_AV_uniform``, use:

.. code-block:: shell

    TESTS="3GPPTR36_777_UMa_AV_uniform 3GPPTR36_777_UMi_AV_uniform" python tests/test_main_program.py

Using a custom location
-----------------------

If you want to share precomputed files among multiple people, you can tell the test suite to take those files from a specific location.
This can be done using the ``PRECOMPUTED_FILES`` environment variable.

If not set, the location will be the default one (``<giulia>/tests/regression_test_dlp``), otherwise, the given one will be used.
Please, note that this path must be absolute, and must exist.

Example:

.. code-block:: shell

    PRECOMPUTED_FILES=/opt/giulia/regression_test_dlp python tests/test_main_program.py
