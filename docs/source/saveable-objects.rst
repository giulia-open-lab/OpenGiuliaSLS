Saveable Objects
================

In some instances, you will want to extract the data calculated from a class, outside from Giulia. For example, to
analyze or visualize it. For this situations, the most simple thing to do is to use a ``Saveable`` object.

Those objects are automatically detected by Giulia, and once defined, you will be able to extract data from them.

To mark a class as Saveable, you must extend ``Saveable``. Giulia then automatically handles its registration,
and allows the user to export its data as needed.

To illustrate an example, let's say we have the following class:

.. code-block:: python

    class Example:
        some_str_output: str
        some_int_output: int

        def __init(self):
            # Initialize outputs
            some_str_output = ""
            some_int_output = 0

        def do_calculations(self):
            some_str_output = "abc"
            some_int_output = 123

As you can see, we have a class that generates two output variables: ``some_str_output`` and ``some_int_output``.

If we ever want to export them to a file after the simulation has been completed, we have to do the following updates:

.. code-block:: diff

    + from giulia.outputs.saveable import Saveable

    - class Example:
    + class Example(Saveable):
          some_str_output: str
          some_int_output: int

          def __init(self):
    +         # This line is important since it tells Giulia to actually register the Saveable
    +         super().__init__()

              # Initialize outputs
              some_str_output = ""
              some_int_output = 0

          def do_calculations(self):
              some_str_output = "abc"
              some_int_output = 123

    +     def variables_list(self) -> List[str]:
    +         return ["some_str_output", "some_int_output"]

And automatically, those variables will be available to be exported under the name of ``Example``: ``Example.some_str_output``.

Please note that the actual value of the variable will be obtained once the simulation is complete, this is, when the
save performance logic is called.

Multiple instances
------------------

This will only work for classes that are initialized once. If a class is used in multiple places on a single run, Giulia
doesn't have a way to differentiate them. For this reason, you will have to define some simple extension classes, and
use them in place of what you have.

For example, following the example above, imagine you have:

.. code-block:: python

    example_a = Example()
    example_b = Example()

To get data from both variables, you will have to define two extra classes, below ``Example``:

.. code-block:: python

    class Example(Saveable):
        ...

    class ExampleA(Example):
        pass

    class ExampleB(Example):
        pass

And then, on the definition, switch ``Example`` with ``ExampleA`` and ``ExampleB``:

.. code-block:: diff

    - example_a = Example()
    + example_a = ExampleA()
    - example_b = Example()
    + example_b = ExampleB()

By using this, you will be able to reference variables like ``ExampleA.some_str_output`` for example.
