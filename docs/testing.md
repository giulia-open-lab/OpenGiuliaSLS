# Running Tests

Giulia is tested by running different models and distributions, and comparing them to some precomputed ones.

To get information about the tests script run:
```shell
python test_main_program.py -h
```

## Precomputed test files

Those are the *expected* value to get from the different Giulia operations. These should be provided by someone, but
if you are sure your Giulia source code is correct, you can generate them by using:
```shell
python test_main_program.py --generate
```
Note that this will disable the comparison of results, and just run the operations requested.

This will also generate a lot of big and unnecessary files for the tests, in order to clean the file structure run:
```shell
cd outputs
find . -type f ! -name 'results-.npz' -exec rm -f {} +
cd ..
```
You will now have the same folder structure, but only `results-.npz` will be retained.
Now the final thing to do is to create the tests source directory, and copy those directories into it:
```shell
cd ..
mv outputs/results/ tests/regression_test_dlp/
```

## Running tests

By default, the tests utility will run all the available tests, if you just want to run a selection of them, select the
ones you want with the `-t` option. Example:
```shell
python tests/test_main_program.py -t 3GPPTR36_777_UMa_AV_uniform 3GPPTR36_777_UMi_AV_uniform
```

## Generating reports

### Resources Usage

By default, the tests utility will monitor the system resources usage automatically. It will take measures every a set
amount of time (by default every `0.1` seconds). You can adjust this value with the `-m` option. Example:
```shell
python tests/test_main_program.py -m 0.5
```
_Will run measurements every half a second._

### Results

We provide two output options for the results of the test, either HTML or Markdown. By default, none are enabled.
Select one of them with `--html True` or `--markdown True` respectively. For example, to enable both:
```shell
python tests/test_main_program.py --html True --markdown True
```
