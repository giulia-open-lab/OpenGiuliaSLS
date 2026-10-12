import os.path

from giulia.fs.path import outputs_dir


def output_file(name: str):
    """
    Returns a file inside the outputs' directory.
    Args:
        name: The name of the file to fetch.
    Returns:
        A full path with the name desired targetting the file requested.
        It will be inside of: ``<Giulia root>/outputs/outputs/<name>``
    """
    return os.path.join(outputs_dir(), 'outputs', name)
