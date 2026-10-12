import os
import shutil
import sys

from giulia.logger import debug

# Configure Giulia into path
this_dir = os.path.dirname(__file__)
root_dir = os.path.join(this_dir, "..", "..")
sys.path.insert(1, root_dir)

from giulia.fs import outputs_dir


def pack_output(output_dir: str, delete_uncompressed: bool = True):
    """
    Packs a directory into a gof file.

    Args:
        output_dir (str): The directory to pack.
        delete_uncompressed (bool, optional): If ``True``, deletes the original directory after packing. Defaults to ``True``.
    """
    # Create a ZIP file with the contents of the directory
    gof_file_path = f"{output_dir}.gof"
    debug('Packing output directory into gof file: %s', gof_file_path)
    if os.path.exists(gof_file_path):
        debug('Output gof file already exists, deleting it: %s', gof_file_path)
        os.remove(gof_file_path)
    shutil.make_archive(output_dir, 'zip', output_dir) # make_archive always uses .zip as extension
    # Rename the .zip file to .gof
    os.rename(f"{output_dir}.zip", gof_file_path)
    # And delete the original directory
    if delete_uncompressed:
        shutil.rmtree(output_dir)


def use_output(output_file: str, delete_gof: bool = True) -> str:
    """
    Unpacks a gof file.

    Args:
        output_file (str): The gof file to unpack.
        delete_gof (bool, optional): If ``True``, deletes the original gof file after extraction. Defaults to ``True``.
        
    Returns:
        str: The directory where the gof file was unpacked.
    """
    # If it's a gof file, extract it to the correct name, removing the extension
    output_directory = output_file.replace(".gof", "")
    shutil.unpack_archive(output_file, output_directory, 'zip')
    # And delete the original gof file
    if delete_gof:
        os.remove(output_file)
    return output_directory


def use_outputs(output_dir: str, delete_gof: bool = True):
    """
    Unpacks all the gof files inside a directory.

    Args:
        output_dir (str): The directory where the gof files are located in.
        delete_gof (bool, optional): If ``True``, deletes the original gof files after extraction. Defaults to ``True``.
    """
    # Obtain a list of the contents of the directory
    files = os.listdir(output_dir)
    for file in files:
        # Ignore non-gof files
        if not file.endswith(".gof"): continue
        file_path = os.path.join(output_dir, file)
        # If it's a gof file, extract it to the correct name, removing the extension
        output_directory = os.path.join(output_dir, file.split(".")[0])
        shutil.unpack_archive(file_path, output_directory, 'zip')
        # And delete the original gof file
        if delete_gof:
            os.remove(file_path)

if __name__ == "__main__":
    directory = os.path.join(outputs_dir(), 'outputs')
    # This extracts all gof files
    use_outputs(directory)
