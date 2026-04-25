"""
@author: David López Pérez
dr.david.lopez@ieee.org 
d.lopez@iteam.upv.es

@author: Matteo Bernabe
matteo.bernabe@ieee.org
matteo.bernabe@iteam.upv.es
"""

import os
from typing import Any, Dict, List, Optional

import yaml

from giulia.inputs.run_config import RunConfig
from giulia.logger import warning
from giulia.outputs import OutputModule

class InputConfig:
    args: Dict[str, Any]
    """
    All the arguments to be passed to giulia_EV.
    """

    run_config: RunConfig
    """
    All the configuration that sets how to run Giulia.
    """

    output_compression: str
    """
    Which compression method to use for the output files.
    Options: none, zip.
    """

    rm_uncompressed: bool
    """
    If ``true``, the original output directory will be removed after completing the compression.
    Doesn't apply if ``output_compression`` is "none".
    """

    output_fields: List[str]
    """
    A list of all the fields to give on the outputs file.
    """

    def __init__(
            self,
            args: Dict[str, Any],
            run_config: RunConfig,
            output_dir: str,
            output_compression: str,
            rm_uncompressed: bool,
            output_fields: Optional[List[str]]
    ):
        """
        Builds a new instance of ``InputConfig``.

        Args:
            args: All the arguments to be passed to giulia_EV.
            output_fields: All the fields to give on the outputs file.
        """
        self.args = args
        self.run_config = run_config
        self.output_dir = output_dir if output_dir is not None else output_dir
        self.output_compression = output_compression
        self.rm_uncompressed = rm_uncompressed
        self.output_fields = output_fields if output_fields is not None and len(output_fields) > 0 else None


    @staticmethod
    def load(file_path: str):
        # Read the file
        with open(file_path, "r") as f:
            data = yaml.safe_load(f.read())
        args: Dict[str, Any] = data['args']

        output_data = data['output']
        run_config: RunConfig = RunConfig.from_config(data['run'])
        output_dir: str = output_data['directory']
        output_compression: str = output_data['compression'] if 'compression' in output_data else 'zip' # zip/none
        rm_uncompressed: bool = output_data['rm_uncompressed'] != 'false' if 'rm_uncompressed' in output_data else False # true/false, default true
        output_fields: List[str] | None = output_data['fields']

        return InputConfig(args, run_config, output_dir, output_compression, rm_uncompressed, output_fields)


    def run(self):
        # Convert the dictionary into proper command line arguments.
        # Also filter entries with null value
        args = list(f"--{k}={v}" for k, v in self.args.items() if v is not None)

        example = self.run_config.example
        if example is None:
            warning('No example provided. Defaulting to "EV"')
            example = 'EV'
        if example == 'EV':
            from examples.giulia_EV import main as giulia_main
        elif example == 'EV_batch':
            warning("Using batched run. Config arguments won't be used.")
            from examples.giulia_EV_batch import run_simulations as giulia_main
        elif example == 'EV_EE_UEassociation':
            from examples.giulia_EV_EE_UEassociation import main as giulia_main
        elif example == 'EV_EE_UEassociation_batch':
            from examples.giulia_EV_EE_UEassociation_batch import main as giulia_main
        elif example == 'EV_HapsSym_Reflector':
            from examples.giulia_EV_HapsSym_Reflector import main as giulia_main
        else:
            raise ValueError(f'Got an unknown example: {example}')

        # Invoke the given main method
        giulia_main(args)

        # Right now the output module is completely decoupled from the main logic, so we have to call the export method
        # manually. This should be automatic in the future once the conversion is thoroughly tested.
        output_module = OutputModule(os.path.realpath(self.output_dir), self.output_compression, self.rm_uncompressed)
        output_module.convert(
            # Include npz files that are not intended for plotting, and are not already computed (mean, median)
            filter_predicate=lambda file_name : file_name.endswith('.npz') and not file_name.startswith('to_plot') and not file_name.endswith('-mean.npz') and not file_name.endswith('-median.npz'),
            # Filter NPY files based on the columns given
            npy_filter=lambda file_name : self.output_fields is None or file_name in self.output_fields,
        )
