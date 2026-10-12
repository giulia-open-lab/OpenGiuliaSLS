import os.path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
import shutil

from giulia import PROJ_DIR
from giulia.inputs.config.OutputFormat import OutputFormat
from giulia.inputs.config.parquet import ParquetFormat
from giulia.inputs.config.npz import NPZFormat
from giulia.inputs.run_config import RunConfig
from giulia.logger import warning, debug


class InputConfig:
    args: Dict[str, Any]
    """
    All the arguments to be passed to giulia_EV.
    """

    run_config: RunConfig
    """
    All the configuration that sets how to run Giulia.
    """

    output_dir: Optional[str]
    """
    The directory where to save the outputs.
    """

    output_fields: Optional[List[str]]
    """
    A list of all the fields to give on the outputs file.
    """

    output_formats: List[OutputFormat]
    """
    A list of formats to use for Giulia's output.
    """

    def __init__(
            self,
            args: Dict[str, Any],
            run_config: RunConfig,
            output_dir: Optional[str] = None,
            output_fields: Optional[List[str]] = None,
            output_formats: Optional[List[OutputFormat]] = None
    ):
        """
        Builds a new instance of ``InputConfig``.

        Args:
            args: All the arguments to be passed to giulia_EV.
            output_fields: All the fields to give on the outputs file.
        """
        self.args = args if args is not None else {}
        self.run_config = run_config
        self.output_dir = output_dir if output_dir is not None else output_dir
        self.output_fields = output_fields if output_fields is not None and len(output_fields) > 0 else None
        self.output_formats = output_formats if output_formats is not None else [ParquetFormat()]

    @staticmethod
    def load(file_path: Optional[str] = None) -> 'InputConfig':
        config_file_path: str = file_path if file_path is not None else os.path.join(PROJ_DIR, 'config.yml')
        if file_path is None:
            debug('No input file provided. Defaulting to "config.yaml"')

        if not os.path.exists(config_file_path) and file_path is None:
            warning('"config.yaml" does not exist. Pre-filling the file with default values.')
            example_file_path = os.path.join(PROJ_DIR, 'config.example.yml')
            shutil.copyfile(example_file_path, config_file_path)

        # Read the file
        with open(config_file_path, "r") as f:
            data = yaml.safe_load(f.read())
        args: Dict[str, Any] = data['args']

        output_data: Dict[str, Any] = data['output']
        run_config: RunConfig = RunConfig.from_config(data['run'])
        output_dir: str = output_data['directory']
        output_fields: List[str] | None = output_data.get('fields')
        output_formats_dict: Optional[Dict[str, Any]] = output_data.get('formats')

        # Parse output formats
        output_formats: List[OutputFormat] = []
        if output_formats_dict is not None:
            if 'parquet' in output_formats_dict:
                config = ParquetFormat.parse(output_formats_dict['parquet'])
                output_formats.append(config)
            if 'npz' in output_formats_dict:
                config = NPZFormat.parse(output_formats_dict['npz'])
                output_formats.append(config)
        else:
            # By default, only parquet in raw format
            output_formats = [ParquetFormat()]

        return InputConfig(args, run_config, output_dir, output_fields, output_formats)

    def run(self) -> Tuple[Callable, List[str]]:
        """
        Prepare the run: parse the arguments and resolve which Giulia entry
        point to use.

        Returns:
            The resolved ``giulia_main`` callable and the parsed command-line
            ``args``, ready to be invoked as ``giulia_main(args, self)``.
        """
        # Convert the dictionary into proper command line arguments.
        # Also filter entries with null value
        args = list(f"--{k}={v}" for k, v in self.args.items() if v is not None)

        if self.output_formats is None or len(self.output_formats) <= 0:
            warning('No output formats specified. Giulia won\'t generate any outputs.')

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

        # Everything is set up: hand the entry point back to the caller.
        return giulia_main, args
