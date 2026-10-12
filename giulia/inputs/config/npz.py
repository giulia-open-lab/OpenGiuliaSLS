from typing import Any, Dict

from giulia.inputs.config.OutputFormat import OutputFormat


class NPZFormat(OutputFormat):
    """
    Class representing the NPZ output format.
    """

    def __init__(self, enable: bool = True):
        super().__init__(enable)

    @staticmethod
    def parse(dict: Dict[str, Any]) -> 'NPZFormat':
        enable = dict.get('enable', True)
        return NPZFormat(enable)
