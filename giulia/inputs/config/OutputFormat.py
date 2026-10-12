class OutputFormat:
    """
    Class to configure output formatting options.
    """

    enable: bool
    """Indicates whether this output formatting is enabled."""

    def __init__(self, enable: bool = True):
        self.enable = enable
