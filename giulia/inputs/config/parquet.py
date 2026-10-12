from typing import Optional, Dict, Any

from giulia.inputs.config.OutputFormat import OutputFormat


class ParquetFormat(OutputFormat):
    """
    Configuration for parquet output format.
    """

    compression: Optional[str]
    """Into which format to compress the parquet files. Options are: 'zip' or ``None``."""

    use_partition_cols: bool
    """
    Whether to use partition columns when writing parquet files.
    If ``True``, the parquet data will be partitioned by episode and step.
    This can improve query performance when reading the parquet files, but it will also increase the number of files and directories created.
    """

    rm_uncompressed: bool
    """Whether to remove uncompressed parquet files after compression."""

    def __init__(self, enable: bool = True, compression: Optional[str] = 'zip', use_partition_cols: bool = True, rm_uncompressed: bool = True):
        super().__init__(enable)
        self.compression = compression
        self.use_partition_cols = use_partition_cols
        self.rm_uncompressed = rm_uncompressed

    def __str__(self):
        return f"ParquetFormat(enable={self.enable}, compression={self.compression}, rm_uncompressed={self.rm_uncompressed})"

    @staticmethod
    def parse(dict: Dict[str, Any]) -> "ParquetFormat":
        enable = dict.get("enable", True)
        compression: Optional[str] = dict.get("compression", None)
        use_partition_cols = dict.get("use_partition_cols", True)
        rm_uncompressed: bool = dict.get("rm_uncompressed", False)

        return ParquetFormat(enable=enable, compression=compression, use_partition_cols=use_partition_cols, rm_uncompressed=rm_uncompressed)
