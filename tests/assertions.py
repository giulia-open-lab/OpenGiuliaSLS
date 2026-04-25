import numpy as np
import pandas as pd
from typing import Union

ValueType = Union[pd.Series, float]

def calculate_error(actual: ValueType, expected: ValueType) -> ValueType:
    """Calculates the error percentage between an expected value and the actual one.
    
    Note: if ``expected`` is ``0``, the division is made by ``1``.

    Args:
        actual: The actual value.
        expected: The theoretical value.

    Returns:
        The error percentage (0.0-1.0)
    """

    if np.allclose(actual, expected):
        return 0.00
    
    else:
        if isinstance(actual, pd.Series) and isinstance(expected, pd.Series):
            # Use np.where for element-wise conditional
            return np.where(expected != 0, abs((actual - expected) / actual), abs(actual - expected))
        else:
            return abs((actual - expected) / actual) if expected != 0 else abs(actual - expected)
