""" General plotting utilities.
"""

import numpy as np


def set_axis(x: np.ndarray, margin: float = 0.05) -> tuple[float, float]:
    """sets the axis limits  with a margin

    Args:
        x: the values of the variable
        margin: the margin to add, a fraction of the range of the variable

    Returns:
        the min and max for the axis.
    """
    x_min, x_max = x.min(), x.max()
    scaled_diff = margin * (x_max - x_min)
    x_min -= scaled_diff
    x_max += scaled_diff
    return x_min, x_max
