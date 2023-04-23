"""
Contains various `scipy` utility programs.
"""
from math import sqrt

import numpy as np
import scipy.stats as sts
from scipy.interpolate import UnivariateSpline

from bs_python_utils.bsnputils import rice_stderr, test_vector
from bs_python_utils.bsutils import bs_error_abort, print_stars


def describe_array(v: np.ndarray, name: str | None = "v"):
    """
    descriptive statistics on an array interpreted as a vector

    Args:
        v: the array
        name: its name

    Returns:
        the `scipy.stats.describe` object
    """
    print_stars(f"{name} has:")
    d = sts.describe(v, None)
    print(f"Number of elements: {d.nobs}")
    print(f"Minimum: {d.minmax[0]}")
    print(f"Maximum: {d.minmax[1]}")
    print(f"Mean: {d.mean}")
    print(f"Stderr: {sqrt(d.variance)}")
    return d


def spline_reg(
    y: np.ndarray,
    x: np.ndarray,
    x_new: np.ndarray | None = None,
    is_sorted: bool | None = False,
    smooth: bool | None = True,
) -> np.array:
    """
    one-dimensional spline interpolation of vector y on vector x

    Args:
        y: vector of y-values
        x: vector of x-values
        x_new: where we evaluate (at the points in `x` by default)
        is_sorted: True if `x` is sorted in increasing order
        smooth: True if we want a smoother; otherwise we go through all points provided
        verbose: prints stuff if True

    Returns:
        values interpolated at `x_new`
    """
    n = test_vector(x)
    ny = test_vector(y)
    if ny != n:
        bs_error_abort("x and y should have the same size")

    if not is_sorted:
        # need to sort by increasing value of x
        order_rhs = np.argsort(x)
        rhs = x[order_rhs]
        lhs = y[order_rhs]
    else:
        rhs, lhs = x, y

    if smooth:
        # we compute a local estimator of the stderr of (y | x) and we use it to enter weights
        sigyx = rice_stderr(lhs, rhs)
        w = 1 / sigyx
        spl = UnivariateSpline(rhs, lhs, w=w)
    else:
        spl = UnivariateSpline(rhs, lhs)

    xeval = x if x_new is None else x_new
    y_pred = spl(xeval)
    return y_pred
