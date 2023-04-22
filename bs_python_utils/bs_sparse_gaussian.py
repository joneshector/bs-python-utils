"""
sets up sparse integration over a Gaussian
"""
import os

import numpy as np

from bs_python_utils.bsutils import bs_error_abort


def setup_sparse_gaussian(ndims: int, iprec: int, GHsparsedir: str = None):
    """
    get nodes and weights for sparse integration Ef(X) with X = N(0,1) in `ndims` dimensions

    usage: nodes, weights = setup_sparse_gaussian(mdims, iprec); intf = f(nodes) @ weights

    Ars:
        ndims: number of dimensions (1 to 5)
        iprec: precision (must be 9, 13, or 17)

    Returns: 
        a pair of  arrays `nodes` and `weights`; \
        `nodes` has `ndims`-1 columns and weights is a vector
    """
    if GHsparsedir is None:
        GHsparsedir = os.path.join(os.getenv("HOME"), "Dropbox/GHsparseGrids")
    if iprec not in [9, 13, 17]:
        bs_error_abort(
            f"We only do sparse integration with precision 9, 13, or 17, not {iprec}"
        )

    if ndims in [1, 2, 3, 4, 5]:
        grid = np.loadtxt(
            os.path.join(GHsparsedir, f"GHsparseGrid{ndims}prec" + str(iprec) + ".txt")
        )
        weights = grid[:, 0]
        nodes = grid[:, 1:]
        return nodes, weights
    else:
        bs_error_abort(
            f"We only do sparse integration in one or two dimensions, not {ndims}"
        )


if __name__ == "__main__":
    n = 5
    iprec = 13

    nodes, weights = setup_sparse_gaussian(n, iprec)

    def f(x):
        return np.sum(x**2, 1)

    intf = f(nodes) @ weights

    print(f"Integral in {n} dimensions should be {n}, it is {intf: 10.6f}")
