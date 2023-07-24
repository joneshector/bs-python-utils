import numpy as np

from bs_python_utils.bivariate_quantiles import (
    bivariate_quantiles,
    bivariate_quantiles_v,
    bivariate_ranks,
    bivariate_ranks_v,
    solve_for_v_,
)
from bs_python_utils.bsnputils import bsgrid, ecdf, npmaxabs


def test_bivariate_quantiles():
    n = 200
    n_nodes = 32

    rng = np.random.default_rng(seed=None)
    y = rng.normal(loc=0, scale=1, size=(n, 2))

    verbose = False

    vstar1 = solve_for_v_(y, n_nodes, verbose)

    n_qtiles = 4
    qtiles = np.arange(n_qtiles + 1) / n_qtiles
    u_qtiles = bsgrid(qtiles, qtiles)
    y_qtiles = bivariate_quantiles_v(y, u_qtiles, vstar1)
    print(f"{y_qtiles=}")

    y_ranks = bivariate_ranks_v(y, vstar1, n_nodes=n_nodes)

    indep_ranks = np.column_stack((ecdf(y[:, 0]), ecdf(y[:, 1])))
    print(f"max difference with indep ranks: {npmaxabs(y_ranks - indep_ranks)}")

    y_qtiles2 = bivariate_quantiles(y, u_qtiles, n_nodes=n_nodes)

    y_ranks2 = bivariate_ranks(y, n_nodes=n_nodes)

    assert np.allclose(y_qtiles, y_qtiles2)
    assert np.allclose(y_ranks, y_ranks2)
