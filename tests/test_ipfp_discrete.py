import numpy as np
import pytest

from bs_python_utils.ipfp_discrete_solvers import ipfp_homo_solver, \
    ipfp_hetero_solver, ipfp_heteroxy_solver


@pytest.fixture
def generate_cs_homo():
    # we generate a Choo and Siow homo matching
    ncat_men = ncat_women = 25
    n_sum_categories = ncat_men + ncat_women
    n_prod_categories = ncat_men * ncat_women

    n_bases = 4
    bases_surplus = np.zeros((ncat_men, ncat_women, n_bases))
    x_men = (np.arange(ncat_men) - ncat_men / 2.0) / ncat_men
    y_women = (np.arange(ncat_women) - ncat_women / 2.0) / ncat_women

    bases_surplus[:, :, 0] = 1
    for iy in range(ncat_women):
        bases_surplus[:, iy, 1] = x_men
    for ix in range(ncat_men):
        bases_surplus[ix, :, 2] = y_women
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            bases_surplus[ix, iy, 3] = (x_men[ix] - y_women[iy]) * (
                x_men[ix] - y_women[iy]
            )

    men_margins = np.random.uniform(1.0, 10.0, size=ncat_men)
    women_margins = np.random.uniform(1.0, 10.0, size=ncat_women)

    true_surplus_params = np.array([3.0, -1.0, -1.0, -2.0])
    true_surplus_matrix = bases_surplus @ true_surplus_params

    return true_surplus_matrix, men_margins, women_margins


def test_homo_solver(generate_cs_homo):
    true_surplus_matrix, men_margins, women_margins = generate_cs_homo
    mus, marg_err_x, marg_err_y = ipfp_homo_solver(
        true_surplus_matrix, men_margins, women_margins, tol=1e-12
    )
    tol = 1e-6
    assert marg_err_x < tol
    assert marg_err_y < tol


def test_gradient_hetero():
    pass


def test_gradient_heteroxy():
    pass



def test_ipfp_hetero(generate_cs_homo):
    true_surplus_matrix, men_margins, women_margins = generate_cs_homo
    #  we test ipfp hetero for tau = 1
    tau = 1.0
    mus_tau, marg_err_x_tau, marg_err_y_tau = ipfp_hetero_solver(
        true_surplus_matrix, men_margins, women_margins, tau
    )
    tol = 1e-6
    assert marg_err_x < tol
    assert marg_err_y < tol


def test_ipfp_hetero(generate_cs_homo):
    true_surplus_matrix, men_margins, women_margins = generate_cs_homo
    # we test ipfp heteroxy for sigma = tau = 1
    sigma_x = np.ones(ncat_men)
    tau_y = np.ones(ncat_women)

    mus_hxy, marg_err_x_hxy, marg_err_y_hxy = ipfp_heteroxy_solver(
        true_surplus_matrix, men_margins, women_margins, sigma_x, tau_y
    )
    muxy_hxy, _, _ = mus_hxy
    _print_simulated_ipfp(muxy_hxy, marg_err_x_hxy, marg_err_y_hxy)

    # and we test ipfp homo w/o singles
    print_stars("Testing ipfp homo w/o singles:")
    # we need as many women as men
    women_margins_nosingles = women_margins * (
        np.sum(men_margins) / np.sum(women_margins)
    )
    muxy_nos, marg_err_x_nos, marg_err_y_nos = ipfp_homo_nosingles_solver(
        true_surplus_matrix, men_margins, women_margins_nosingles, gr=False
    )
    _print_simulated_ipfp(muxy_nos, marg_err_x_nos, marg_err_y_nos)

    # check the grad_f
    iman = 3
    iwoman = 17

    GRADIENT_STEP = 1e-6

    if do_test_gradient_heteroxy:
        mus_hxy, marg_err_x_hxy, marg_err_y_hxy, dmus_hxy = ipfp_heteroxy_solver(
            true_surplus_matrix, men_margins, women_margins, sigma_x, tau_y, gr=True
        )
        muij = mus_hxy[0][iman, iwoman]
        muij_x0 = mus_hxy[1][iman]
        muij_0y = mus_hxy[2][iwoman]
        gradij = dmus_hxy[0][iman * ncat_women + iwoman, :]
        gradij_x0 = dmus_hxy[1][iman, :]
        gradij_0y = dmus_hxy[2][iwoman, :]
        n_cols_rhs = n_prod_categories + 2 * n_sum_categories
        gradij_numeric = np.zeros(n_cols_rhs)
        gradij_numeric_x0 = np.zeros(n_cols_rhs)
        gradij_numeric_0y = np.zeros(n_cols_rhs)
        icoef = 0
        for ix in range(ncat_men):
            men_marg = men_margins.copy()
            men_marg[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroxy_solver(
                true_surplus_matrix, men_marg, women_margins, sigma_x, tau_y
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for iy in range(ncat_women):
            women_marg = women_margins.copy()
            women_marg[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroxy_solver(
                true_surplus_matrix, men_margins, women_marg, sigma_x, tau_y
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for i1 in range(ncat_men):
            for i2 in range(ncat_women):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = ipfp_heteroxy_solver(
                    surplus_mat, men_margins, women_margins, sigma_x, tau_y
                )
                gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
                gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
                gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
                icoef += 1
        for ix in range(ncat_men):
            sigma = sigma_x.copy()
            sigma[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroxy_solver(
                true_surplus_matrix, men_margins, women_margins, sigma, tau_y
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for iy in range(ncat_women):
            tau = tau_y.copy()
            tau[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_heteroxy_solver(
                true_surplus_matrix, men_margins, women_margins, sigma_x, tau
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1

        diff_gradients = gradij_numeric - gradij
        error_gradient = np.abs(diff_gradients)

        describe_array(error_gradient, "error on the numerical grad_f, heteroxy")

        diff_gradients_x0 = gradij_numeric_x0 - gradij_x0
        error_gradient_x0 = np.abs(diff_gradients_x0)

        describe_array(error_gradient_x0, "error on the numerical grad_f x0, heteroxy")

        diff_gradients_0y = gradij_numeric_0y - gradij_0y
        error_gradient_0y = np.abs(diff_gradients_0y)

        describe_array(error_gradient_0y, "error on the numerical grad_f 0y, heteroxy")

    if do_test_gradient_hetero:
        tau = 1.0
        mus_h, marg_err_x_h, marg_err_y_h, dmus_h = ipfp_hetero_solver(
            true_surplus_matrix, men_margins, women_margins, tau, gr=True
        )
        muij = mus_h[0][iman, iwoman]
        gradij = dmus_h[0][iman * ncat_women + iwoman, :]
        n_cols_rhs = n_prod_categories + n_sum_categories + 1
        gradij_numeric = np.zeros(n_cols_rhs)
        icoef = 0
        for ix in range(ncat_men):
            men_marg = men_margins.copy()
            men_marg[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_hetero_solver(
                true_surplus_matrix, men_marg, women_margins, tau
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            icoef += 1
        for iy in range(ncat_women):
            women_marg = women_margins.copy()
            women_marg[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = ipfp_hetero_solver(
                true_surplus_matrix, men_margins, women_marg, tau
            )
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            icoef += 1
        for i1 in range(ncat_men):
            for i2 in range(ncat_women):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = ipfp_hetero_solver(
                    surplus_mat, men_margins, women_margins, tau
                )
                gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
                icoef += 1
        tau_plus = tau + GRADIENT_STEP
        mus, marg_err_x, marg_err_y = ipfp_hetero_solver(
            true_surplus_matrix, men_margins, women_margins, tau_plus
        )
        gradij_numeric[-1] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP

        error_gradient = np.abs(gradij_numeric - gradij)

        describe_array(error_gradient, "error on the numerical grad_f, hetero")
