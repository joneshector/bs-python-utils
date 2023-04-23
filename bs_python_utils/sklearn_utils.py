"""
Cntains various `scikit-learn` utility programs.
"""


import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def skl_npreg_lasso(
    y: np.ndarray, X: np.ndarray, alpha: float, degree: int | None = 4
) -> np.ndarray:
    """
    Lasso nonparametric regression of `y` over polynomials of `X`

    Args:
        y:  shape `(nobs)`
        X: shape  `(nobs, nfeatures)`
        alpha:  Lasso penalty parameter
        degree: highest total degree

    Returns:
        the `(nobs)` array `E(y\\vert X)` over the sample
    """

    # first scale the X variables
    stdsc = StandardScaler()
    sfit = stdsc.fit(X)
    X_scaled = sfit.transform(X)
    pf = PolynomialFeatures(degree)
    # Create the features and fit
    X_poly = pf.fit_transform(X_scaled)
    # now run Lasso
    reg = Lasso(alpha=alpha).fit(X_poly, y)
    expy_X = reg.predict(X_poly)
    return expy_X


def plot_lasso_path(y: np.ndarray, X: np.ndarray, eps: float = 1e-3) -> None:
    """
    plot Lasso coefficient paths

    Args:
        y:  shape `(nobs)`
        X: shape  `(nobs, nfeatures)`
        eps: length of path

    Returns:
        plots the paths
    """
    # Compute paths
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(X, y, eps)

    plt.clf()
    # Display results
    plt.figure(1)
    colors = cycle(["b", "r", "g", "c", "k"])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    for coef_l, c in zip(coefs_lasso, colors, strict=True):
        plt.plot(neg_log_alphas_lasso, coef_l, c=c)

    plt.xlabel("-Log(alpha)")
    plt.ylabel("coefficients")
    plt.title("Lasso Paths")
    plt.axis("tight")

    plt.show()

    return


if __name__ == "__main__":
    n_obs = 10000
    X1 = -2.0 + 3.0 * np.random.uniform(size=n_obs)
    X2 = np.random.normal(loc=1.0, scale=2.0, size=n_obs)
    y = X1 * X2 * X2 / 100.0 - (X1 / 5.0 - X2 / 3.0) ** 3 + np.random.normal(size=n_obs)

    X = np.column_stack((X1, X2))

    from itertools import cycle

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import lasso_path

    plt.style.use("seaborn")

    degree = 10
    stdsc = StandardScaler()
    sfit = stdsc.fit(X)
    X_scaled = sfit.transform(X)
    pf = PolynomialFeatures(degree)
    # Create the features and fit
    X_poly = pf.fit_transform(X_scaled)

    y_pred = skl_npreg_lasso(y, X)

    plt.clf()

    ax = plt.axes()
    ax.scatter(y, y_pred)
    ax.plot(y, y, "-r")
    plt.show()
