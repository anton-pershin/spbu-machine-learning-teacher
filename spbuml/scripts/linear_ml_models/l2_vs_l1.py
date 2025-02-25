import numpy as np
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso


COLOR_PALETTE = {
    "dark_blue": "#173F5F",
    "light_blue": "#20639B",
    "light_green": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
}


def get_predictions(estimator, X_train, y_train, X_test):
    estimator.fit(X_train, y_train)
    print(estimator)
    print(estimator.coef_)
    print()
    y_pred = estimator.predict(X_test)
    return y_pred


def transform_x_to_poly_features(X, poly_degree):
    assert poly_degree >= 1
    X_with_poly_features = np.zeros((X.shape[0], poly_degree))
    for i in range(poly_degree):
        X_with_poly_features[:, i] = X[:, 0] ** (i + 1)
    return X_with_poly_features


def f(x):
    return x**3 + x**2 + x


if __name__ == "__main__":
    poly_degree = 12
    estimators = [
        dict(name="no_reg", label="No reg.", estimator=Ridge(alpha=0)),
        dict(
            name="l2", label="Ridge ($\lambda = 10^{-2}$)", estimator=Ridge(alpha=1e-2)
        ),
        dict(
            name="l1", label="Lasso ($\lambda = 10^{-2}$)", estimator=Lasso(alpha=1e-2)
        ),
    ]

    # Fit estimators
    x_min, x_max = -1.0, 1.0
    n_samples = 30
    sigma = 0.5
    X = np.random.uniform(low=x_min, high=x_max, size=n_samples).reshape((-1, 1))
    epsilon = norm(loc=0.0, scale=sigma).rvs(size=X.shape[0])
    y = f(X[:, 0]) + epsilon
    X_pred = np.linspace(x_min, x_max, 100).reshape((-1, 1))
    for i, est_info in enumerate(estimators):
        y_pred = get_predictions(
            est_info["estimator"],
            transform_x_to_poly_features(X, poly_degree),
            y,
            transform_x_to_poly_features(X_pred, poly_degree),
        )

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.plot(
            X[:, 0],
            y,
            "o",
            color=COLOR_PALETTE["red"],
            alpha=0.2,
            markersize=10,
            label=r"$\{(x^{(i)}, y^{(i)})\}$",
        )
        ax.plot(
            X_pred[:, 0],
            f(X_pred[:, 0]),
            color=COLOR_PALETTE["red"],
            alpha=0.2,
            linewidth=3,
            label=r"$E[Y|x]$",
        )
        ax.plot(
            X_pred[:, 0],
            y_pred,
            color=COLOR_PALETTE["light_green"],
            linewidth=3,
            label=est_info["label"],
        )
        ax.grid()
        ax.legend(fontsize=10)
        ax.set_xlabel(r"$x$", fontsize=12)
        ax.set_ylabel(r"$y$", fontsize=12)
        ax.set_ylim((-3, 5))
        fig.tight_layout()
        fig.savefig(f"{est_info['name']}_example.svg")
        plt.show()
    print("qwer")
