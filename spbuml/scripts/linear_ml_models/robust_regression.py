import numpy as np
from scipy.stats import norm, cauchy
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor


COLOR_PALETTE = {
    "dark_blue": "#173F5F",
    "light_blue": "#20639B",
    "light_green": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
}


def get_predictions(X_train, y_train, X_test):
    mean_estimator = LinearRegression()
    median_estimator = QuantileRegressor(
        quantile=0.5, alpha=0.0
    )  # disable regulalization
    mean_estimator.fit(X_train, y_train)
    median_estimator.fit(X_train, y_train)
    y_mean_pred = mean_estimator.predict(X_test)
    y_median_pred = median_estimator.predict(X_test)
    return y_mean_pred, y_median_pred


if __name__ == "__main__":
    sigma_values = [1.0, 2.0, 3.0]

    fig, axes = plt.subplots(1, len(sigma_values), figsize=(12, 4))

    # Fit estimators
    x_min, x_max = 0.0, 10.0
    n_samples = 20
    X = np.random.uniform(low=x_min, high=x_max, size=n_samples).reshape((-1, 1))
    X_pred = np.linspace(x_min, x_max, 10).reshape((-1, 1))
    for i, sigma in enumerate(sigma_values):
        # epsilon = norm(loc=0.0, scale=sigma).rvs(size=X.shape[0])
        epsilon = cauchy(scale=1.0).rvs(size=X.shape[0])
        y_noise_free = 2.0 * X[:, 0]
        y = y_noise_free + epsilon
        y_mean_pred, y_median_pred = get_predictions(X, y, X_pred)
        ax = axes[i]
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
            X[:, 0],
            y_noise_free,
            color=COLOR_PALETTE["red"],
            alpha=0.2,
            linewidth=3,
            label=r"$E[Y|x]$",
        )
        ax.plot(
            X_pred[:, 0],
            y_mean_pred,
            color=COLOR_PALETTE["light_green"],
            linewidth=3,
            label="LS linear regression",
        )
        ax.plot(
            X_pred[:, 0],
            y_median_pred,
            color=COLOR_PALETTE["light_blue"],
            linewidth=3,
            label="LAD linear regression",
        )
        ax.grid()
        ax.legend(fontsize=10)
    axes[0].set_ylabel(r"$y$", fontsize=12)
    for ax in axes:
        ax.set_xlabel(r"$x$", fontsize=12)
    fig.tight_layout()
    fig.savefig("robust_regression.svg")
    plt.show()
