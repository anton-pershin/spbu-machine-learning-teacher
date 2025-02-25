import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso


COLOR_PALETTE = {
    "dark_blue": "#173F5F",
    "light_blue": "#20639B",
    "light_green": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
}


class EpanechnikovKernel:
    def __init__(self, lmbd=1.0) -> None:
        self.lmbd = lmbd

    def __call__(self, x_0, x):
        z = np.linalg.norm(x_0 - x, axis=1) / self.lmbd
        support_mask = (z <= 1).astype(np.float_)
        return 3.0 / 4 * (1 - z**2) * support_mask


class KernelRegression:
    def __init__(self, kernel):
        self.kernel = kernel
        self.coef_ = []
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = X
        self._y = y

    def predict(self, X):
        y = np.zeros((len(X),), dtype=np.float_)
        for i in range(len(X)):
            y[i] = (
                self.kernel(self._X, X[i])
                @ self._y
                / np.sum(self.kernel(self._X, X[i]))
            )
        return y


class LoessRegression:
    def __init__(self, kernel):
        self.kernel = kernel
        self.coef_ = []
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = X
        self._y = y

    def predict(self, X):
        y = np.zeros((len(X),), dtype=np.float_)
        for i in range(len(X)):
            K = np.diag(self.kernel(self._X, X[i]))
            theta = np.linalg.inv(self._X.T @ K @ self._X) @ self._X.T @ K @ self._y
            y[i] = theta.T @ X[i]
        return y


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


def f1(x):
    return x**3 + x**2 + x


def f2(x):
    return x**3 - 4 * x**2 + x


def generate_training_dataset(func, n_samples, x_min, x_max):
    sigma = 0.5
    range_size = x_max - x_min
    # x_1 = x_min + (range_size / 2)
    # x_2 = x_min + (range_size / 2) + (range_size / 10)
    x_1 = x_min + (range_size / 2) + (range_size / 5)
    x_2 = x_min + (range_size / 2) + (range_size / 5) + (range_size / 10)
    n_samples_1 = int(2.0 / 5 * n_samples)
    n_samples_2 = int(1.0 / 5 * n_samples)
    #    n_samples_1 = int(2.0 / 5 * n_samples)
    #    n_samples_2 = int(1.0 / 5 * n_samples)
    # X = np.random.uniform(low=x_min, high=x_max, size=n_samples).reshape((-1, 1))
    X = np.concatenate(
        (
            np.random.uniform(low=x_min, high=x_1, size=n_samples_1),
            np.random.uniform(
                low=x_1,
                high=x_2,
                size=n_samples_2,
            ),
            np.random.uniform(
                low=x_2,
                high=x_max,
                size=n_samples_1,
            ),
        )
    ).reshape((-1, 1))
    epsilon = norm(loc=0.0, scale=sigma).rvs(size=X.shape[0])
    y = func(X[:, 0]) + epsilon
    return X, y


def prepend_unit_column(X):
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def add_training_dataset_to_axis(ax, X, y, X_pred, func):
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
        func(X_pred[:, 0]),
        color=COLOR_PALETTE["red"],
        alpha=0.2,
        linewidth=3,
        label=r"$E[Y|x]$",
    )


def add_prediction_to_axis(ax, X_pred, y_pred, label):
    ax.plot(
        X_pred[:, 0],
        y_pred,
        color=COLOR_PALETTE["light_green"],
        linewidth=3,
        label=label,
    )


def add_kernel_to_axis(ax, X_pred, y_pred, estimator):
    x_mid_idx = int(X_pred.shape[0] // 2)
    x_0 = X_pred[x_mid_idx, 0]
    y_0 = y_pred[x_mid_idx]
    x_kernel = np.linspace(x_0 - 0.6, x_0 + 0.6, 100)
    y_kernel = y_0 + 2.5 * (estimator.kernel(x_0, x_kernel.reshape(-1, 1)) - 1.0)
    ax.fill_between(
        x=x_kernel,
        y1=y_kernel.min(),
        y2=y_kernel,
        color=COLOR_PALETTE["light_blue"],
        alpha=0.5,
    )


def make_figure_pretty(fig, ax):
    ax.grid()
    ax.legend(fontsize=10)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$y$", fontsize=12)
    fig.tight_layout()


def plot_comparison_between_global_model_and_local_model():
    func = f1
    poly_degree = 12
    estimators = [
        dict(
            name="global_regr",
            label="Global regression",
            transform=True,
            estimator=Ridge(alpha=0),
        ),
        dict(
            name="local_kernel_regr",
            label="Kernel regression",
            transform=False,
            estimator=KernelRegression(kernel=EpanechnikovKernel(lmbd=0.5)),
        ),
    ]

    # Fit estimators
    x_min, x_max = -1.0, 1.0
    n_samples = 30
    X, y = generate_training_dataset(
        func=func, n_samples=n_samples, x_min=x_min, x_max=x_max
    )
    X_pred = np.linspace(x_min, x_max, 100).reshape((-1, 1))
    for i, est_info in enumerate(estimators):
        if est_info["transform"]:
            y_pred = get_predictions(
                est_info["estimator"],
                transform_x_to_poly_features(X, poly_degree),
                y,
                transform_x_to_poly_features(X_pred, poly_degree),
            )
        else:
            y_pred = get_predictions(
                est_info["estimator"],
                X,
                y,
                X_pred,
            )

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        add_training_dataset_to_axis(ax=ax, X=X, y=y, X_pred=X_pred, func=func)
        add_prediction_to_axis(
            ax=ax, X_pred=X_pred, y_pred=y_pred, label=est_info["label"]
        )
        if isinstance(est_info["estimator"], KernelRegression):
            add_kernel_to_axis(
                ax=ax, X_pred=X_pred, y_pred=y_pred, estimator=est_info["estimator"]
            )
        ax.set_ylim((-3, 5))
        make_figure_pretty(fig, ax)
        fig.savefig(f"{est_info['name']}_example.svg")
        # plt.show()


def plot_kernel_regression_artifacts_near_boundaries(X, y, x_min, x_max, func):
    estimator = KernelRegression(kernel=EpanechnikovKernel(lmbd=0.5))  # 0.5

    X_pred = np.linspace(x_min, x_max, 100).reshape((-1, 1))
    y_pred = get_predictions(estimator, X, y, X_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    add_training_dataset_to_axis(ax=ax, X=X, y=y, X_pred=X_pred, func=func)
    add_prediction_to_axis(
        ax=ax, X_pred=X_pred, y_pred=y_pred, label="Kernel regression"
    )
    add_kernel_to_axis(ax=ax, X_pred=X_pred, y_pred=y_pred, estimator=estimator)
    ax.set_ylim((-7, 2))
    make_figure_pretty(fig, ax)
    fig.savefig(f"kernel_regression_artifacts.svg")
    plt.show()
    print()


def plot_loess_regression(X, y, x_min, x_max, func):
    estimator = LoessRegression(kernel=EpanechnikovKernel(lmbd=0.5))  # 0.5
    X = prepend_unit_column(X)

    X_pred = np.linspace(x_min, x_max, 100).reshape((-1, 1))
    X_pred = prepend_unit_column(X_pred)
    y_pred = get_predictions(estimator, X, y, X_pred)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    add_training_dataset_to_axis(
        ax=ax,
        X=X[:, 1].reshape((-1, 1)),
        y=y,
        X_pred=X_pred[:, 1].reshape((-1, 1)),
        func=func,
    )
    add_prediction_to_axis(
        ax=ax,
        X_pred=X_pred[:, 1].reshape((-1, 1)),
        y_pred=y_pred,
        label="LOESS regression",
    )
    add_kernel_to_axis(
        ax=ax, X_pred=X_pred[:, 1].reshape((-1, 1)), y_pred=y_pred, estimator=estimator
    )
    ax.set_ylim((-7, 2))
    make_figure_pretty(fig, ax)
    fig.savefig(f"loess_example.svg")
    plt.show()
    print()


if __name__ == "__main__":
    np.random.seed(42)

    func = f2
    x_min, x_max = -1.0, 2.0
    n_samples = 80
    X, y = generate_training_dataset(
        func=func, n_samples=n_samples, x_min=x_min, x_max=x_max
    )

    # plot_comparison_between_global_model_and_local_model()
    plot_kernel_regression_artifacts_near_boundaries(X, y, x_min, x_max, func)
    plot_loess_regression(X, y, x_min, x_max, func)
