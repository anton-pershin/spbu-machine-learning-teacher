from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay


COLOR_PALETTE = {
    "dark_blue": "#173F5F",
    "light_blue": "#20639B",
    "light_green": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
}


class ParzenRosenblattKde:
    def __init__(self, kernel):
        self.kernel = kernel
        self.coef_ = []
        self._X = None
        self._y = None

    def fit(self, X, y=None):
        self._X = X
        self._y = y

    def predict(self, X):
        y = np.zeros((len(X),), dtype=np.float_)
        for i in range(len(X)):
            y[i] = len(X) * np.sum(self.kernel(self._X, X[i]))
        return y


def gaussian(t):
    return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-1.0 / 2 * t**2)


def kernel(x_0, x):
    lmbd = 1.0
    z = np.linalg.norm(x_0 - x, axis=1) / lmbd
    return gaussian(z)


def generate_blob(means, cov_matrix, n):
    return multivariate_normal(means, cov_matrix).rvs(size=n)


def generate_dataset(n_classes: int, n_per_class: int, x_dim: int):
    n_samples = n_classes * n_per_class
    means = np.array(
        [
            [-10, -10],
            [0, 0],
            [10, 10],
        ],
        dtype=np.float_,
    )
    sigma_squared = 7
    cov_matricies = np.array(
        [
            [
                [sigma_squared, 0],
                [0, sigma_squared],
            ],
            [
                [sigma_squared, 0],
                [0, sigma_squared],
            ],
            [
                [sigma_squared, 0],
                [0, sigma_squared],
            ],
        ],
        dtype=np.float_,
    )
    X = np.zeros((n_samples, x_dim), dtype=np.float_)
    y = np.concatenate(
        [
            np.full((n_per_class,), class_i, dtype=np.int_)
            for class_i in range(n_classes)
        ]
    )
    for class_i in range(n_classes):
        samples_per_class = generate_blob(
            means=means[class_i], cov_matrix=cov_matricies[class_i], n=n_per_class
        )
        X[class_i * n_per_class : (class_i + 1) * n_per_class] = samples_per_class

    return X, y


def add_training_data_to_axis(X, n_classes, class_colors, ax, three_dim=False):
    for class_i in range(n_classes):
        X_per_class = X[class_i * n_per_class : (class_i + 1) * n_per_class]
        if three_dim:
            ax.scatter(
                X_per_class[:, 0],
                X_per_class[:, 1],
                zs=0,
                zdir="z",
                color=COLOR_PALETTE[class_colors[class_i]],
            )
        else:
            ax.scatter(
                X_per_class[:, 0],
                X_per_class[:, 1],
                color=COLOR_PALETTE[class_colors[class_i]],
            )


def make_figure_pretty(fig, ax):
    ax.set_xlabel(r"$x_1$", fontsize=12)
    ax.set_ylabel(r"$x_2$", fontsize=12)
    ax.grid()
    fig.tight_layout()


def plot_kde_predictive_distrs(X, y, n_classes: int):
    class_colors = list(COLOR_PALETTE.keys())[1 : n_classes + 1]

    # lda = LinearDiscriminantAnalysis(store_covariance=True)
    # lda.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    add_training_data_to_axis(X, n_classes, class_colors, ax)

    # cov_inv = np.linalg.inv(lda.covariance_)
    # a = np.log(lda.priors_) - 1 / 2.0 * np.diag(lda.means_ @ cov_inv @ lda.means_.T)
    # b = lda.means_ @ cov_inv
    # Theta = np.concatenate((a.reshape(-1, 1), b), axis=1).T

    # contours
    x1 = np.linspace(-15.0, 15.0, 50)
    x2 = np.linspace(-15.0, 15.0, 50)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))

    for class_i in range(n_classes):
        kde = ParzenRosenblattKde(kernel=kernel)
        kde.fit(X[y == class_i])
        P = kde.predict(pos.reshape(-1, 2)).reshape((len(x1), len(x2)))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        add_training_data_to_axis(X, n_classes, class_colors, ax, three_dim=True)
        surf = ax.plot_surface(
            X1, X2, P, linewidth=0, antialiased=False, cmap=cm.coolwarm, alpha=0.5
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.view_init(elev=20.0, azim=-35)
        fig.tight_layout()
        fig.savefig(f"kde_class_{class_i + 1}_pred_prob.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    n_classes = 3
    n_per_class = 100
    x_dim = 2
    X, y = generate_dataset(n_classes=n_classes, n_per_class=n_per_class, x_dim=x_dim)
    plot_kde_predictive_distrs(X, y, n_classes=n_classes)
