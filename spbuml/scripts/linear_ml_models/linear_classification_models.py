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


def plot_linear_regression_boundary(X, y, n_classes: int):
    X_lr = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    Y_lr = np.zeros((len(y), n_classes))
    Y_lr[np.arange(Y_lr.shape[0]), y] = 1
    Theta = np.linalg.inv(X_lr.T @ X_lr) @ X_lr.T @ Y_lr

    class_colors = list(COLOR_PALETTE.keys())[1 : n_classes + 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    add_training_data_to_axis(X, n_classes, class_colors, ax)

    for (class_1_i, class_2_i), (x1_min, x1_max) in zip(
        (
            (0, 2),
            (0, 1),
            (1, 2),
        ),
        (
            (-10, 0),
            (0, 10),
            (0, 10),
        ),
    ):
        separator = (
            lambda x: (Theta[0, class_2_i] - Theta[0, class_1_i])
            / (Theta[2, class_1_i] - Theta[2, class_2_i])
            + (Theta[1, class_2_i] - Theta[1, class_1_i])
            / (Theta[2, class_1_i] - Theta[2, class_2_i])
            * x
        )
        x1 = np.linspace(x1_min, x1_max, 20)
        ax.plot(x1, separator(x1), linewidth=3, color="black")

    make_figure_pretty(fig, ax)
    fig.savefig("lin_reg_classifier_3_classes.svg")


def plot_lda_boundary(X, y, n_classes: int):
    class_colors = list(COLOR_PALETTE.keys())[1 : n_classes + 1]

    lda = LinearDiscriminantAnalysis(store_covariance=True)
    lda.fit(X, y)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    add_training_data_to_axis(X, n_classes, class_colors, ax)

    cov_inv = np.linalg.inv(lda.covariance_)
    a = np.log(lda.priors_) - 1 / 2.0 * np.diag(lda.means_ @ cov_inv @ lda.means_.T)
    b = lda.means_ @ cov_inv
    Theta = np.concatenate((a.reshape(-1, 1), b), axis=1).T

    for (class_1_i, class_2_i), (x1_min, x1_max) in zip(
        (
            (0, 1),
            (1, 2),
        ),
        (
            (-15, 0),
            (-2, 12),
        ),
    ):
        separator = (
            lambda x: (Theta[0, class_2_i] - Theta[0, class_1_i])
            / (Theta[2, class_1_i] - Theta[2, class_2_i])
            + (Theta[1, class_2_i] - Theta[1, class_1_i])
            / (Theta[2, class_1_i] - Theta[2, class_2_i])
            * x
        )
        x1 = np.linspace(x1_min, x1_max, 20)
        ax.plot(x1, separator(x1), linewidth=3, color="black")

    make_figure_pretty(fig, ax)
    fig.savefig("lda_classifier_3_classes.svg")

    # contours
    x1 = np.linspace(-15.0, 15.0, 50)
    x2 = np.linspace(-15.0, 15.0, 50)
    X1, X2 = np.meshgrid(x1, x2)
    pos = np.dstack((X1, X2))

    for class_i in range(n_classes):
        pdf = multivariate_normal(lda.means_[class_i], lda.covariance_).pdf
        P = pdf(pos)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        add_training_data_to_axis(X, n_classes, class_colors, ax, three_dim=True)
        surf = ax.plot_surface(
            X1, X2, P, linewidth=0, antialiased=False, cmap=cm.coolwarm, alpha=0.5
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.view_init(elev=20.0, azim=-35)
        fig.tight_layout()
        fig.savefig(f"lda_class_{class_i + 1}_pred_prob.png", dpi=150)
        plt.show()


def plot_logistic_regression_boundary(X, y, n_classes: int):
    class_colors = list(COLOR_PALETTE.keys())[1 : n_classes + 1]

    lr = LogisticRegression(penalty=None)
    lr.fit(X, y + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    add_training_data_to_axis(X, n_classes, class_colors, ax)
    Theta = np.concatenate((lr.intercept_.reshape(-1, 1), lr.coef_), axis=1).T

    for (class_1_i, class_2_i), (x1_min, x1_max) in zip(
        (
            (0, 1),
            (1, 2),
        ),
        (
            (-15, 0),
            (-2, 12),
        ),
    ):
        separator = (
            lambda x: (Theta[0, class_2_i] - Theta[0, class_1_i])
            / (Theta[2, class_1_i] - Theta[2, class_2_i])
            + (Theta[1, class_2_i] - Theta[1, class_1_i])
            / (Theta[2, class_1_i] - Theta[2, class_2_i])
            * x
        )
        x1 = np.linspace(x1_min, x1_max, 20)
        ax.plot(x1, separator(x1), linewidth=3, color="black")

    make_figure_pretty(fig, ax)
    fig.savefig("log_reg_classifier_3_classes.svg")

    # countours
    x1 = np.linspace(-15.0, 15.0, 50)
    x2 = np.linspace(-15.0, 15.0, 50)
    X1, X2 = np.meshgrid(x1, x2)

    for class_i in range(n_classes):
        pdf_denominator = lambda x1, x2: np.sum(
            [np.exp(Theta[:, j] @ [1.0, x1, x2]) for j in range(n_classes)], axis=0
        )
        pdf = lambda x1, x2: np.exp(
            Theta[:, class_i] @ [1.0, x1, x2]
        ) / pdf_denominator(x1, x2)

        P = pdf(X1, X2)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        add_training_data_to_axis(X, n_classes, class_colors, ax, three_dim=True)
        surf = ax.plot_surface(
            X1, X2, P, linewidth=0, antialiased=False, cmap=cm.coolwarm, alpha=0.5
        )
        fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.view_init(elev=20.0, azim=-35)
        fig.tight_layout()
        fig.savefig(f"log_reg_class_{class_i + 1}_pred_prob.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    n_classes = 3
    n_per_class = 100
    x_dim = 2
    X, y = generate_dataset(n_classes=n_classes, n_per_class=n_per_class, x_dim=x_dim)
    plot_linear_regression_boundary(X=X, y=y, n_classes=n_classes)
    plot_lda_boundary(X, y, n_classes=n_classes)
    plot_logistic_regression_boundary(X, y, n_classes=n_classes)
