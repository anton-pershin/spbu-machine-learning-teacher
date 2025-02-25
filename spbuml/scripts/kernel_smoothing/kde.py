import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    np.random.seed(42)
    kde = ParzenRosenblattKde(kernel=kernel)
    X_train = np.concatenate(
        (
            np.random.normal(loc=1.0, scale=1.0, size=30),
            np.random.normal(loc=5.0, scale=2.0, size=20),
        )
    ).reshape((-1, 1))
    kde.fit(X_train)

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    x = np.linspace(-5, 12, 100)
    ax.plot(x, kde.predict(x), linewidth=4, color=COLOR_PALETTE["light_green"])
    ax.scatter(
        X_train, [0] * len(X_train), s=30, marker="|", color=COLOR_PALETTE["red"]
    )
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$\hat{f}_X(x)$", fontsize=12)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f"kde_example.svg")
    #    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
    ax.hist(
        X_train.reshape(-1), bins=20, color=COLOR_PALETTE["light_green"], density=True
    )
    ax.scatter(
        X_train, [0] * len(X_train), s=30, marker="|", color=COLOR_PALETTE["red"]
    )
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$\hat{f}_X(x)$", fontsize=12)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f"hist_example.svg")
#    plt.show()
