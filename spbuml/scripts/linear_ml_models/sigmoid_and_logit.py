from signal import Sigmasks
import numpy as np
import matplotlib.pyplot as plt


COLOR_PALETTE = {
    "dark_blue": "#173F5F",
    "light_blue": "#20639B",
    "light_green": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
}


def sigmoid(x: float):
    return np.exp(x) / (1 + np.exp(x))


def logit(p: float):
    return np.log(p / (1 - p))


if __name__ == "__main__":
    x = np.linspace(-8.0, 8.0, 200)
    p = np.linspace(1e-3, 1 - 1e-3, 200)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(x, sigmoid(x), linewidth=4)
    ax.set_xlabel(r"$x$", fontsize=12)
    ax.set_ylabel(r"$g(x)$", fontsize=12)
    ax.grid()
    fig.tight_layout()
    fig.savefig("sigmoid.svg")

    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(p, logit(p), linewidth=4)
    ax.set_xlabel(r"$p$", fontsize=12)
    ax.set_ylabel(r"$x = g^{-1}(p)$", fontsize=12)
    ax.grid()
    fig.tight_layout()
    fig.savefig("logit.svg")
    plt.show()
