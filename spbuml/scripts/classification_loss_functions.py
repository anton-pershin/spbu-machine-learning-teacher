import numpy as np
import matplotlib.pyplot as plt


def logistic_loss(v):
    return 1. / np.log(2.) * np.log(1. + np.exp(-v))


def exponential_loss(v):
    return np.exp(-v)


def zero_one_loss(v):
    return (v < 0).astype(np.float_)


if __name__ == "__main__":
    v = np.linspace(-2, 2, 201)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for loss, label in zip(
        (zero_one_loss, exponential_loss, logistic_loss),
        (r"$\phi_{0-1}$", r"$\phi_{exp}$", r"$\phi_{log}$")
    ):
        ax.plot(v, loss(v), linewidth=4, label=label)
    ax.grid()
    ax.set_xlabel(r"$v$", fontsize=12)
    ax.set_ylabel(r"$\phi_{\cdot}$", fontsize=12)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig("classification_loss_functions.svg")
