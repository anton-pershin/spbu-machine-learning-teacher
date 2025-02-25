import numpy as np
import matplotlib.pyplot as plt


def mse_loss(v):
    return v**2


def mae_loss(v):
    return np.abs(v)


def huber_loss(v):
    delta = 0.5
    return np.where(np.abs(v) > delta, delta * np.abs(v) - delta**2 / 2., 1./2 * v**2)


if __name__ == "__main__":
    v = np.linspace(-2, 2, 201)

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    for loss, label in zip(
        (mse_loss, mae_loss, huber_loss),
        (r"$\phi_{mse}$", r"$\phi_{mae}$", r"$\phi_{hub}$")
    ):
        ax.plot(v, loss(v), linewidth=4, label=label)
    ax.grid()
    ax.set_xlabel(r"$v$", fontsize=12)
    ax.set_ylabel(r"$\phi_{\cdot}$", fontsize=12)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig("regression_loss_functions.svg")
