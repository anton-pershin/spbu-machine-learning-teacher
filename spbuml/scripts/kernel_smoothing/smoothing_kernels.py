import numpy as np
import matplotlib.pyplot as plt

COLOR_PALETTE = {
    "dark_blue": "#173F5F",
    "light_blue": "#20639B",
    "light_green": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
}


def _unit_support_mask(t):
    return (np.abs(t) <= 1).astype(np.float_)


def uniform(t):
    return 1.0 / 2 * _unit_support_mask(t)


def epanechnikov(t):
    return 3.0 / 4 * (1 - t**2) * _unit_support_mask(t)


def tricube(t):
    return 70.0 / 81 * (1.0 - np.abs(t) ** 3) ** 3 * _unit_support_mask(t)


def gaussian(t):
    return 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-1.0 / 2 * t**2)


if __name__ == "__main__":
    np.random.seed(42)
    kernel_infos = [
        dict(name="uniform", label="Uniform", func=uniform),
        dict(name="epanechnikov", label="Epanechnikov", func=epanechnikov),
        dict(name="tricube", label="Tricube", func=tricube),
        dict(name="gaussian", label="Gaussian", func=gaussian),
    ]
    t = np.linspace(-4, 4, 100)
    for kernel_info in kernel_infos:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
        ax.plot(
            t, kernel_info["func"](t), linewidth=4, color=COLOR_PALETTE["light_green"]
        )
        ax.set_xlabel(r"t", fontsize=12)
        ax.grid()
        fig.tight_layout()
        fig.savefig(f"{kernel_info['name']}_smooth_kernel.svg")
        plt.show()
    print("qwer")
