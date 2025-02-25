import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import pymc3
import arviz


COLOR_PALETTE = {
    "dark_blue": "#173F5F",
    "light_blue": "#20639B",
    "light_green": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
}


def build_prior(theta_0, V_0):
    return multivariate_normal(mean=theta_0, cov=V_0)


def build_posterior(theta_0, V_0, X, sigma, y):
    V_0_inv = np.linalg.inv(V_0)
    V_n = sigma**2 * np.linalg.inv(sigma**2 * V_0_inv + X.T @ X)
    theta_n = V_n @ V_0_inv @ theta_0 + 1.0 / sigma**2 * V_n @ X.T @ y
    return multivariate_normal(mean=theta_n, cov=V_n)


def sample_predictors(posterior, n):
    return posterior.rvs(size=n)


def plot_posterior_distribution_and_sampled_predictors_wrt_v_0(
    X, y, y_noise_free, sigma, theta_true
):
    theta_0 = np.array([0.0, 0.0])
    V_0_sigma_values = [0.1, 1.0, 5.0, 10.0]
    theta_0_coord, theta_1_coord = np.mgrid[-1:2:0.01, -0.5:1.5:0.01]
    theta_coords = np.dstack((theta_0_coord, theta_1_coord))

    fig, axes = plt.subplots(3, len(V_0_sigma_values), figsize=(12, 7))
    for i, V_0_sigma in enumerate(V_0_sigma_values):
        ax_prior, ax_posterior, ax_sampled_predictors = axes[:, i]
        V_0 = V_0_sigma**2 * np.eye(len(theta_0))
        prior_distr = build_prior(theta_0, V_0)
        posterior_distr = build_posterior(theta_0, V_0, X, sigma, y)
        ax_prior.contourf(theta_0_coord, theta_1_coord, prior_distr.pdf(theta_coords))
        ax_posterior.contourf(
            theta_0_coord, theta_1_coord, posterior_distr.pdf(theta_coords)
        )
        for ax in (ax_prior, ax_posterior):
            ax.plot(*theta_true, "o", markersize=12, color="red")
            ax.set_xlabel(r"$\theta_0$", fontsize=12)
            if i == 0:
                ax_prior.set_ylabel(r"$\theta_1$", fontsize=12)
                ax_posterior.set_ylabel(r"$\theta_1$", fontsize=12)
        if V_0_sigma == 0.1:
            ax_prior.set_title(r"$V_0 = 0.01 E$", fontsize=18)
        else:
            ax_prior.set_title(
                r"$V_0 = " + f"{int(V_0_sigma**2)}" + r" E$", fontsize=18
            )

        theta_sampled_collection = sample_predictors(posterior_distr, n=10)
        ax_sampled_predictors.plot(
            X[:, 1], y_noise_free, "-", linewidth=4, color="tab:blue"
        )
        ax_sampled_predictors.plot(X[:, 1], y, "o", markersize=10, color="tab:blue")
        for theta_sampled in theta_sampled_collection:
            ax_sampled_predictors.plot(
                X[:, 1],
                X @ theta_sampled,
                "-",
                linewidth=2,
                color="tab:orange",
                alpha=0.5,
            )

        ax_sampled_predictors.grid()
        ax_sampled_predictors.set_xlabel(r"$x$", fontsize=12)
        ax_sampled_predictors.set_ylabel(r"$y$", fontsize=12)
    fig.tight_layout()
    fig.savefig("posterior_distribution_and_sampled_predictors_wrt_v_0.png", dpi=150)
    plt.show()


def plot_posterior_distribution_and_sampled_predictors_wrt_n(sigma, theta_true, x_lims):
    theta_0 = np.array([0.0, 0.0])
    V_0_sigma = 1.0
    V_0 = V_0_sigma**2 * np.eye(len(theta_0))
    n_values = [5, 10, 50, 100]
    theta_0_coord, theta_1_coord = np.mgrid[-1.5:2:0.01, 0.5:1.5:0.01]
    theta_coords = np.dstack((theta_0_coord, theta_1_coord))

    fig, axes = plt.subplots(2, len(n_values), figsize=(12, 6))
    for i, n in enumerate(n_values):
        ax_posterior, ax_sampled_predictors = axes[:, i]
        X, y, y_noise_free = generate_dataset(
            n=n, sigma=sigma, theta=theta_true, x_lims=x_lims
        )
        posterior_distr = build_posterior(theta_0, V_0, X, sigma, y)

        ax_posterior.contourf(
            theta_0_coord, theta_1_coord, posterior_distr.pdf(theta_coords)
        )
        ax_posterior.plot(*theta_true, "o", markersize=12, color="red")
        ax_posterior.set_xlabel(r"$\theta_0$", fontsize=12)
        ax_posterior.set_ylabel(r"$\theta_1$", fontsize=12)
        ax_posterior.set_title(r"$N = " + f"{n}" + r"$", fontsize=18)

        theta_sampled_collection = sample_predictors(posterior_distr, n=10)
        ax_sampled_predictors.plot(
            X[:, 1], y_noise_free, "-", linewidth=4, color="tab:blue"
        )
        ax_sampled_predictors.plot(X[:, 1], y, "o", markersize=10, color="tab:blue")
        for theta_sampled in theta_sampled_collection:
            ax_sampled_predictors.plot(
                X[:, 1],
                X @ theta_sampled,
                "-",
                linewidth=2,
                color="tab:orange",
                alpha=0.5,
            )

        ax_sampled_predictors.grid()
        ax_sampled_predictors.set_xlabel(r"$x$", fontsize=12)
        ax_sampled_predictors.set_ylabel(r"$y$", fontsize=12)
    fig.tight_layout()
    fig.savefig("posterior_distribution_and_sampled_predictors_wrt_n.png", dpi=150)
    plt.show()


def plot_aposterior_predictive_distribution(sigma_value, theta_true, x_lims):
    X_data, y_data, y_noise_free_data = generate_dataset(
        n=n, sigma=sigma_value, theta=theta_true, x_lims=x_lims
    )
    x_for_sampling = np.linspace(x_lims[0], 2 * x_lims[1], 20)

    with pymc3.Model() as model:
        sigma = pymc3.HalfCauchy("sigma", beta=2)
        theta_0 = pymc3.Normal("theta_0", 0, sigma=5)
        theta_1 = pymc3.Normal("theta_1", 0, sigma=5)
        x = pymc3.Data("x", X_data[:, 1])
        mu = pymc3.Deterministic("mu", theta_0 + theta_1 * x)
        pymc3.Normal("y", mu=mu, sigma=sigma, observed=y_data)
        trace = pymc3.sample(3000, cores=2)
        pymc3.set_data({"x": x_for_sampling})
        post_pred_trace = pymc3.sample_posterior_predictive(
            trace, var_names=["y", "mu"]
        )

    x_lims_extrapol = (x_lims[1], 2 * x_lims[1])
    X_data_extrapol, y_data_extrapol, y_noise_free_data_extrapol = generate_dataset(
        n=n, sigma=sigma_value, theta=theta_true, x_lims=x_lims_extrapol
    )
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(X_data[:, 1], y_noise_free_data, color="tab:blue")
    ax.plot(
        [X_data_extrapol[:, 1].min(), X_data_extrapol[:, 1].max()],
        [y_noise_free_data_extrapol.min(), y_noise_free_data_extrapol.max()],
        "--",
        color="tab:blue",
    )
    ax.plot(X_data[:, 1], y_data, "o", markersize=8, color="tab:blue")
    arviz.plot_hdi(x=x_for_sampling, y=post_pred_trace["y"], hdi_prob=0.66, ax=ax)
    ax_ylims = ax.get_ylim()
    ax.fill_betweenx(
        y=ax_ylims,
        x1=x_lims_extrapol[0],
        x2=x_lims_extrapol[1],
        color="gray",
        alpha=0.5,
    )
    ax.text(x=12.5, y=0, s="Extrapolation", fontdict=dict(fontsize=16))

    ax.set_ylim(ax_ylims)
    ax.set_xlabel(r"x", fontsize=12)
    ax.set_ylabel(r"y(x)", fontsize=12)
    ax.grid()
    fig.tight_layout()
    fig.savefig("posterior_predictive.svg")
    plt.show()
    print("qwer")


def generate_dataset(n: int, sigma: float, theta, x_lims: tuple[float, float]):
    X = np.ones((n, 2))
    X[:, 1] = np.random.uniform(*x_lims, size=n)
    y_noise_free = X @ theta
    y = y_noise_free + np.random.normal(loc=0.0, scale=sigma, size=n)
    return X, y, y_noise_free


if __name__ == "__main__":
    np.random.seed(43)

    n = 10
    sigma = 2.0
    x_lims = (0.0, 10.0)
    theta = np.array([1.0, 1.0])
    X, y, y_noise_free = generate_dataset(n=n, sigma=sigma, theta=theta, x_lims=x_lims)
    # plot_posterior_distribution_and_sampled_predictors_wrt_v_0(
    #    X=X, y=y, y_noise_free=y_noise_free, sigma=sigma, theta_true=theta
    # )
    # plot_posterior_distribution_and_sampled_predictors_wrt_n(
    #    sigma=sigma, theta_true=theta, x_lims=x_lims
    # )
    plot_aposterior_predictive_distribution(
        sigma_value=sigma, theta_true=theta, x_lims=x_lims
    )
