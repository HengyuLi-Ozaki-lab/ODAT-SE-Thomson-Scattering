#!/usr/bin/env python3
"""
Analysis and visualization of Thomson scattering parameter inversion results.

Generates:
1. Chi-squared landscape from Grid Search (mapper)
2. Posterior distributions from PAMC
3. REMC sampling at different temperatures
4. Algorithm convergence comparison
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# True parameters
TE_TRUE = 500.0
NE_TRUE = 3.0

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUTPUT_DIR = os.path.join(BASE_DIR, "analysis", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_chi2_landscape():
    """Plot the chi-squared landscape from Grid Search (mapper) results."""
    colormap_file = os.path.join(BASE_DIR, "mapper", "output", "ColorMap.txt")
    if not os.path.exists(colormap_file):
        print("Skipping chi2 landscape: mapper output not found")
        return

    data = np.loadtxt(colormap_file)
    Te = data[:, 0]
    ne = data[:, 1]
    chi2 = data[:, 2]

    # Reshape into 2D grid
    Te_unique = np.unique(Te)
    ne_unique = np.unique(ne)
    nTe = len(Te_unique)
    nne = len(ne_unique)
    chi2_grid = chi2.reshape(nTe, nne)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    pcm = ax.pcolormesh(
        Te_unique, ne_unique, chi2_grid.T,
        norm=LogNorm(vmin=1, vmax=1e4),
        cmap="viridis_r", shading="nearest"
    )
    plt.colorbar(pcm, ax=ax, label=r"$\chi^2$")

    # Mark true value
    ax.plot(TE_TRUE, NE_TRUE, "r*", markersize=15, label=f"True ({TE_TRUE}, {NE_TRUE})")

    # Mark best point
    best_idx = np.argmin(chi2)
    ax.plot(Te[best_idx], ne[best_idx], "wx", markersize=12, markeredgewidth=2,
            label=f"Best ({Te[best_idx]:.0f}, {ne[best_idx]:.2f})")

    # Contour lines
    levels = [2, 5, 10, 50, 100, 500]
    cs = ax.contour(Te_unique, ne_unique, chi2_grid.T, levels=levels,
                    colors="white", linewidths=0.5, linestyles="--")
    ax.clabel(cs, fmt="%.0f", fontsize=8, colors="white")

    ax.set_xlabel(r"$T_e$ (eV)", fontsize=12)
    ax.set_ylabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)", fontsize=12)
    ax.set_title(r"$\chi^2$ Landscape (Grid Search)", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "chi2_landscape.png"), dpi=150)
    plt.close(fig)
    print("Saved: chi2_landscape.png")


def plot_pamc_posterior():
    """Plot posterior distributions from PAMC results."""
    # Find result files at the lowest temperature (highest beta = most constrained)
    pamc_dir = os.path.join(BASE_DIR, "pamc", "output", "0")
    if not os.path.exists(pamc_dir):
        print("Skipping posterior: PAMC output not found")
        return

    # Find the last temperature file (lowest T = tightest posterior)
    result_files = sorted([f for f in os.listdir(pamc_dir) if f.startswith("result_T")])
    if not result_files:
        print("Skipping posterior: no result_T files found")
        return

    # The last temperature index has the tightest posterior
    last_file = os.path.join(pamc_dir, result_files[-1])
    data = np.loadtxt(last_file)

    # Columns: step, walker, T, fx, x1(Te), x2(ne), weight, ancestor
    Te_samples = data[:, 4]
    ne_samples = data[:, 5]
    weights = data[:, 6] if data.shape[1] > 6 else np.ones(len(data))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # 1D marginal: Te
    ax = axes[0]
    ax.hist(Te_samples, bins=50, density=True, alpha=0.7, color="steelblue",
            weights=weights, label="Posterior")
    ax.axvline(TE_TRUE, color="red", linestyle="--", linewidth=2, label=f"True ({TE_TRUE})")
    Te_mean = np.average(Te_samples, weights=weights)
    ax.axvline(Te_mean, color="orange", linestyle="-", linewidth=1.5,
               label=f"Mean ({Te_mean:.1f})")
    ax.set_xlabel(r"$T_e$ (eV)", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(r"Posterior: $T_e$", fontsize=13)
    ax.legend(fontsize=9)

    # 1D marginal: ne
    ax = axes[1]
    ax.hist(ne_samples, bins=50, density=True, alpha=0.7, color="darkorange",
            weights=weights, label="Posterior")
    ax.axvline(NE_TRUE, color="red", linestyle="--", linewidth=2, label=f"True ({NE_TRUE})")
    ne_mean = np.average(ne_samples, weights=weights)
    ax.axvline(ne_mean, color="steelblue", linestyle="-", linewidth=1.5,
               label=f"Mean ({ne_mean:.2f})")
    ax.set_xlabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)", fontsize=12)
    ax.set_ylabel("Probability density", fontsize=12)
    ax.set_title(r"Posterior: $n_e$", fontsize=13)
    ax.legend(fontsize=9)

    # 2D joint posterior
    ax = axes[2]
    ax.scatter(Te_samples, ne_samples, s=1, alpha=0.3, c="steelblue")
    ax.plot(TE_TRUE, NE_TRUE, "r*", markersize=15, zorder=10, label="True value")
    ax.set_xlabel(r"$T_e$ (eV)", fontsize=12)
    ax.set_ylabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)", fontsize=12)
    ax.set_title("Joint Posterior (PAMC)", fontsize=13)
    ax.legend(fontsize=9)

    fig.suptitle("PAMC Posterior Distributions (Lowest Temperature)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "posterior_pamc.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: posterior_pamc.png")


def plot_exchange_temperatures():
    """Plot REMC samples at different temperatures."""
    exchange_dir = os.path.join(BASE_DIR, "exchange", "output")
    if not os.path.exists(exchange_dir):
        print("Skipping REMC plot: exchange output not found")
        return

    result_files = sorted([f for f in os.listdir(exchange_dir) if f.startswith("result_T")])
    if not result_files:
        print("Skipping REMC plot: no result_T files found")
        return

    n_temps = len(result_files)
    # Select a few representative temperatures
    indices = [0, n_temps // 3, 2 * n_temps // 3, n_temps - 1]
    indices = [i for i in indices if i < n_temps]

    fig, axes = plt.subplots(1, len(indices), figsize=(4 * len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        filepath = os.path.join(exchange_dir, result_files[idx])
        with open(filepath, "r") as f:
            first_line = f.readline()
        T_val = float(first_line.split("=")[1].strip()) if "=" in first_line else idx

        data = np.loadtxt(filepath)
        # Columns: step, walker, T, fx, x1(Te), x2(ne)
        Te_samples = data[:, 3]
        ne_samples = data[:, 4]

        ax.scatter(Te_samples, ne_samples, s=1, alpha=0.3)
        ax.plot(TE_TRUE, NE_TRUE, "r*", markersize=12, zorder=10)
        ax.set_xlabel(r"$T_e$ (eV)")
        ax.set_ylabel(r"$n_e$")
        ax.set_title(f"T = {T_val:.2f}")
        ax.set_xlim([0, 5000])
        ax.set_ylim([0, 10])

    fig.suptitle("REMC Samples at Different Temperatures", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "exchange_temperatures.png"), dpi=150)
    plt.close(fig)
    print("Saved: exchange_temperatures.png")


def plot_minsearch_convergence():
    """Plot Nelder-Mead convergence trace."""
    simplex_file = os.path.join(BASE_DIR, "minsearch", "output", "0", "SimplexData.txt")
    if not os.path.exists(simplex_file):
        print("Skipping minsearch convergence: output not found")
        return

    data = np.loadtxt(simplex_file)
    steps = data[:, 0]
    Te_vals = data[:, 1]
    ne_vals = data[:, 2]
    fx_vals = data[:, 3]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].semilogy(steps, fx_vals, "b-o", markersize=3)
    axes[0].axhline(fx_vals[-1], color="gray", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\chi^2$")
    axes[0].set_title("Objective Function Convergence")

    axes[1].plot(steps, Te_vals, "b-o", markersize=3)
    axes[1].axhline(TE_TRUE, color="red", linestyle="--", label=f"True ({TE_TRUE})")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$T_e$ (eV)")
    axes[1].set_title(r"$T_e$ Convergence")
    axes[1].legend()

    axes[2].plot(steps, ne_vals, "b-o", markersize=3)
    axes[2].axhline(NE_TRUE, color="red", linestyle="--", label=f"True ({NE_TRUE})")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)")
    axes[2].set_title(r"$n_e$ Convergence")
    axes[2].legend()

    fig.suptitle("Nelder-Mead Convergence", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "minsearch_convergence.png"), dpi=150)
    plt.close(fig)
    print("Saved: minsearch_convergence.png")


def plot_pamc_free_energy():
    """Plot PAMC free energy vs inverse temperature."""
    fx_file = os.path.join(BASE_DIR, "pamc", "output", "fx.txt")
    if not os.path.exists(fx_file):
        print("Skipping free energy plot: PAMC fx.txt not found")
        return

    data = np.loadtxt(fx_file)
    # Columns: beta, <f(x)>, stderr(f(x)), nreplica, logZ, acceptance_ratio
    beta = data[:, 0]
    fmean = data[:, 1]
    ferr = data[:, 2]
    logZ = data[:, 4]
    acc = data[:, 5]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].errorbar(beta, fmean, yerr=ferr, fmt="o-", markersize=4, capsize=3)
    axes[0].set_xlabel(r"$\beta$ (inverse temperature)")
    axes[0].set_ylabel(r"$\langle f(\theta) \rangle$")
    axes[0].set_title("Mean Objective Function")
    axes[0].set_xscale("log")

    axes[1].plot(beta, logZ, "s-", markersize=4, color="darkorange")
    axes[1].set_xlabel(r"$\beta$")
    axes[1].set_ylabel(r"$\ln(Z/Z_0)$")
    axes[1].set_title("Log Partition Function (Free Energy)")
    axes[1].set_xscale("log")

    axes[2].plot(beta, acc, "^-", markersize=4, color="green")
    axes[2].set_xlabel(r"$\beta$")
    axes[2].set_ylabel("Acceptance Ratio")
    axes[2].set_title("MCMC Acceptance Ratio")
    axes[2].set_xscale("log")
    axes[2].set_ylim([0, 1])

    fig.suptitle("PAMC Diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "pamc_diagnostics.png"), dpi=150)
    plt.close(fig)
    print("Saved: pamc_diagnostics.png")

    # Print free energy
    print(f"  Final logZ = {logZ[-1]:.4f}")


def main():
    print("=" * 60)
    print("Generating analysis plots...")
    print("=" * 60)

    plot_chi2_landscape()
    plot_minsearch_convergence()
    plot_exchange_temperatures()
    plot_pamc_posterior()
    plot_pamc_free_energy()

    print(f"\nAll figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
