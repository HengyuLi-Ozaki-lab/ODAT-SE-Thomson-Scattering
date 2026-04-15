#!/usr/bin/env python3
"""
ODAT-SE test using real LHD Thomson scattering parameters.

Uses Te and ne values from LHD Shot #175916 (t=36300.1 ms) at 5 radial
positions covering the full plasma profile. For each point:
  1. Generates synthetic polychromator signals from the real Te, ne
  2. Adds Poisson + readout noise (realistic noise model)
  3. Runs Nelder-Mead and PAMC to recover Te, ne
  4. Compares inversion results with the LHD analyzed values

Data source: LHD Thomson scattering analyzed data
  DOI: 10.57451/lhd.thomson.175916.1
"""

import sys
import os
import time
import shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import odatse
import odatse.util.toml
import odatse.solver.function

from thomson_model import (
    compute_channel_signals, make_objective_function,
    FILTER_CENTERS, N_CHANNELS, LAMBDA_LASER
)


# ============================================================
# Realistic noise model based on LHD Thomson scattering system
# ============================================================
PHOTON_SCALE = 5000.0    # signal -> photon count scale
READOUT_NOISE = 3.0      # readout noise in photoelectrons
# Stray light: strong near laser wavelength (1064nm), Ch3 at 1020nm is closest
STRAY_LIGHT = np.array([0.0, 0.001, 0.05, 0.005, 0.0])


def generate_realistic_data(Te_true, ne_true, seed=None):
    """Generate synthetic polychromator data with realistic noise."""
    if seed is not None:
        np.random.seed(seed)

    true_signals = compute_channel_signals(Te_true, ne_true)

    # Realistic noise: Poisson (signal + stray light) + readout
    N_signal = true_signals * PHOTON_SCALE
    N_stray = STRAY_LIGHT * PHOTON_SCALE
    N_total = N_signal + N_stray

    # Poisson realization + Gaussian readout
    observed_counts = np.random.poisson(np.maximum(N_total, 0)) + \
                      READOUT_NOISE * np.random.randn(N_CHANNELS)
    observed_signals = observed_counts / PHOTON_SCALE

    # Uncertainty: sqrt(signal + stray + readout^2) / scale
    sigma = np.sqrt(np.maximum(N_total, 1) + READOUT_NOISE**2) / PHOTON_SCALE

    return true_signals, observed_signals, sigma


def run_single_point(Te_true, ne_true, label, seed=42):
    """Run Nelder-Mead and PAMC for a single (Te, ne) point."""
    base_dir = os.path.dirname(os.path.abspath(__file__))

    true_signals, observed, sigma = generate_realistic_data(Te_true, ne_true, seed=seed)
    objective_fn = make_objective_function(observed, sigma)

    # Quick sanity check
    chi2_at_true = objective_fn(np.array([Te_true, ne_true]))

    results = {"label": label, "Te_true": Te_true, "ne_true": ne_true,
               "chi2_at_true": chi2_at_true, "observed": observed, "sigma": sigma}

    # --- Nelder-Mead ---
    algo_dir = os.path.join(base_dir, "minsearch")
    orig_dir = os.getcwd()
    os.chdir(algo_dir)
    output_dir = os.path.join(algo_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Adjust initial point and scale based on expected parameter range
    inp = odatse.util.toml.load("input.toml")
    inp["algorithm"]["param"]["initial_list"] = [Te_true * 0.5, ne_true * 0.5]
    inp["algorithm"]["minimize"]["initial_scale_list"] = [Te_true * 0.3, ne_true * 0.3]
    info = odatse.Info(inp)

    solver = odatse.solver.function.Solver(info)
    solver.set_function(objective_fn)
    runner = odatse.Runner(solver, info)
    alg_module = odatse.algorithm.choose_algorithm("minsearch")
    alg = alg_module.Algorithm(info, runner)

    t0 = time.perf_counter()
    result_nm = alg.main()
    t_nm = time.perf_counter() - t0

    os.chdir(orig_dir)
    results["minsearch"] = {
        "Te": result_nm["x"][0], "ne": result_nm["x"][1],
        "chi2": result_nm["fx"], "time": t_nm,
        "Te_err%": abs(result_nm["x"][0] - Te_true) / Te_true * 100,
        "ne_err%": abs(result_nm["x"][1] - ne_true) / ne_true * 100,
    }

    # --- PAMC ---
    algo_dir = os.path.join(base_dir, "pamc")
    os.chdir(algo_dir)
    output_dir = os.path.join(algo_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    inp = odatse.util.toml.load("input.toml")
    # Adjust step size based on parameter magnitude
    inp["algorithm"]["param"]["step_list"] = [Te_true * 0.1, ne_true * 0.1]
    info = odatse.Info(inp)

    solver = odatse.solver.function.Solver(info)
    solver.set_function(objective_fn)
    runner = odatse.Runner(solver, info)
    alg_module = odatse.algorithm.choose_algorithm("pamc")
    alg = alg_module.Algorithm(info, runner)

    t0 = time.perf_counter()
    result_pamc = alg.main()
    t_pamc = time.perf_counter() - t0

    # Read posterior samples at lowest T
    pamc_output = os.path.join(algo_dir, "output", "0")
    result_files = sorted([f for f in os.listdir(pamc_output) if f.startswith("result_T")])
    last_file = os.path.join(pamc_output, result_files[-1])
    samples = np.loadtxt(last_file)
    Te_samples = samples[:, 4]
    ne_samples = samples[:, 5]
    weights = samples[:, 6] if samples.shape[1] > 6 else np.ones(len(samples))

    Te_mean = np.average(Te_samples, weights=weights)
    ne_mean = np.average(ne_samples, weights=weights)
    Te_std = np.sqrt(np.average((Te_samples - Te_mean)**2, weights=weights))
    ne_std = np.sqrt(np.average((ne_samples - ne_mean)**2, weights=weights))

    os.chdir(orig_dir)
    results["pamc"] = {
        "Te_best": result_pamc["x"][0], "ne_best": result_pamc["x"][1],
        "chi2": result_pamc["fx"], "time": t_pamc,
        "Te_mean": Te_mean, "ne_mean": ne_mean,
        "Te_std": Te_std, "ne_std": ne_std,
        "Te_err%": abs(Te_mean - Te_true) / Te_true * 100,
        "ne_err%": abs(ne_mean - ne_true) / ne_true * 100,
    }

    return results


def print_results_table(all_results):
    """Print formatted comparison table."""
    print("\n" + "=" * 110)
    print("ODAT-SE INVERSION RESULTS vs LHD ANALYZED DATA  (Shot #175916, t=36300.1 ms)")
    print("=" * 110)

    # ne_true is already in 1e19 m^-3 units
    # Nelder-Mead results
    print("\n--- Nelder-Mead (point estimate) ---")
    print(f"{'Region':<14} | {'Te_LHD':>7} {'Te_inv':>7} {'err%':>6} | "
          f"{'ne_LHD':>7} {'ne_inv':>7} {'err%':>6} | {'chi2':>7} {'time':>6}")
    print("-" * 95)
    for r in all_results:
        m = r["minsearch"]
        print(f"{r['label']:<14} | {r['Te_true']:7.0f} {m['Te']:7.0f} {m['Te_err%']:5.1f}% | "
              f"{r['ne_true']:7.3f} {m['ne']:7.3f} {m['ne_err%']:5.1f}% | {m['chi2']:7.2f} {m['time']:5.2f}s")

    # PAMC results
    print("\n--- PAMC (posterior mean +/- std) ---")
    print(f"{'Region':<14} | {'Te_LHD':>7} {'Te_mean':>8}+/-{'Te_std':>6} {'err%':>6} | "
          f"{'ne_LHD':>7} {'ne_mean':>8}+/-{'ne_std':>6} {'err%':>6}")
    print("-" * 105)
    for r in all_results:
        p = r["pamc"]
        print(f"{r['label']:<14} | {r['Te_true']:7.0f} {p['Te_mean']:8.1f}+/-{p['Te_std']:6.1f} {p['Te_err%']:5.1f}% | "
              f"{r['ne_true']:7.3f} {p['ne_mean']:8.4f}+/-{p['ne_std']:6.4f} {p['ne_err%']:5.1f}%")


def plot_results(all_results):
    """Generate comparison plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = [r["label"] for r in all_results]
    Te_true = [r["Te_true"] for r in all_results]
    ne_true = [r["ne_true"] / 1000 for r in all_results]  # 1e16 -> 1e19

    Te_nm = [r["minsearch"]["Te"] for r in all_results]
    ne_nm = [r["minsearch"]["ne"] / 1000 for r in all_results]

    Te_pamc = [r["pamc"]["Te_mean"] for r in all_results]
    Te_pamc_err = [r["pamc"]["Te_std"] for r in all_results]
    ne_pamc = [r["pamc"]["ne_mean"] / 1000 for r in all_results]
    ne_pamc_err = [r["pamc"]["ne_std"] / 1000 for r in all_results]

    x = np.arange(len(labels))
    w = 0.25

    # Te comparison
    ax = axes[0]
    ax.bar(x - w, Te_true, w, label="LHD analyzed", color="gray", alpha=0.7)
    ax.bar(x, Te_nm, w, label="Nelder-Mead", color="steelblue")
    ax.bar(x + w, Te_pamc, w, yerr=Te_pamc_err, label="PAMC", color="darkorange", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$T_e$ (eV)")
    ax.set_title(r"Electron Temperature $T_e$")
    ax.legend(fontsize=9)

    # ne comparison
    ax = axes[1]
    ax.bar(x - w, ne_true, w, label="LHD analyzed", color="gray", alpha=0.7)
    ax.bar(x, ne_nm, w, label="Nelder-Mead", color="steelblue")
    ax.bar(x + w, ne_pamc, w, yerr=ne_pamc_err, label="PAMC", color="darkorange", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)")
    ax.set_title(r"Electron Density $n_e$")
    ax.legend(fontsize=9)

    fig.suptitle("ODAT-SE Inversion vs LHD Thomson Data (Shot #175916)", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "analysis", "figures", "lhd_comparison.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {fig_path}")


def main():
    print("=" * 70)
    print("ODAT-SE Test with Real LHD Thomson Scattering Data")
    print("LHD Shot #175916, t = 36300.1 ms")
    print("=" * 70)

    # Load test points
    npz = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lhd_test_points.npz"),
                  allow_pickle=True)
    selected = npz["selected"]
    labels = list(npz["labels"])

    # ne is in units of 1e16 m^-3 in the LHD data
    # Our forward model uses ne in 1e19 m^-3
    # Need to be consistent: pass ne in the same unit as the forward model expects

    all_results = []
    for i, (row, label) in enumerate(zip(selected, labels)):
        Te_true = row[2]           # eV
        ne_true_1e16 = row[4]      # 1e16 m^-3
        ne_true_1e19 = ne_true_1e16 / 1000.0  # convert to 1e19 m^-3

        print(f"\n{'='*60}")
        print(f"Point {i+1}/5: {label}")
        print(f"  LHD values: Te = {Te_true:.0f} eV, ne = {ne_true_1e19:.3f} x 1e19 m^-3")
        print(f"{'='*60}")

        result = run_single_point(Te_true, ne_true_1e19, label, seed=42 + i)
        all_results.append(result)

    print_results_table(all_results)
    plot_results(all_results)

    # Save all results
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "lhd_test_results.npz")
    np.savez(results_path, results=all_results)
    print(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
