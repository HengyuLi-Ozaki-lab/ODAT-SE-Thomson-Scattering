#!/usr/bin/env python3
"""
Comprehensive ODAT-SE Thomson Scattering Test Suite.

Runs all tests using LHD Shot #175916 Thomson scattering data:
  1. Algorithm benchmark (5 algorithms on mid-radius point)
  2. Multi-point inversion (5 radial positions)
  3. Full profile scan (all good spatial points)
  4. Bayesian model selection (Maxwell vs Kappa)
  5. Performance benchmarks (noise, Te coverage, PAMC scalability)
  6. Figure generation

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py benchmark    # Run only algorithm benchmark
    python run_tests.py profile      # Run only profile scan
    python run_tests.py model        # Run only model selection
    python run_tests.py perf         # Run only performance benchmarks
    python run_tests.py figures      # Regenerate figures only
"""

import sys
import os
import time
import shutil
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import odatse
import odatse.util.toml
import odatse.solver.function

from thomson_model import (
    compute_channel_signals, make_objective_function,
    make_kappa_objective_function, N_CHANNELS, FILTER_CENTERS,
    LAMBDA_LASER, thomson_spectrum, WAVELENGTHS
)

# ============================================================
# Configuration
# ============================================================
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
LHD_DATA_FILE = os.path.join(DATA_DIR, "thomson_175916.txt")

# Realistic noise model
PHOTON_SCALE = 5000.0
READOUT_NOISE = 3.0
STRAY_LIGHT = np.array([0.0, 0.001, 0.05, 0.005, 0.0])


# ============================================================
# Utilities
# ============================================================
def generate_realistic_data(Te_true, ne_true, seed=None):
    """Generate synthetic polychromator data with realistic noise."""
    if seed is not None:
        np.random.seed(seed)
    true_signals = compute_channel_signals(Te_true, ne_true)
    N_signal = true_signals * PHOTON_SCALE
    N_stray = STRAY_LIGHT * PHOTON_SCALE
    N_total = N_signal + N_stray
    observed_counts = np.random.poisson(np.maximum(N_total, 0)) + \
                      READOUT_NOISE * np.random.randn(N_CHANNELS)
    observed_signals = observed_counts / PHOTON_SCALE
    sigma = np.sqrt(np.maximum(N_total, 1) + READOUT_NOISE**2) / PHOTON_SCALE
    return true_signals, observed_signals, sigma


def load_lhd_profile():
    """Load and filter LHD Thomson data, return best time slice."""
    data = np.loadtxt(LHD_DATA_FILE, delimiter=',', comments='#')
    t_target = 36300.1
    mask = (np.abs(data[:, 0] - t_target) < 0.5) & \
           (data[:, 3] < data[:, 2] * 0.3) & \
           (data[:, 2] > 20) & (data[:, 4] > 10) & (data[:, 5] > 0)
    profile = data[mask]
    profile = profile[profile[:, 1].argsort()]
    return profile


def select_test_points(profile):
    """Select 5 representative radial positions."""
    Te = profile[:, 2]
    targets = {
        'Edge':        np.argmin(np.abs(Te - 150)),
        'Pedestal':    np.argmin(np.abs(Te - 500)),
        'Mid-radius':  np.argmin(np.abs(Te - 1500)),
        'Near-core':   np.argmin(np.abs(Te - 3000)),
        'Core':        np.argmin(np.abs(Te - 5000)),
    }
    return {label: profile[idx] for label, idx in targets.items()}


def run_odatse(config_file, objective_fn, overrides=None):
    """Run a single ODAT-SE algorithm. Returns (result, elapsed_seconds)."""
    inp = odatse.util.toml.load(config_file)
    if overrides:
        for key_path, value in overrides.items():
            parts = key_path.split(".")
            d = inp
            for p in parts[:-1]:
                d = d[p]
            d[parts[-1]] = value

    # Ensure output_dir is absolute
    out_dir = os.path.join(BASE_DIR, inp["base"].get("output_dir", "output"))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    inp["base"]["output_dir"] = out_dir

    orig_dir = os.getcwd()
    os.chdir(BASE_DIR)
    try:
        info = odatse.Info(inp)
        solver = odatse.solver.function.Solver(info)
        solver.set_function(objective_fn)
        runner = odatse.Runner(solver, info)
        alg_module = odatse.algorithm.choose_algorithm(info.algorithm["name"])
        alg = alg_module.Algorithm(info, runner)
        t0 = time.perf_counter()
        result = alg.main()
        elapsed = time.perf_counter() - t0
        return result, elapsed
    finally:
        os.chdir(orig_dir)


# ============================================================
# Test 1: Algorithm Benchmark
# ============================================================
def test_algorithm_benchmark(Te_true, ne_true):
    """Run all 5 algorithms on a single (Te, ne) point."""
    print("\n" + "=" * 70)
    print("TEST 1: Algorithm Benchmark")
    print(f"  True: Te = {Te_true:.0f} eV, ne = {ne_true:.3f} x10^19 m^-3")
    print("=" * 70)

    _, obs, sig = generate_realistic_data(Te_true, ne_true, seed=44)
    obj_fn = make_objective_function(obs, sig)

    algos = {
        "minsearch": {"algorithm.param.initial_list": [Te_true * 0.5, ne_true * 0.5],
                      "algorithm.minimize.initial_scale_list": [Te_true * 0.3, ne_true * 0.3]},
        "mapper": {},
        "bayes": {},
        "exchange": {"algorithm.param.step_list": [Te_true * 0.1, ne_true * 0.1]},
        "pamc": {"algorithm.param.step_list": [Te_true * 0.1, ne_true * 0.1]},
    }

    results = {}
    for algo_name, overrides in algos.items():
        config_file = os.path.join(CONFIG_DIR, f"{algo_name}.toml")
        out_dir_name = f"results/{algo_name}"
        all_overrides = {"base.output_dir": out_dir_name}
        all_overrides.update(overrides)

        print(f"\n  Running {algo_name}...", end=" ", flush=True)
        try:
            result, elapsed = run_odatse(config_file, obj_fn, all_overrides)
            if result and "x" in result:
                Te_inv, ne_inv = result["x"][0], result["x"][1]
                fx = float(result.get("fx", np.nan))
                if isinstance(fx, np.ndarray):
                    fx = float(fx.flat[0])
                results[algo_name] = {"Te": Te_inv, "ne": ne_inv, "chi2": fx, "time": elapsed}
                print(f"Te={Te_inv:.0f}, ne={ne_inv:.3f}, chi2={fx:.2f}, time={elapsed:.2f}s")
            else:
                results[algo_name] = {"Te": np.nan, "ne": np.nan, "chi2": np.nan, "time": elapsed}
                print(f"completed in {elapsed:.2f}s (no x in result)")
        except Exception as e:
            print(f"FAILED: {e}")
            results[algo_name] = None

    # Summary table
    print(f"\n{'Algorithm':<15} {'Te (eV)':<10} {'ne (1e19)':<10} {'chi2':<10} {'Time (s)':<10}")
    print("-" * 55)
    for name, r in results.items():
        if r:
            print(f"{name:<15} {r['Te']:<10.1f} {r['ne']:<10.4f} {r['chi2']:<10.2f} {r['time']:<10.3f}")

    return results


# ============================================================
# Test 2: Multi-point Inversion
# ============================================================
def test_multipoint_inversion(test_points):
    """Run Nelder-Mead and PAMC at 5 radial positions."""
    print("\n" + "=" * 70)
    print("TEST 2: Multi-point Inversion (5 radial positions)")
    print("=" * 70)

    all_results = []
    for i, (label, row) in enumerate(test_points.items()):
        Te_true = row[2]
        ne_true = row[4] / 1000.0  # 1e16 -> 1e19

        print(f"\n  Point {i+1}/5: {label} (Te={Te_true:.0f}, ne={ne_true:.3f})")

        _, obs, sig = generate_realistic_data(Te_true, ne_true, seed=42 + i)
        obj_fn = make_objective_function(obs, sig)

        point_result = {"label": label, "Te_true": Te_true, "ne_true": ne_true}

        # Nelder-Mead
        config = os.path.join(CONFIG_DIR, "minsearch.toml")
        overrides = {
            "base.output_dir": f"results/multipoint/{label}/minsearch",
            "algorithm.param.initial_list": [max(Te_true * 0.5, 20), max(ne_true * 0.5, 0.1)],
            "algorithm.minimize.initial_scale_list": [max(Te_true * 0.3, 10), max(ne_true * 0.3, 0.05)],
        }
        result, elapsed = run_odatse(config, obj_fn, overrides)
        point_result["minsearch"] = {
            "Te": result["x"][0], "ne": result["x"][1], "chi2": result["fx"], "time": elapsed,
            "Te_err%": abs(result["x"][0] - Te_true) / Te_true * 100,
            "ne_err%": abs(result["x"][1] - ne_true) / ne_true * 100,
        }
        print(f"    NM: Te={result['x'][0]:.0f} ({point_result['minsearch']['Te_err%']:.1f}%), "
              f"ne={result['x'][1]:.3f} ({point_result['minsearch']['ne_err%']:.1f}%)")

        # PAMC
        config = os.path.join(CONFIG_DIR, "pamc.toml")
        overrides = {
            "base.output_dir": f"results/multipoint/{label}/pamc",
            "algorithm.param.step_list": [max(Te_true * 0.1, 15), max(ne_true * 0.1, 0.03)],
        }
        result, elapsed = run_odatse(config, obj_fn, overrides)

        # Read posterior
        pamc_out = os.path.join(BASE_DIR, f"results/multipoint/{label}/pamc", "0")
        rf = sorted([f for f in os.listdir(pamc_out) if f.startswith("result_T")])
        samples = np.loadtxt(os.path.join(pamc_out, rf[-1]))
        Te_s, ne_s, w = samples[:, 4], samples[:, 5], samples[:, 6]
        Te_m, ne_m = np.average(Te_s, weights=w), np.average(ne_s, weights=w)
        Te_std = np.sqrt(np.average((Te_s - Te_m)**2, weights=w))
        ne_std = np.sqrt(np.average((ne_s - ne_m)**2, weights=w))

        point_result["pamc"] = {
            "Te_mean": Te_m, "ne_mean": ne_m, "Te_std": Te_std, "ne_std": ne_std,
            "chi2": result["fx"], "time": elapsed,
            "Te_err%": abs(Te_m - Te_true) / Te_true * 100,
            "ne_err%": abs(ne_m - ne_true) / ne_true * 100,
        }
        print(f"    PAMC: Te={Te_m:.0f}+/-{Te_std:.0f} ({point_result['pamc']['Te_err%']:.1f}%), "
              f"ne={ne_m:.3f}+/-{ne_std:.4f} ({point_result['pamc']['ne_err%']:.1f}%)")

        all_results.append(point_result)

    return all_results


# ============================================================
# Test 3: Full Profile Scan
# ============================================================
def test_profile_scan(profile):
    """Run Nelder-Mead on all good spatial points."""
    print("\n" + "=" * 70)
    print(f"TEST 3: Full Profile Scan ({len(profile)} points)")
    print("=" * 70)

    R_all = profile[:, 1]
    Te_all = profile[:, 2]
    ne_all = profile[:, 4] / 1000.0

    Te_inv, ne_inv = [], []

    for i, (R, Te, ne) in enumerate(zip(R_all, Te_all, ne_all)):
        _, obs, sig = generate_realistic_data(Te, ne, seed=100 + i)
        obj_fn = make_objective_function(obs, sig)
        config = os.path.join(CONFIG_DIR, "minsearch.toml")
        overrides = {
            "base.output_dir": f"results/profile_scan/pt{i}",
            "algorithm.param.initial_list": [max(Te * 0.5, 20), max(ne * 0.5, 0.1)],
            "algorithm.minimize.initial_scale_list": [max(Te * 0.3, 10), max(ne * 0.3, 0.05)],
        }
        result, _ = run_odatse(config, obj_fn, overrides)
        Te_inv.append(result["x"][0])
        ne_inv.append(result["x"][1])
        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{len(profile)} points")

    Te_inv = np.array(Te_inv)
    ne_inv = np.array(ne_inv)
    Te_err = np.abs(Te_inv - Te_all) / Te_all * 100
    ne_err = np.abs(ne_inv - ne_all) / ne_all * 100

    print(f"\n  Te error: median={np.median(Te_err):.1f}%, mean={np.mean(Te_err):.1f}%")
    print(f"  ne error: median={np.median(ne_err):.1f}%, mean={np.mean(ne_err):.1f}%")

    return {"R": R_all, "Te_true": Te_all, "ne_true": ne_all,
            "Te_inv": Te_inv, "ne_inv": ne_inv,
            "Te_err": Te_err, "ne_err": ne_err, "profile": profile}


# ============================================================
# Test 4: Model Selection
# ============================================================
def test_model_selection():
    """Bayesian model selection: Maxwell vs Kappa."""
    print("\n" + "=" * 70)
    print("TEST 4: Bayesian Model Selection (Maxwell vs Kappa)")
    print("=" * 70)

    Te_true, ne_true = 500.0, 3.0
    _, obs, sig = generate_realistic_data(Te_true, ne_true, seed=42)

    results = {}
    for model_name, config_name, fn_maker in [
        ("Maxwell", "model_maxwell.toml", lambda: make_objective_function(obs, sig)),
        ("Kappa", "model_kappa.toml", lambda: make_kappa_objective_function(obs, sig)),
    ]:
        config = os.path.join(CONFIG_DIR, config_name)
        obj_fn = fn_maker()
        overrides = {"base.output_dir": f"results/model_selection/{model_name.lower()}"}
        result, elapsed = run_odatse(config, obj_fn, overrides)

        # Read logZ
        fx_file = os.path.join(BASE_DIR, f"results/model_selection/{model_name.lower()}", "fx.txt")
        logZ = None
        if os.path.exists(fx_file):
            fx_data = np.loadtxt(fx_file)
            logZ = fx_data[-1, 4]

        results[model_name] = {"result": result, "logZ": logZ, "time": elapsed}
        print(f"  {model_name}: chi2={result['fx']:.2f}, logZ={logZ:.2f}, time={elapsed:.1f}s")

    ln_B = results["Maxwell"]["logZ"] - results["Kappa"]["logZ"]
    print(f"\n  Bayes factor: ln(B) = {ln_B:.2f}")
    if abs(ln_B) > 5:
        print(f"  -> Decisive evidence for {'Maxwell' if ln_B > 0 else 'Kappa'}")
    results["ln_B"] = ln_B
    return results


# ============================================================
# Test 5: Performance Benchmarks
# ============================================================
def test_noise_sensitivity(Te_true=1495.0, ne_true=1.577):
    """Test inversion accuracy across SNR levels."""
    print("\n  Noise sensitivity...")
    snr_levels = [5, 10, 15, 20, 30, 50, 100]
    n_trials = 10
    results = {}

    for snr in snr_levels:
        Te_errs, ne_errs = [], []
        for trial in range(n_trials):
            np.random.seed(1000 + trial)
            true_sig = compute_channel_signals(Te_true, ne_true)
            sigma = true_sig / snr
            obs = true_sig + sigma * np.random.randn(N_CHANNELS)
            fn = make_objective_function(obs, sigma)
            config = os.path.join(CONFIG_DIR, "minsearch.toml")
            overrides = {
                "base.output_dir": f"results/perf/noise_snr{snr}_t{trial}",
                "algorithm.param.initial_list": [Te_true * 0.5, ne_true * 0.5],
                "algorithm.minimize.initial_scale_list": [Te_true * 0.3, ne_true * 0.3],
            }
            res, _ = run_odatse(config, fn, overrides)
            Te_errs.append(abs(res["x"][0] - Te_true) / Te_true * 100)
            ne_errs.append(abs(res["x"][1] - ne_true) / ne_true * 100)

        results[snr] = {"Te_mean": np.mean(Te_errs), "Te_std": np.std(Te_errs),
                        "ne_mean": np.mean(ne_errs), "ne_std": np.std(ne_errs)}
        print(f"    SNR={snr:3d}: Te={np.mean(Te_errs):.1f}+/-{np.std(Te_errs):.1f}%, "
              f"ne={np.mean(ne_errs):.1f}+/-{np.std(ne_errs):.1f}%")
    return results


def test_te_coverage(ne_fixed=1.5):
    """Test accuracy across Te range."""
    print("\n  Te coverage scan...")
    Te_scan = [50, 100, 200, 500, 1000, 2000, 3000, 5000, 8000]
    results = {}

    for Te in Te_scan:
        _, obs, sig = generate_realistic_data(Te, ne_fixed, seed=200)
        fn = make_objective_function(obs, sig)
        config = os.path.join(CONFIG_DIR, "minsearch.toml")
        overrides = {
            "base.output_dir": f"results/perf/te_{Te}",
            "algorithm.param.initial_list": [max(Te * 0.5, 20), ne_fixed * 0.5],
            "algorithm.minimize.initial_scale_list": [max(Te * 0.3, 10), ne_fixed * 0.3],
        }
        res, _ = run_odatse(config, fn, overrides)
        Te_inv, ne_inv = res["x"][0], res["x"][1]
        results[Te] = {"Te_inv": Te_inv, "ne_inv": ne_inv, "chi2": res["fx"],
                       "Te_err%": abs(Te_inv - Te) / Te * 100,
                       "ne_err%": abs(ne_inv - ne_fixed) / ne_fixed * 100}
        print(f"    Te={Te:5d}: inv={Te_inv:.0f} ({results[Te]['Te_err%']:.1f}%)")
    return results


def test_pamc_scalability(Te_true=1495.0, ne_true=1.577):
    """Test PAMC with different replica counts."""
    print("\n  PAMC scalability...")
    _, obs, sig = generate_realistic_data(Te_true, ne_true, seed=44)
    fn = make_objective_function(obs, sig)
    replica_counts = [20, 50, 100, 200]
    results = {}

    for nrep in replica_counts:
        config = os.path.join(CONFIG_DIR, "pamc.toml")
        overrides = {
            "base.output_dir": f"results/perf/pamc_rep{nrep}",
            "algorithm.param.step_list": [Te_true * 0.1, ne_true * 0.1],
            "algorithm.pamc.nreplica_per_proc": nrep,
        }
        res, elapsed = run_odatse(config, fn, overrides)

        # Read posterior
        pamc_out = os.path.join(BASE_DIR, f"results/perf/pamc_rep{nrep}", "0")
        rf = sorted([f for f in os.listdir(pamc_out) if f.startswith("result_T")])
        samples = np.loadtxt(os.path.join(pamc_out, rf[-1]))
        Te_s, w = samples[:, 4], samples[:, 6]
        Te_m = np.average(Te_s, weights=w)
        Te_std = np.sqrt(np.average((Te_s - Te_m)**2, weights=w))

        fx_data = np.loadtxt(os.path.join(BASE_DIR, f"results/perf/pamc_rep{nrep}", "fx.txt"))
        logZ = fx_data[-1, 4]

        results[nrep] = {"time": elapsed, "Te_std": Te_std, "logZ": logZ}
        print(f"    nrep={nrep:3d}: time={elapsed:.2f}s, Te_std={Te_std:.1f}, logZ={logZ:.1f}")
    return results


def test_performance():
    """Run all performance benchmarks."""
    print("\n" + "=" * 70)
    print("TEST 5: Performance Benchmarks")
    print("=" * 70)
    noise = test_noise_sensitivity()
    coverage = test_te_coverage()
    scalability = test_pamc_scalability()
    return {"noise": noise, "coverage": coverage, "scalability": scalability}


# ============================================================
# Figure Generation
# ============================================================
def generate_figures(benchmark_res=None, multipoint_res=None, profile_res=None,
                     model_res=None, perf_res=None):
    """Generate all analysis figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.patheffects as pe

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # -- Thomson spectrum + filters --
    from thomson_model import compute_filter_responses, FILTER_SIGMA
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    for Te, c in [(100, "#1565C0"), (500, "#2E7D32"), (1500, "#F57C00"), (5000, "#C62828")]:
        ax.plot(WAVELENGTHS, thomson_spectrum(WAVELENGTHS, Te), color=c, lw=1.5, label=f"$T_e$ = {Te} eV")
    filters = compute_filter_responses()
    for i in range(N_CHANNELS):
        ax.fill_between(WAVELENGTHS, 0, filters[i] / filters[i].max() * 0.003, alpha=0.2, color="gray")
        ax.text(FILTER_CENTERS[i], 0.0032, f"Ch{i+1}", ha="center", fontsize=8, color="gray")
    ax.axvline(LAMBDA_LASER, color="red", ls="--", lw=0.8, alpha=0.5, label="Laser 1064 nm")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Spectral density (1/nm)")
    ax.set_title("Thomson Scattering Spectra and Polychromator Channels")
    ax.legend(fontsize=9); ax.set_xlim(750, 1400); ax.set_ylim(0, 0.012)
    fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/thomson_spectrum_filters.png", dpi=200); plt.close()
    print("  Saved: thomson_spectrum_filters.png")

    # -- Chi2 landscape (LHD mid-radius) --
    Te_true, ne_true = 1495.0, 1.577
    _, obs, sig = generate_realistic_data(Te_true, ne_true, seed=44)
    obj_fn = make_objective_function(obs, sig)
    Te_g = np.linspace(50, 4500, 200); ne_g = np.linspace(0.1, 4.5, 200)
    Te_m, ne_m = np.meshgrid(Te_g, ne_g)
    chi2 = np.array([[obj_fn(np.array([t, n])) for t in Te_g] for n in ne_g])
    chi2_min = chi2.min(); best = np.unravel_index(chi2.argmin(), chi2.shape)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    pcm = ax.pcolormesh(Te_g, ne_g, chi2, norm=LogNorm(vmin=max(chi2_min, 1), vmax=1e4),
                        cmap="viridis_r", shading="nearest")
    plt.colorbar(pcm, ax=ax, label=r"$\chi^2$")
    ax.plot(Te_true, ne_true, "r*", markersize=15, label=f"True ({Te_true:.0f}, {ne_true:.2f})")
    ax.plot(Te_g[best[1]], ne_g[best[0]], "wx", markersize=12, markeredgewidth=2,
            label=f"Best ({Te_g[best[1]]:.0f}, {ne_g[best[0]]:.2f})")
    levels = [chi2_min + d for d in [1, 2.3, 5, 10, 20, 50, 100, 500]]
    cs = ax.contour(Te_g, ne_g, chi2, levels=levels, colors="white", linewidths=0.5, linestyles="--")
    ax.clabel(cs, fmt="%.0f", fontsize=8, colors="white")
    ax.set_xlabel(r"$T_e$ (eV)"); ax.set_ylabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)")
    ax.set_title(r"$\chi^2$ Landscape — LHD Mid-radius ($T_e=1495$ eV, $n_e=1.577$)")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/chi2_landscape.png", dpi=200); plt.close()
    print("  Saved: chi2_landscape.png")

    # -- PAMC posterior (from results if available) --
    pamc_out = os.path.join(BASE_DIR, "results/multipoint/Mid-radius/pamc", "0")
    if os.path.exists(pamc_out):
        rf = sorted([f for f in os.listdir(pamc_out) if f.startswith("result_T")])
        # Use intermediate T for visible spread
        mid_idx = max(0, len(rf) - 8)
        data_post = np.loadtxt(os.path.join(pamc_out, rf[mid_idx]))
        Te_s, ne_s, w = data_post[:, 4], data_post[:, 5], data_post[:, 6]
        Te_pm, ne_pm = np.average(Te_s, weights=w), np.average(ne_s, weights=w)
        Te_ps = np.sqrt(np.average((Te_s - Te_pm)**2, weights=w))
        ne_ps = np.sqrt(np.average((ne_s - ne_pm)**2, weights=w))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        axes[0].hist(Te_s, bins=60, density=True, alpha=0.7, color="steelblue", weights=w)
        axes[0].axvline(Te_true, color="red", ls="--", lw=2, label=f"True ({Te_true:.0f})")
        axes[0].axvline(Te_pm, color="orange", lw=1.5, label=f"Mean ({Te_pm:.0f}\u00b1{Te_ps:.0f})")
        axes[0].set_xlabel(r"$T_e$ (eV)"); axes[0].set_ylabel("Probability density")
        axes[0].set_title(r"Marginal posterior: $T_e$"); axes[0].legend(fontsize=9)

        axes[1].hist(ne_s, bins=60, density=True, alpha=0.7, color="darkorange", weights=w)
        axes[1].axvline(ne_true, color="red", ls="--", lw=2, label=f"True ({ne_true:.3f})")
        axes[1].axvline(ne_pm, color="steelblue", lw=1.5, label=f"Mean ({ne_pm:.3f}\u00b1{ne_ps:.3f})")
        axes[1].set_xlabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)"); axes[1].set_ylabel("Probability density")
        axes[1].set_title(r"Marginal posterior: $n_e$"); axes[1].legend(fontsize=9)

        h = axes[2].hist2d(Te_s, ne_s, bins=40, weights=w, cmap="Blues", density=True)
        axes[2].plot(Te_true, ne_true, "r*", markersize=18, zorder=10, markeredgecolor="white",
                     markeredgewidth=0.8, label="True value")
        axes[2].set_xlabel(r"$T_e$ (eV)"); axes[2].set_ylabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)")
        axes[2].set_title("Joint posterior (PAMC)"); axes[2].legend(fontsize=10)
        plt.colorbar(h[3], ax=axes[2], label="Density")
        fig.suptitle(r"PAMC Posterior — LHD Mid-radius ($T_e^{\rm true}=1495$ eV, $n_e^{\rm true}=1.577$)",
                     fontsize=13, y=1.02)
        fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/posterior_pamc.png", dpi=200, bbox_inches="tight"); plt.close()
        print("  Saved: posterior_pamc.png")

        # -- PAMC annealing (T=high and T=low) --
        d_high = np.loadtxt(os.path.join(pamc_out, rf[0]))
        d_low = np.loadtxt(os.path.join(pamc_out, rf[mid_idx]))
        np.random.seed(0)
        d_high_sub = d_high[np.random.choice(len(d_high), min(3000, len(d_high)), replace=False)]
        fill_levels = np.logspace(np.log10(max(chi2_min, 1)), 3.5, 30)
        contour_levels = sorted(set([chi2_min + d for d in [1, 2.3, 5, 10, 20, 50, 100, 300, 1000]]))

        for layout, figsize, fname in [((1, 2), (13, 5.5), "pamc_search_process"), ((2, 1), (8, 10), "pamc_search_vertical")]:
            fig, axes = plt.subplots(*layout, figsize=figsize)
            T_high = d_high[0, 2]; T_low = d_low[0, 2]
            for ax, data, sub in [(axes[0] if layout[0] == 1 else axes[0], d_high_sub, f"Prior exploration  ($T = {T_high:.0f}$)"),
                                   (axes[1], d_low, f"Posterior concentration  ($T = {T_low:.1f}$)")]:
                ts, ns, fs = data[:, 4], data[:, 5], data[:, 3]
                ax.contourf(Te_m, ne_m, chi2, levels=fill_levels, cmap="YlGnBu_r", alpha=0.35, extend="max")
                sc = ax.scatter(ts, ns, c=fs, s=4, alpha=0.5, cmap="inferno_r",
                                norm=LogNorm(vmin=max(chi2_min, 10), vmax=800), rasterized=True, edgecolors="none")
                cs = ax.contour(Te_m, ne_m, chi2, levels=contour_levels, colors="#444444", linewidths=0.6)
                for cl in ax.clabel(cs, fmt="%.0f", fontsize=9, colors="#333333"):
                    cl.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
                ax.plot(Te_true, ne_true, "*", color="#00FF00", markersize=20, markeredgecolor="black", markeredgewidth=1.0)
                ax.annotate(f"True\n({Te_true:.0f} eV, {ne_true:.2f})",
                            xy=(Te_true, ne_true), xytext=(Te_true + 700, ne_true + 0.85),
                            fontsize=10, color="#222222", ha="center",
                            arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#aaaaaa", alpha=0.9))
                ax.set_xlabel(r"$T_e$ (eV)"); ax.set_ylabel(r"$n_e$ ($\times\,10^{19}$ m$^{-3}$)")
                ax.set_title(sub, fontsize=14, pad=8); ax.set_xlim(50, 4500); ax.set_ylim(0.1, 4.5)
                ax.grid(True, alpha=0.15); ax.tick_params(direction="in")
            fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/{fname}.png", dpi=250, bbox_inches="tight", facecolor="white"); plt.close()
            print(f"  Saved: {fname}.png")

    # -- Profile scan --
    if profile_res:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        R = profile_res["R"]; Te_t = profile_res["Te_true"]; ne_t = profile_res["ne_true"]
        Te_i = profile_res["Te_inv"]; ne_i = profile_res["ne_inv"]
        dTe = profile_res["profile"][:, 3]; dne = profile_res["profile"][:, 5] / 1000.0
        axes[0].errorbar(R, Te_t, yerr=dTe, fmt='o', ms=3, color='gray', alpha=0.5, label='LHD analyzed', capsize=2)
        axes[0].plot(R, Te_i, 's', ms=3, color='steelblue', label='ODAT-SE (Nelder-Mead)')
        axes[0].set_ylabel(r"$T_e$ (eV)"); axes[0].set_title("LHD Shot #175916 — Profile Reconstruction"); axes[0].legend(); axes[0].set_ylim(bottom=0)
        axes[1].errorbar(R, ne_t, yerr=dne, fmt='o', ms=3, color='gray', alpha=0.5, label='LHD analyzed', capsize=2)
        axes[1].plot(R, ne_i, 's', ms=3, color='darkorange', label='ODAT-SE (Nelder-Mead)')
        axes[1].set_xlabel("Major radius R (mm)"); axes[1].set_ylabel(r"$n_e$ ($\times 10^{19}$ m$^{-3}$)"); axes[1].legend(); axes[1].set_ylim(bottom=0)
        fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/lhd_profile_scan.png", dpi=150); plt.close()
        print("  Saved: lhd_profile_scan.png")

    # -- Performance figures --
    if perf_res:
        # Noise sensitivity
        noise = perf_res["noise"]
        snrs = sorted(noise.keys())
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(snrs, [noise[s]["Te_mean"] for s in snrs], [noise[s]["Te_std"] for s in snrs],
                    fmt="o-", capsize=4, color="#1565C0", label=r"$T_e$ error")
        ax.errorbar(snrs, [noise[s]["ne_mean"] for s in snrs], [noise[s]["ne_std"] for s in snrs],
                    fmt="s-", capsize=4, color="#F57C00", label=r"$n_e$ error")
        ax.set_xlabel("SNR"); ax.set_ylabel("Relative error (%)")
        ax.set_title("Inversion Accuracy vs. Noise Level"); ax.legend(); ax.set_xscale("log")
        ax.set_xticks(snrs); ax.set_xticklabels(snrs); ax.grid(True, alpha=0.2)
        fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/noise_sensitivity.png", dpi=200); plt.close()
        print("  Saved: noise_sensitivity.png")

        # Te coverage
        cov = perf_res["coverage"]
        Te_scan = sorted(cov.keys())
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        a1.plot([0, 10000], [0, 10000], "k--", lw=0.8, alpha=0.3)
        a1.plot(Te_scan, [cov[t]["Te_inv"] for t in Te_scan], "o-", color="#2E7D32")
        a1.set_xlabel("True $T_e$ (eV)"); a1.set_ylabel("Inverted $T_e$ (eV)"); a1.set_title("(a) Accuracy"); a1.grid(True, alpha=0.2)
        a2.bar(range(len(Te_scan)), [cov[t]["Te_err%"] for t in Te_scan], color="#1565C0", alpha=0.7)
        a2.set_xticks(range(len(Te_scan))); a2.set_xticklabels([str(t) for t in Te_scan], rotation=45)
        a2.set_xlabel("True $T_e$ (eV)"); a2.set_ylabel("Error (%)"); a2.set_title("(b) Relative Error"); a2.grid(True, alpha=0.2, axis="y")
        fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/te_coverage.png", dpi=200); plt.close()
        print("  Saved: te_coverage.png")

        # PAMC scalability
        scal = perf_res["scalability"]
        nreps = sorted(scal.keys())
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4))
        a1.plot(nreps, [scal[n]["time"] for n in nreps], "o-", color="#C62828"); a1.set_xlabel("Replicas"); a1.set_ylabel("Time (s)"); a1.set_title("(a) Time"); a1.grid(True, alpha=0.2)
        a2.plot(nreps, [scal[n]["Te_std"] for n in nreps], "s-", color="#1565C0"); a2.set_xlabel("Replicas"); a2.set_ylabel("$T_e$ std (eV)"); a2.set_title("(b) Posterior Width"); a2.grid(True, alpha=0.2)
        a3.plot(nreps, [scal[n]["logZ"] for n in nreps], "^-", color="#2E7D32"); a3.set_xlabel("Replicas"); a3.set_ylabel("log(Z)"); a3.set_title("(c) Free Energy"); a3.grid(True, alpha=0.2)
        fig.suptitle("PAMC Scalability", fontsize=13, y=1.01); fig.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/pamc_scalability.png", dpi=200, bbox_inches="tight"); plt.close()
        print("  Saved: pamc_scalability.png")


# ============================================================
# Main
# ============================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    print("=" * 70)
    print("ODAT-SE Thomson Scattering — Comprehensive Test Suite")
    print(f"LHD Shot #175916, Data: {LHD_DATA_FILE}")
    print("=" * 70)

    # Load LHD data
    profile = load_lhd_profile()
    test_points = select_test_points(profile)
    mid = test_points["Mid-radius"]
    Te_mid, ne_mid = mid[2], mid[4] / 1000.0

    print(f"Profile: {len(profile)} good points, t=36300.1 ms")
    print(f"Mid-radius: Te={Te_mid:.0f} eV, ne={ne_mid:.3f} x10^19 m^-3")

    benchmark_res = multipoint_res = profile_res = model_res = perf_res = None

    if mode in ("all", "benchmark"):
        benchmark_res = test_algorithm_benchmark(Te_mid, ne_mid)

    if mode in ("all", "multipoint"):
        multipoint_res = test_multipoint_inversion(test_points)

    if mode in ("all", "profile"):
        profile_res = test_profile_scan(profile)

    if mode in ("all", "model"):
        model_res = test_model_selection()

    if mode in ("all", "perf"):
        perf_res = test_performance()

    if mode in ("all", "figures"):
        print("\n" + "=" * 70)
        print("Generating figures...")
        print("=" * 70)
        generate_figures(benchmark_res, multipoint_res, profile_res, model_res, perf_res)

    print("\n" + "=" * 70)
    print("All tests completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
