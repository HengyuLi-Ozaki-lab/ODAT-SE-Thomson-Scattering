#!/usr/bin/env python3
"""
Pure Synthetic Benchmark for ODAT-SE Thomson Scattering Inference.

All tests use purely synthetic data with exactly known ground truth.
No LHD data dependency — this is a clean evaluation of ODAT-SE's
intrinsic inference performance.

Tests:
  1. accuracy   — 5 algorithms × 5 parameter points
  2. bias       — Monte Carlo bias analysis (100 trials × 5 points)
  3. posterior   — PAMC posterior coverage validation (50 trials)
  4. snr        — SNR sweep (6 levels × 20 trials)
  5. model      — Bidirectional model selection (Maxwell ↔ Kappa)
  6. channels   — Channel count impact on bias (3, 5, 7, 9, 11 channels)
  7. scaling    — PAMC replica count scalability

Usage:
    python run_synthetic_benchmark.py           # Run all tests
    python run_synthetic_benchmark.py accuracy   # Single test
    python run_synthetic_benchmark.py channels   # Channel test only
"""

import sys, os, time, shutil, json
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import odatse
import odatse.util.toml
import odatse.solver.function

from thomson_model import (
    compute_channel_signals, thomson_spectrum, make_objective_function,
    make_kappa_objective_function, N_CHANNELS, FILTER_CENTERS,
    LAMBDA_LASER, WAVELENGTHS, compute_filter_responses, FILTER_SIGMA, ME_C2_EV
)

CONFIG_DIR = os.path.join(BASE_DIR, "config")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "synthetic_benchmark")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

# Ground truth test points
TEST_POINTS = [
    ("A_edge",     100,  1.0),
    ("B_moderate", 500,  2.0),
    ("C_mid",      1500, 1.5),
    ("D_high",     3000, 1.0),
    ("E_core",     5000, 0.5),
]

# Noise model
PHOTON_SCALE = 5000.0
READOUT_NOISE = 3.0
STRAY_LIGHT = np.array([0.0, 0.001, 0.05, 0.005, 0.0])


def generate_data(Te, ne, seed, snr_override=None, n_channels=None, filter_responses=None):
    """Generate synthetic polychromator data with Poisson + readout noise."""
    np.random.seed(seed)
    if n_channels is None:
        signals = compute_channel_signals(Te, ne)
        stray = STRAY_LIGHT
    else:
        # Custom channel configuration
        spectrum = thomson_spectrum(WAVELENGTHS, Te)
        signals = np.zeros(n_channels)
        for i in range(n_channels):
            signals[i] = ne * np.trapz(spectrum * filter_responses[i], WAVELENGTHS)
        stray = np.zeros(n_channels)

    if snr_override is not None:
        sigma = signals / snr_override
        noise = sigma * np.random.randn(len(signals))
        obs = signals + noise
        return signals, obs, sigma

    N_sig = signals * PHOTON_SCALE
    N_str = stray * PHOTON_SCALE
    N_tot = N_sig + N_str
    obs = (np.random.poisson(np.maximum(N_tot, 0)) +
           READOUT_NOISE * np.random.randn(len(signals))) / PHOTON_SCALE
    sigma = np.sqrt(np.maximum(N_tot, 1) + READOUT_NOISE**2) / PHOTON_SCALE
    return signals, obs, sigma


def run_nm(obj_fn, Te_true, ne_true, out_tag, overrides=None):
    """Run Nelder-Mead and return (Te_inv, ne_inv, chi2, time)."""
    config = os.path.join(CONFIG_DIR, "minsearch.toml")
    inp = odatse.util.toml.load(config)
    inp["base"]["output_dir"] = os.path.join(RESULTS_DIR, out_tag)
    inp["algorithm"]["param"]["initial_list"] = [max(Te_true * 0.5, 20), max(ne_true * 0.5, 0.1)]
    inp["algorithm"]["minimize"]["initial_scale_list"] = [max(Te_true * 0.3, 10), max(ne_true * 0.3, 0.05)]
    if overrides:
        for k, v in overrides.items():
            parts = k.split(".")
            d = inp
            for p in parts[:-1]: d = d[p]
            d[parts[-1]] = v
    out = inp["base"]["output_dir"]
    if os.path.exists(out): shutil.rmtree(out)

    orig = os.getcwd(); os.chdir(BASE_DIR)
    info = odatse.Info(inp)
    solver = odatse.solver.function.Solver(info)
    solver.set_function(obj_fn)
    runner = odatse.Runner(solver, info)
    alg = odatse.algorithm.choose_algorithm("minsearch").Algorithm(info, runner)
    t0 = time.perf_counter()
    result = alg.main()
    elapsed = time.perf_counter() - t0
    os.chdir(orig)
    return result["x"][0], result["x"][1], result["fx"], elapsed


def run_odatse(algo_name, obj_fn, Te_true, ne_true, out_tag, extra_overrides=None):
    """Run any ODAT-SE algorithm."""
    config = os.path.join(CONFIG_DIR, f"{algo_name}.toml")
    inp = odatse.util.toml.load(config)
    inp["base"]["output_dir"] = os.path.join(RESULTS_DIR, out_tag)
    if algo_name == "minsearch":
        inp["algorithm"]["param"]["initial_list"] = [max(Te_true * 0.5, 20), max(ne_true * 0.5, 0.1)]
        inp["algorithm"]["minimize"]["initial_scale_list"] = [max(Te_true * 0.3, 10), max(ne_true * 0.3, 0.05)]
    elif algo_name in ("exchange", "pamc"):
        inp["algorithm"]["param"]["step_list"] = [max(Te_true * 0.1, 10), max(ne_true * 0.1, 0.03)]
    if extra_overrides:
        for k, v in extra_overrides.items():
            parts = k.split(".")
            d = inp
            for p in parts[:-1]: d = d[p]
            d[parts[-1]] = v
    out = inp["base"]["output_dir"]
    if os.path.exists(out): shutil.rmtree(out)

    orig = os.getcwd(); os.chdir(BASE_DIR)
    info = odatse.Info(inp)
    solver = odatse.solver.function.Solver(info)
    solver.set_function(obj_fn)
    runner = odatse.Runner(solver, info)
    alg = odatse.algorithm.choose_algorithm(algo_name).Algorithm(info, runner)
    t0 = time.perf_counter()
    result = alg.main()
    elapsed = time.perf_counter() - t0
    os.chdir(orig)
    return result, elapsed


# ============================================================
# Test 1: Single-point accuracy
# ============================================================
def test_accuracy():
    print("\n" + "=" * 70)
    print("TEST 1: Single-point Accuracy (5 algorithms × 5 points)")
    print("=" * 70)

    algos = ["minsearch", "mapper", "bayes", "exchange", "pamc"]
    results = {}

    for label, Te, ne in TEST_POINTS:
        _, obs, sig = generate_data(Te, ne, seed=42)
        obj_fn = make_objective_function(obs, sig)
        results[label] = {}

        for algo in algos:
            print(f"  {label} / {algo}...", end=" ", flush=True)
            try:
                res, elapsed = run_odatse(algo, obj_fn, Te, ne, f"acc/{label}/{algo}")
                if res and "x" in res:
                    Te_i, ne_i = res["x"][0], res["x"][1]
                    fx = float(res["fx"]) if not isinstance(res["fx"], np.ndarray) else float(res["fx"].flat[0])
                    results[label][algo] = {"Te": Te_i, "ne": ne_i, "chi2": fx, "time": elapsed,
                                            "Te_err%": abs(Te_i - Te) / Te * 100,
                                            "ne_err%": abs(ne_i - ne) / ne * 100}
                    print(f"Te={Te_i:.0f}({results[label][algo]['Te_err%']:.1f}%), ne={ne_i:.3f}, {elapsed:.2f}s")
                else:
                    results[label][algo] = None
                    print(f"done ({elapsed:.2f}s)")
            except Exception as e:
                results[label][algo] = None
                print(f"FAILED: {e}")

    # Summary
    print(f"\n{'Point':<12} {'Algo':<12} {'Te_true':<8} {'Te_inv':<8} {'Te%':<6} {'ne_true':<8} {'ne_inv':<8} {'ne%':<6} {'time':<6}")
    print("-" * 80)
    for label, Te, ne in TEST_POINTS:
        for algo in algos:
            r = results[label].get(algo)
            if r:
                print(f"{label:<12} {algo:<12} {Te:<8} {r['Te']:<8.0f} {r['Te_err%']:<6.1f} {ne:<8.2f} {r['ne']:<8.3f} {r['ne_err%']:<6.1f} {r['time']:<6.2f}")
    return results


# ============================================================
# Test 2: Monte Carlo bias analysis
# ============================================================
def test_bias():
    print("\n" + "=" * 70)
    print("TEST 2: Monte Carlo Bias Analysis (100 trials × 5 points)")
    print("=" * 70)

    N = 100
    results = {}
    for label, Te, ne in TEST_POINTS:
        Te_arr, ne_arr = [], []
        for i in range(N):
            _, obs, sig = generate_data(Te, ne, seed=3000 + i)
            fn = make_objective_function(obs, sig)
            Te_i, ne_i, _, _ = run_nm(fn, Te, ne, f"bias/{label}/{i}")
            Te_arr.append(Te_i); ne_arr.append(ne_i)
            if (i + 1) % 25 == 0: print(f"  {label}: {i+1}/{N}")
        Te_arr = np.array(Te_arr); ne_arr = np.array(ne_arr)
        results[label] = {
            "Te_true": Te, "ne_true": ne,
            "Te_mean": Te_arr.mean(), "Te_std": Te_arr.std(),
            "Te_bias": Te_arr.mean() - Te, "Te_bias%": (Te_arr.mean() - Te) / Te * 100,
            "ne_mean": ne_arr.mean(), "ne_std": ne_arr.std(),
            "ne_bias": ne_arr.mean() - ne, "ne_bias%": (ne_arr.mean() - ne) / ne * 100,
            "Te_all": Te_arr, "ne_all": ne_arr,
        }
        print(f"  {label}: Te bias={results[label]['Te_bias']:+.1f} ({results[label]['Te_bias%']:+.1f}%), "
              f"ne bias={results[label]['ne_bias']:+.4f} ({results[label]['ne_bias%']:+.1f}%)")
    return results


# ============================================================
# Test 3: PAMC posterior validation
# ============================================================
def test_posterior():
    print("\n" + "=" * 70)
    print("TEST 3: PAMC Posterior Validation (50 trials, point C)")
    print("=" * 70)

    Te_true, ne_true = 1500, 1.5
    N = 50
    pamc_results = []

    for i in range(N):
        _, obs, sig = generate_data(Te_true, ne_true, seed=4000 + i)
        fn = make_objective_function(obs, sig)
        res, elapsed = run_odatse("pamc", fn, Te_true, ne_true, f"post/{i}")

        pamc_out = os.path.join(RESULTS_DIR, f"post/{i}", "0")
        rf = sorted([f for f in os.listdir(pamc_out) if f.startswith("result_T")])
        samples = np.loadtxt(os.path.join(pamc_out, rf[-1]))
        Te_s, ne_s, w = samples[:, 4], samples[:, 5], samples[:, 6]
        Te_m = np.average(Te_s, weights=w); ne_m = np.average(ne_s, weights=w)
        Te_std = np.sqrt(np.average((Te_s - Te_m)**2, weights=w))
        ne_std = np.sqrt(np.average((ne_s - ne_m)**2, weights=w))

        pamc_results.append({
            "Te_mean": Te_m, "ne_mean": ne_m, "Te_std": Te_std, "ne_std": ne_std,
            "Te_in_1s": abs(Te_true - Te_m) < Te_std,
            "Te_in_2s": abs(Te_true - Te_m) < 2 * Te_std,
            "ne_in_1s": abs(ne_true - ne_m) < ne_std,
            "ne_in_2s": abs(ne_true - ne_m) < 2 * ne_std,
        })
        if (i + 1) % 10 == 0: print(f"  {i+1}/{N}")

    Te_1s = np.mean([r["Te_in_1s"] for r in pamc_results]) * 100
    Te_2s = np.mean([r["Te_in_2s"] for r in pamc_results]) * 100
    ne_1s = np.mean([r["ne_in_1s"] for r in pamc_results]) * 100
    ne_2s = np.mean([r["ne_in_2s"] for r in pamc_results]) * 100

    print(f"\n  Coverage: Te 1σ={Te_1s:.0f}% (exp 68%), 2σ={Te_2s:.0f}% (exp 95%)")
    print(f"           ne 1σ={ne_1s:.0f}% (exp 68%), 2σ={ne_2s:.0f}% (exp 95%)")

    return {"pamc_results": pamc_results, "Te_true": Te_true, "ne_true": ne_true,
            "Te_1s": Te_1s, "Te_2s": Te_2s, "ne_1s": ne_1s, "ne_2s": ne_2s}


# ============================================================
# Test 4: SNR sweep
# ============================================================
def test_snr():
    print("\n" + "=" * 70)
    print("TEST 4: SNR Sweep (point C, 6 levels × 20 trials)")
    print("=" * 70)

    Te_true, ne_true = 1500, 1.5
    snr_levels = [5, 10, 20, 50, 100, 200]
    N = 20
    results = {}

    for snr in snr_levels:
        Te_errs, ne_errs = [], []
        for i in range(N):
            _, obs, sig = generate_data(Te_true, ne_true, seed=6000 + i, snr_override=snr)
            fn = make_objective_function(obs, sig)
            Te_i, ne_i, _, _ = run_nm(fn, Te_true, ne_true, f"snr/{snr}/{i}")
            Te_errs.append(abs(Te_i - Te_true) / Te_true * 100)
            ne_errs.append(abs(ne_i - ne_true) / ne_true * 100)
        results[snr] = {"Te_mean": np.mean(Te_errs), "Te_std": np.std(Te_errs),
                        "ne_mean": np.mean(ne_errs), "ne_std": np.std(ne_errs)}
        print(f"  SNR={snr:3d}: Te={np.mean(Te_errs):.1f}±{np.std(Te_errs):.1f}%, "
              f"ne={np.mean(ne_errs):.1f}±{np.std(ne_errs):.1f}%")
    return results


# ============================================================
# Test 5: Bidirectional model selection
# ============================================================
def test_model_selection():
    print("\n" + "=" * 70)
    print("TEST 5: Bidirectional Model Selection")
    print("=" * 70)

    results = {}

    def run_pamc_model(obj_fn, config_name, out_tag):
        """Run PAMC using a specific model config (handles dimension correctly)."""
        config = os.path.join(CONFIG_DIR, config_name)
        inp = odatse.util.toml.load(config)
        inp["base"]["output_dir"] = os.path.join(RESULTS_DIR, out_tag)
        out = inp["base"]["output_dir"]
        if os.path.exists(out): shutil.rmtree(out)
        orig = os.getcwd(); os.chdir(BASE_DIR)
        info = odatse.Info(inp)
        solver = odatse.solver.function.Solver(info)
        solver.set_function(obj_fn)
        runner = odatse.Runner(solver, info)
        alg = odatse.algorithm.choose_algorithm("pamc").Algorithm(info, runner)
        result = alg.main()
        os.chdir(orig)
        fx_file = os.path.join(out, "fx.txt")
        logZ = np.loadtxt(fx_file)[-1, 4] if os.path.exists(fx_file) else None
        return result, logZ

    # A: Data from Maxwell → should favor Maxwell
    print("\n  A: Data generated from Maxwell (Te=500, ne=3.0)")
    Te, ne = 500.0, 3.0
    _, obs, sig = generate_data(Te, ne, seed=7000)
    for model, config_name, fn_maker in [
        ("maxwell", "model_maxwell.toml", lambda: make_objective_function(obs, sig)),
        ("kappa", "model_kappa.toml", lambda: make_kappa_objective_function(obs, sig)),
    ]:
        res, logZ = run_pamc_model(fn_maker(), config_name, f"model/maxwell_data/{model}")
        results[f"maxwell_data_{model}"] = {"logZ": logZ, "chi2": res["fx"]}
        print(f"    {model}: chi2={res['fx']:.2f}, logZ={logZ:.2f}")

    ln_B_maxwell = results["maxwell_data_maxwell"]["logZ"] - results["maxwell_data_kappa"]["logZ"]
    print(f"    ln(B) = {ln_B_maxwell:.2f} → {'Maxwell' if ln_B_maxwell > 0 else 'Kappa'} favored")

    # B: Data from Kappa (κ=5) → should favor Kappa
    print("\n  B: Data generated from Kappa (Te=500, ne=3.0, κ=5)")
    from scipy.special import gamma as gamma_func
    np.random.seed(7001)
    Te_k, ne_k, kappa = 500.0, 3.0, 5.0
    sigma_lambda = LAMBDA_LASER * np.sqrt(2.0 * Te_k / ME_C2_EV)
    delta = WAVELENGTHS - LAMBDA_LASER
    u2 = (delta / sigma_lambda) ** 2
    norm = gamma_func(kappa + 1) / (gamma_func(kappa - 0.5) * np.sqrt(np.pi * kappa) * sigma_lambda)
    kappa_spectrum = norm * (1.0 + u2 / kappa) ** (-(kappa + 1))
    filters = compute_filter_responses()
    kappa_signals = np.zeros(N_CHANNELS)
    for i in range(N_CHANNELS):
        kappa_signals[i] = ne_k * np.trapz(kappa_spectrum * filters[i], WAVELENGTHS)
    sig_k = kappa_signals / 20  # SNR=20
    obs_k = kappa_signals + sig_k * np.random.randn(N_CHANNELS)

    for model, config_name, fn_maker in [
        ("maxwell", "model_maxwell.toml", lambda: make_objective_function(obs_k, sig_k)),
        ("kappa", "model_kappa.toml", lambda: make_kappa_objective_function(obs_k, sig_k)),
    ]:
        res, logZ = run_pamc_model(fn_maker(), config_name, f"model/kappa_data/{model}")
        results[f"kappa_data_{model}"] = {"logZ": logZ, "chi2": res["fx"]}
        print(f"    {model}: chi2={res['fx']:.2f}, logZ={logZ:.2f}")

    ln_B_kappa = results["kappa_data_maxwell"]["logZ"] - results["kappa_data_kappa"]["logZ"]
    print(f"    ln(B) = {ln_B_kappa:.2f} → {'Maxwell' if ln_B_kappa > 0 else 'Kappa'} favored")

    results["ln_B_maxwell_data"] = ln_B_maxwell
    results["ln_B_kappa_data"] = ln_B_kappa
    return results


# ============================================================
# Test 6: Channel count impact
# ============================================================
def test_channels():
    print("\n" + "=" * 70)
    print("TEST 6: Channel Count Impact (3–11 channels, 50 NM trials)")
    print("=" * 70)

    Te_true, ne_true = 1500, 1.5
    channel_counts = [3, 5, 7, 9, 11]
    N = 50
    results = {}

    for n_ch in channel_counts:
        # Generate filter centers: evenly spaced in 800–1300 nm, excluding 1044–1084 nm
        all_centers = np.linspace(820, 1280, n_ch + 4)
        centers = [c for c in all_centers if abs(c - LAMBDA_LASER) > 20][:n_ch]
        centers = np.array(sorted(centers))

        # Build filter responses
        filt = np.zeros((n_ch, len(WAVELENGTHS)))
        for i in range(n_ch):
            filt[i] = np.exp(-0.5 * ((WAVELENGTHS - centers[i]) / FILTER_SIGMA) ** 2)

        # Build objective function factory for this channel config
        def make_obj(obs_data, sigma_data, filt_local=filt, n_local=n_ch, centers_local=centers):
            def objective(x):
                Te, ne = x[0], x[1]
                if Te <= 0 or ne <= 0: return 1e10
                spec = thomson_spectrum(WAVELENGTHS, Te)
                model = np.zeros(n_local)
                for i in range(n_local):
                    model[i] = ne * np.trapz(spec * filt_local[i], WAVELENGTHS)
                return np.sum(((obs_data - model) / sigma_data) ** 2)
            return objective

        Te_arr, ne_arr = [], []
        for i in range(N):
            _, obs, sig = generate_data(Te_true, ne_true, seed=8000 + i,
                                        n_channels=n_ch, filter_responses=filt)
            fn = make_obj(obs, sig)
            Te_i, ne_i, _, _ = run_nm(fn, Te_true, ne_true, f"ch/{n_ch}/{i}")
            Te_arr.append(Te_i); ne_arr.append(ne_i)
            if (i + 1) % 25 == 0: print(f"  {n_ch}ch: {i+1}/{N}")

        Te_arr = np.array(Te_arr); ne_arr = np.array(ne_arr)
        results[n_ch] = {
            "centers": centers.tolist(),
            "Te_bias": Te_arr.mean() - Te_true, "Te_bias%": (Te_arr.mean() - Te_true) / Te_true * 100,
            "Te_std": Te_arr.std(),
            "ne_bias": ne_arr.mean() - ne_true, "ne_bias%": (ne_arr.mean() - ne_true) / ne_true * 100,
            "ne_std": ne_arr.std(),
            "Te_all": Te_arr, "ne_all": ne_arr,
        }
        print(f"  {n_ch:2d} ch: Te bias={results[n_ch]['Te_bias']:+.1f}eV ({results[n_ch]['Te_bias%']:+.1f}%), "
              f"std={results[n_ch]['Te_std']:.1f}, ne bias={results[n_ch]['ne_bias']:+.4f} ({results[n_ch]['ne_bias%']:+.1f}%)")
    return results


# ============================================================
# Test 7: PAMC scaling
# ============================================================
def test_scaling():
    print("\n" + "=" * 70)
    print("TEST 7: PAMC Replica Scaling")
    print("=" * 70)

    Te_true, ne_true = 1500, 1.5
    _, obs, sig = generate_data(Te_true, ne_true, seed=42)
    fn = make_objective_function(obs, sig)
    replica_counts = [20, 50, 100, 200, 500]
    results = {}

    for nrep in replica_counts:
        res, elapsed = run_odatse("pamc", fn, Te_true, ne_true, f"scale/{nrep}",
                                  extra_overrides={"algorithm.pamc.nreplica_per_proc": nrep})
        pamc_out = os.path.join(RESULTS_DIR, f"scale/{nrep}", "0")
        rf = sorted([f for f in os.listdir(pamc_out) if f.startswith("result_T")])
        samples = np.loadtxt(os.path.join(pamc_out, rf[-1]))
        Te_s, w = samples[:, 4], samples[:, 6]
        Te_m = np.average(Te_s, weights=w)
        Te_std = np.sqrt(np.average((Te_s - Te_m)**2, weights=w))
        fx_data = np.loadtxt(os.path.join(RESULTS_DIR, f"scale/{nrep}", "fx.txt"))
        logZ = fx_data[-1, 4]

        results[nrep] = {"time": elapsed, "Te_std": Te_std, "logZ": logZ}
        print(f"  nrep={nrep:3d}: time={elapsed:.2f}s, Te_std={Te_std:.1f}, logZ={logZ:.1f}")
    return results


# ============================================================
# Figure generation
# ============================================================
def generate_figures(acc=None, bias=None, post=None, snr=None,
                     model=None, channels=None, scaling=None):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11})

    # --- Fig 1: Bias scatter (5 subplots) ---
    if bias:
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        for ax, (label, Te, ne) in zip(axes, TEST_POINTS):
            r = bias[label]
            ax.scatter(r["Te_all"], r["ne_all"], s=6, alpha=0.5, c="#1565C0")
            ax.plot(Te, ne, "r*", ms=15, zorder=10, mec="black", mew=0.8)
            ax.plot(r["Te_mean"], r["ne_mean"], "D", color="orange", ms=8, zorder=9, mec="black")
            ax.set_title(f"{label}\nTe:{r['Te_bias%']:+.1f}%, ne:{r['ne_bias%']:+.1f}%", fontsize=10)
            ax.set_xlabel(r"$T_e$ (eV)"); ax.set_ylabel(r"$n_e$")
            ax.axvline(Te, color="red", ls=":", lw=0.5, alpha=0.4)
            ax.axhline(ne, color="red", ls=":", lw=0.5, alpha=0.4)
        fig.suptitle("Monte Carlo Bias Analysis (100 NM trials per point)", fontsize=14, y=1.02)
        fig.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/synth_bias_scatter.png", dpi=200, bbox_inches="tight"); plt.close()
        print("  Saved: synth_bias_scatter.png")

        # Bias vs Te
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        Te_vals = [Te for _, Te, _ in TEST_POINTS]
        Te_biases = [bias[l]["Te_bias%"] for l, _, _ in TEST_POINTS]
        ne_biases = [bias[l]["ne_bias%"] for l, _, _ in TEST_POINTS]
        Te_stds = [bias[l]["Te_std"] for l, _, _ in TEST_POINTS]
        a1.bar(range(len(Te_vals)), Te_biases, color="#1565C0", alpha=0.7)
        a1.set_xticks(range(len(Te_vals))); a1.set_xticklabels([str(t) for t in Te_vals])
        a1.set_xlabel("True $T_e$ (eV)"); a1.set_ylabel("$T_e$ bias (%)"); a1.set_title("$T_e$ Bias vs Temperature")
        a1.axhline(0, color="red", ls="--"); a1.grid(True, alpha=0.2, axis="y")
        a2.bar(range(len(Te_vals)), ne_biases, color="#F57C00", alpha=0.7)
        a2.set_xticks(range(len(Te_vals))); a2.set_xticklabels([str(t) for t in Te_vals])
        a2.set_xlabel("True $T_e$ (eV)"); a2.set_ylabel("$n_e$ bias (%)"); a2.set_title("$n_e$ Bias vs Temperature")
        a2.axhline(0, color="red", ls="--"); a2.grid(True, alpha=0.2, axis="y")
        fig.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/synth_bias_vs_te.png", dpi=200); plt.close()
        print("  Saved: synth_bias_vs_te.png")

    # --- PAMC coverage ---
    if post:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        cats = [r"$T_e$ 1σ", r"$T_e$ 2σ", r"$n_e$ 1σ", r"$n_e$ 2σ"]
        obs = [post["Te_1s"], post["Te_2s"], post["ne_1s"], post["ne_2s"]]
        exp = [68.3, 95.4, 68.3, 95.4]
        x = np.arange(4)
        ax.bar(x - 0.18, obs, 0.35, label="Observed", color="#2E7D32", alpha=0.8)
        ax.bar(x + 0.18, exp, 0.35, label="Expected", color="#BDBDBD", alpha=0.8)
        for xi, v in zip(x, obs): ax.text(xi - 0.18, v + 1, f"{v:.0f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("Coverage (%)")
        ax.set_title(f"PAMC Posterior Coverage (50 trials, Te=1500, ne=1.5)"); ax.legend(); ax.set_ylim(0, 110)
        fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/synth_coverage.png", dpi=200); plt.close()
        print("  Saved: synth_coverage.png")

    # --- SNR sweep ---
    if snr:
        snrs = sorted(snr.keys())
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.errorbar(snrs, [snr[s]["Te_mean"] for s in snrs], [snr[s]["Te_std"] for s in snrs],
                    fmt="o-", capsize=4, color="#1565C0", label=r"$T_e$")
        ax.errorbar(snrs, [snr[s]["ne_mean"] for s in snrs], [snr[s]["ne_std"] for s in snrs],
                    fmt="s-", capsize=4, color="#F57C00", label=r"$n_e$")
        ax.set_xlabel("SNR"); ax.set_ylabel("Relative error (%)"); ax.set_xscale("log")
        ax.set_xticks(snrs); ax.set_xticklabels(snrs)
        ax.set_title("Inversion Accuracy vs SNR (pure synthetic)"); ax.legend(); ax.grid(True, alpha=0.2)
        fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/synth_snr.png", dpi=200); plt.close()
        print("  Saved: synth_snr.png")

    # --- Model selection ---
    if model:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = [0, 1]
        vals = [model["ln_B_maxwell_data"], model["ln_B_kappa_data"]]
        colors = ["#2E7D32" if v > 0 else "#C62828" for v in vals]
        bars = ax.bar(x, vals, color=colors, alpha=0.8, width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(["Maxwell-generated\ndata", "Kappa-generated\ndata"])
        ax.set_ylabel("ln(Bayes factor)  [+ favors Maxwell]")
        ax.set_title("Bidirectional Model Selection")
        ax.axhline(5, color="gray", ls="--", lw=0.8, alpha=0.5); ax.text(1.3, 5.5, "decisive", fontsize=8, color="gray")
        ax.axhline(-5, color="gray", ls="--", lw=0.8, alpha=0.5); ax.text(1.3, -6, "decisive", fontsize=8, color="gray")
        ax.axhline(0, color="black", lw=0.5)
        for xi, v in zip(x, vals): ax.text(xi, v + (1 if v > 0 else -2), f"{v:.1f}", ha="center", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
        fig.tight_layout(); fig.savefig(f"{FIGURES_DIR}/synth_model_selection.png", dpi=200); plt.close()
        print("  Saved: synth_model_selection.png")

    # --- Channel count impact ---
    if channels:
        n_chs = sorted(channels.keys())
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
        Te_bias_abs = [abs(channels[n]["Te_bias%"]) for n in n_chs]
        Te_stds = [channels[n]["Te_std"] for n in n_chs]
        ne_bias_abs = [abs(channels[n]["ne_bias%"]) for n in n_chs]

        a1.plot(n_chs, Te_bias_abs, "o-", color="#1565C0", lw=2, ms=8, label="|Te bias|")
        a1.plot(n_chs, ne_bias_abs, "s-", color="#F57C00", lw=2, ms=8, label="|ne bias|")
        a1.set_xlabel("Number of channels"); a1.set_ylabel("Absolute bias (%)")
        a1.set_title("(a) MLE Bias vs Channel Count"); a1.legend(); a1.grid(True, alpha=0.2)

        a2.plot(n_chs, Te_stds, "o-", color="#1565C0", lw=2, ms=8, label="Te std")
        a2.set_xlabel("Number of channels"); a2.set_ylabel("MLE scatter (eV)")
        a2.set_title("(b) MLE Scatter vs Channel Count"); a2.legend(); a2.grid(True, alpha=0.2)

        fig.suptitle("Effect of Polychromator Channel Count on Inference Quality", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/synth_channel_impact.png", dpi=200, bbox_inches="tight"); plt.close()
        print("  Saved: synth_channel_impact.png")

    # --- Scaling ---
    if scaling:
        nreps = sorted(scaling.keys())
        fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14, 4))
        a1.plot(nreps, [scaling[n]["time"] for n in nreps], "o-", color="#C62828", lw=1.5)
        a1.set_xlabel("Replicas"); a1.set_ylabel("Time (s)"); a1.set_title("(a) Computation Time"); a1.grid(True, alpha=0.2)
        a2.plot(nreps, [scaling[n]["Te_std"] for n in nreps], "s-", color="#1565C0", lw=1.5)
        a2.set_xlabel("Replicas"); a2.set_ylabel("Te std (eV)"); a2.set_title("(b) Posterior Width"); a2.grid(True, alpha=0.2)
        a3.plot(nreps, [scaling[n]["logZ"] for n in nreps], "^-", color="#2E7D32", lw=1.5)
        a3.set_xlabel("Replicas"); a3.set_ylabel("log(Z)"); a3.set_title("(c) Free Energy"); a3.grid(True, alpha=0.2)
        fig.suptitle("PAMC Scalability (pure synthetic)", fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(f"{FIGURES_DIR}/synth_scaling.png", dpi=200, bbox_inches="tight"); plt.close()
        print("  Saved: synth_scaling.png")


# ============================================================
# Main
# ============================================================
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("ODAT-SE Thomson Scattering — Pure Synthetic Benchmark")
    print("=" * 70)

    acc = bias = post = snr_res = model = channels = scaling = None

    if mode in ("all", "accuracy"):  acc = test_accuracy()
    if mode in ("all", "bias"):      bias = test_bias()
    if mode in ("all", "posterior"):  post = test_posterior()
    if mode in ("all", "snr"):       snr_res = test_snr()
    if mode in ("all", "model"):     model = test_model_selection()
    if mode in ("all", "channels"):  channels = test_channels()
    if mode in ("all", "scaling"):   scaling = test_scaling()

    if mode == "all" or mode == "figures":
        print("\n" + "=" * 70)
        print("Generating figures...")
        print("=" * 70)
        generate_figures(acc, bias, post, snr_res, model, channels, scaling)

    print("\n" + "=" * 70)
    print("Synthetic benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
