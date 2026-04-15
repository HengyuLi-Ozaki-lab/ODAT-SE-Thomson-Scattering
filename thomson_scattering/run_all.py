#!/usr/bin/env python3
"""
Master script: Run all 5 ODAT-SE algorithms for Thomson scattering parameter inversion.

This script:
1. Loads pre-generated synthetic data
2. Runs each algorithm (minsearch, mapper, bayes, exchange, pamc)
3. Collects results and timing information
4. Prints a comparison summary

Usage:
    cd thomson_scattering/
    python3 run_all.py [algorithm_name]

    If algorithm_name is given, run only that algorithm.
    Otherwise, run all algorithms sequentially.
"""

import sys
import os
import time
import shutil
import numpy as np

# Ensure the parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import odatse
import odatse.util.toml
import odatse.solver.function

from thomson_model import make_objective_function


def load_synthetic_data():
    """Load synthetic data from .npz file."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_data.npz")
    data = np.load(data_path)
    return data["observed_signals"], data["sigma"], float(data["Te_true"]), float(data["ne_true"])


def run_algorithm(algo_name, observed_signals, sigma):
    """
    Run a single ODAT-SE algorithm.

    Parameters
    ----------
    algo_name : str
        Algorithm directory name (e.g., "minsearch", "mapper").
    observed_signals : np.ndarray
        Observed channel signals.
    sigma : np.ndarray
        Measurement uncertainties.

    Returns
    -------
    result : dict
        Algorithm result dictionary.
    elapsed : float
        Wall-clock time in seconds.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    algo_dir = os.path.join(base_dir, algo_name)

    # Clean previous output
    output_dir = os.path.join(algo_dir, "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Save current directory and change to algorithm directory
    orig_dir = os.getcwd()
    os.chdir(algo_dir)

    try:
        # Load TOML config
        inp = odatse.util.toml.load("input.toml")
        info = odatse.Info(inp)

        # Create solver with Thomson scattering objective function
        solver = odatse.solver.function.Solver(info)
        objective_fn = make_objective_function(observed_signals, sigma)
        solver.set_function(objective_fn)

        # Create runner
        runner = odatse.Runner(solver, info)

        # Choose algorithm
        alg_module = odatse.algorithm.choose_algorithm(info.algorithm["name"])

        # Create and run algorithm
        alg = alg_module.Algorithm(info, runner)

        print(f"\n{'='*60}")
        print(f"Running {algo_name} ...")
        print(f"{'='*60}")

        t_start = time.perf_counter()
        result = alg.main()
        t_end = time.perf_counter()
        elapsed = t_end - t_start

        print(f"\n{algo_name} completed in {elapsed:.2f} seconds")
        if result:
            print(f"  Result: {result}")

        return result, elapsed

    finally:
        os.chdir(orig_dir)


def print_summary(results, Te_true, ne_true):
    """Print a comparison summary of all algorithm results."""
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON SUMMARY")
    print("=" * 80)
    print(f"True parameters: Te = {Te_true} eV, ne = {ne_true} x 1e19 m^-3")
    print()
    print(f"{'Algorithm':<15} {'Te (eV)':<12} {'ne (1e19)':<12} {'chi2':<12} {'Time (s)':<10}")
    print("-" * 65)

    for algo_name, (result, elapsed) in results.items():
        if result and "x" in result:
            x = np.atleast_1d(result["x"])
            Te_inv = float(x[0]) if len(x) > 0 else float("nan")
            ne_inv = float(x[1]) if len(x) > 1 else float("nan")
            fx_val = result.get("fx", float("nan"))
            if isinstance(fx_val, np.ndarray):
                fx_val = float(fx_val.flat[0]) if fx_val.size > 0 else float("nan")
            else:
                fx_val = float(fx_val)
            print(f"{algo_name:<15} {Te_inv:<12.2f} {ne_inv:<12.4f} {fx_val:<12.4f} {elapsed:<10.2f}")
        else:
            print(f"{algo_name:<15} {'---':<12} {'---':<12} {'---':<12} {elapsed:<10.2f}")


def main():
    # Load synthetic data
    observed_signals, sigma, Te_true, ne_true = load_synthetic_data()

    print("=" * 60)
    print("ODAT-SE Thomson Scattering Benchmark")
    print("=" * 60)
    print(f"True parameters: Te = {Te_true} eV, ne = {ne_true} x 1e19 m^-3")
    print(f"Observed signals: {observed_signals}")
    print(f"Uncertainties:    {sigma}")

    # Determine which algorithms to run
    all_algos = ["minsearch", "mapper", "bayes", "exchange", "pamc"]

    if len(sys.argv) > 1:
        algos_to_run = [a for a in sys.argv[1:] if a in all_algos]
        if not algos_to_run:
            print(f"Unknown algorithm: {sys.argv[1]}")
            print(f"Available: {all_algos}")
            sys.exit(1)
    else:
        algos_to_run = all_algos

    # Run algorithms
    results = {}
    for algo in algos_to_run:
        try:
            result, elapsed = run_algorithm(algo, observed_signals, sigma)
            results[algo] = (result, elapsed)
        except Exception as e:
            print(f"\nERROR running {algo}: {e}")
            import traceback
            traceback.print_exc()
            results[algo] = (None, 0.0)

    # Print summary
    print_summary(results, Te_true, ne_true)


if __name__ == "__main__":
    main()
