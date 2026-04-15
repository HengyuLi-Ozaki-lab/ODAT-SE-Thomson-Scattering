#!/usr/bin/env python3
"""
Bayesian Model Selection: Maxwell vs. Kappa EVDF

Runs PAMC for two competing models and compares their free energies
(log partition functions) to compute the Bayes factor.

Model 1 (Maxwell): 2 parameters (Te, ne) - Gaussian spectral shape
Model 2 (Kappa):   3 parameters (Te, ne, kappa) - power-law tails

Since the synthetic data was generated from a Maxwell distribution,
we expect strong evidence favoring the Maxwell model.

Bayes factor interpretation (Jeffreys scale):
  |ln(B)| < 1    : Not worth more than a bare mention
  1 < |ln(B)| < 3 : Positive evidence
  3 < |ln(B)| < 5 : Strong evidence
  |ln(B)| > 5     : Very strong / decisive evidence
"""

import os
import sys
import shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import odatse
import odatse.util.toml
import odatse.solver.function
import odatse.algorithm.pamc

from thomson_model import make_objective_function, make_kappa_objective_function


def load_synthetic_data():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "synthetic_data.npz")
    data = np.load(data_path)
    return data["observed_signals"], data["sigma"]


def run_pamc(toml_file, objective_fn, model_name):
    """Run PAMC for a given model and return the log partition function."""
    orig_dir = os.getcwd()
    work_dir = os.path.dirname(os.path.abspath(toml_file))
    os.chdir(work_dir)

    toml_basename = os.path.basename(toml_file)

    # Clean previous output
    inp = odatse.util.toml.load(toml_basename)
    info = odatse.Info(inp)
    output_dir = os.path.join(work_dir, info.base["output_dir"].name)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # Re-load after cleanup
    inp = odatse.util.toml.load(toml_basename)
    info = odatse.Info(inp)

    solver = odatse.solver.function.Solver(info)
    solver.set_function(objective_fn)

    runner = odatse.Runner(solver, info)
    alg = odatse.algorithm.pamc.Algorithm(info, runner)

    print(f"\n{'='*50}")
    print(f"Running PAMC for {model_name} model (dim={info.base['dimension']})")
    print(f"{'='*50}")

    result = alg.main()

    # Read free energy from fx.txt
    fx_file = os.path.join(output_dir, "fx.txt")
    logZ = None
    if os.path.exists(fx_file):
        fx_data = np.loadtxt(fx_file)
        logZ = fx_data[-1, 4]  # last row, column 5 = logZ

    os.chdir(orig_dir)
    return result, logZ


def main():
    observed_signals, sigma = load_synthetic_data()

    print("=" * 60)
    print("BAYESIAN MODEL SELECTION: Maxwell vs. Kappa EVDF")
    print("=" * 60)

    # Model 1: Maxwell (Gaussian) - 2 parameters
    maxwell_fn = make_objective_function(observed_signals, sigma)
    maxwell_toml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_maxwell.toml")
    maxwell_result, logZ_maxwell = run_pamc(maxwell_toml, maxwell_fn, "Maxwell")

    # Model 2: Kappa - 3 parameters
    kappa_fn = make_kappa_objective_function(observed_signals, sigma)
    kappa_toml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input_kappa.toml")
    kappa_result, logZ_kappa = run_pamc(kappa_toml, kappa_fn, "Kappa")

    # Compute Bayes factor
    print("\n" + "=" * 60)
    print("MODEL SELECTION RESULTS")
    print("=" * 60)

    if maxwell_result and "fx" in maxwell_result:
        print(f"Maxwell model: best chi2 = {maxwell_result['fx']:.4f}")
        print(f"  Best Te = {maxwell_result['x'][0]:.2f} eV")
        print(f"  Best ne = {maxwell_result['x'][1]:.4f} x 1e19 m^-3")

    if kappa_result and "fx" in kappa_result:
        print(f"\nKappa model:  best chi2 = {kappa_result['fx']:.4f}")
        print(f"  Best Te    = {kappa_result['x'][0]:.2f} eV")
        print(f"  Best ne    = {kappa_result['x'][1]:.4f} x 1e19 m^-3")
        print(f"  Best kappa = {kappa_result['x'][2]:.2f}")

    print(f"\nLog partition function:")
    print(f"  Maxwell: logZ = {logZ_maxwell:.4f}" if logZ_maxwell is not None else "  Maxwell: logZ = N/A")
    print(f"  Kappa:   logZ = {logZ_kappa:.4f}" if logZ_kappa is not None else "  Kappa:   logZ = N/A")

    if logZ_maxwell is not None and logZ_kappa is not None:
        ln_B = logZ_maxwell - logZ_kappa
        print(f"\nBayes factor (Maxwell vs Kappa):")
        print(f"  ln(B) = logZ_Maxwell - logZ_Kappa = {ln_B:.4f}")

        if abs(ln_B) < 1:
            evidence = "Not worth more than a bare mention"
        elif abs(ln_B) < 3:
            evidence = "Positive evidence"
        elif abs(ln_B) < 5:
            evidence = "Strong evidence"
        else:
            evidence = "Very strong / decisive evidence"

        favored = "Maxwell" if ln_B > 0 else "Kappa"
        print(f"  Favored model: {favored}")
        print(f"  Strength: {evidence}")
        print(f"\n  (Since the data was generated from Maxwell distribution,")
        print(f"   we expect strong evidence favoring Maxwell.)")


if __name__ == "__main__":
    main()
