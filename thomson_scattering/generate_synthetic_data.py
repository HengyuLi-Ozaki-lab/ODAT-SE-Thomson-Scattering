"""
Generate synthetic Thomson scattering polychromator data.

Creates noisy synthetic data from known (true) plasma parameters,
simulating a typical tokamak Thomson scattering measurement.

True parameters:
  - Te = 500 eV (electron temperature)
  - ne = 3.0 x 10^19 m^-3 (electron density)
  - SNR = 20 (signal-to-noise ratio per channel)
"""

import numpy as np
import os
from thomson_model import compute_channel_signals, FILTER_CENTERS, N_CHANNELS

# True plasma parameters
TE_TRUE = 500.0   # eV
NE_TRUE = 3.0     # 1e19 m^-3
SNR = 20           # signal-to-noise ratio

# Random seed for reproducibility
SEED = 42


def generate_data():
    """Generate synthetic Thomson scattering data with noise."""
    np.random.seed(SEED)

    # Compute true (noise-free) signals
    true_signals = compute_channel_signals(TE_TRUE, NE_TRUE)

    # Noise model: Gaussian noise with sigma = signal / SNR
    sigma = true_signals / SNR
    noise = sigma * np.random.randn(N_CHANNELS)
    observed_signals = true_signals + noise

    return true_signals, observed_signals, sigma


def main():
    true_signals, observed_signals, sigma = generate_data()

    print("=" * 60)
    print("Synthetic Thomson Scattering Data Generation")
    print("=" * 60)
    print(f"True parameters: Te = {TE_TRUE} eV, ne = {NE_TRUE} x 1e19 m^-3")
    print(f"SNR = {SNR}, Random seed = {SEED}")
    print(f"Number of channels = {N_CHANNELS}")
    print()

    print("Channel | Center(nm) | True Signal | Observed | Sigma")
    print("-" * 60)
    for i in range(N_CHANNELS):
        print(
            f"  {i+1:2d}    |  {FILTER_CENTERS[i]:7.1f}   | "
            f"{true_signals[i]:10.6f} | {observed_signals[i]:10.6f} | {sigma[i]:10.6f}"
        )
    print()

    # Save data
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(data_dir, "synthetic_data.npz")
    np.savez(
        data_path,
        true_signals=true_signals,
        observed_signals=observed_signals,
        sigma=sigma,
        Te_true=TE_TRUE,
        ne_true=NE_TRUE,
        filter_centers=FILTER_CENTERS,
    )
    print(f"Data saved to: {data_path}")

    # Print values for hardcoding (useful for TOML reference files)
    print("\n--- Values for hardcoding ---")
    print(f"observed_signals = {observed_signals.tolist()}")
    print(f"sigma = {sigma.tolist()}")


if __name__ == "__main__":
    main()
