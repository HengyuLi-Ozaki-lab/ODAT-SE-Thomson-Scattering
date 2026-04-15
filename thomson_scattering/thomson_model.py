"""
Thomson Scattering Forward Model for ODAT-SE

This module implements the forward model for non-relativistic Thomson scattering
diagnostics in a polychromator configuration. It provides:
  - Gaussian spectral shape calculation for Thomson scattered light
  - 5-channel polychromator response model with Gaussian bandpass filters
  - Chi-squared objective function compatible with ODAT-SE's function solver

Physics:
  The Thomson scattering spectrum from a Maxwellian plasma has a Gaussian shape:
    S(lambda) ~ exp(-0.5 * (delta_lambda / sigma_lambda)^2)
  where sigma_lambda = lambda_0 * sqrt(2 * Te / (m_e * c^2))
  and lambda_0 = 1064 nm (Nd:YAG laser wavelength).

Reference:
  Morishita et al., arXiv:2511.06330 (2025)
"""

import numpy as np


# --- Physical Constants ---
LAMBDA_LASER = 1064.0  # nm, Nd:YAG laser wavelength
ME_C2_EV = 511.0e3     # electron rest mass energy in eV

# --- Polychromator Configuration ---
# 5 channels with Gaussian bandpass filters
# Centers chosen to sample the Thomson spectrum around 1064 nm
FILTER_CENTERS = np.array([900.0, 960.0, 1020.0, 1120.0, 1200.0])  # nm
FILTER_FWHM = 40.0  # nm, full width at half maximum for all channels
FILTER_SIGMA = FILTER_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # ~17 nm
N_CHANNELS = 5

# Wavelength integration grid
LAMBDA_MIN = 750.0   # nm
LAMBDA_MAX = 1400.0  # nm
N_LAMBDA = 2000
WAVELENGTHS = np.linspace(LAMBDA_MIN, LAMBDA_MAX, N_LAMBDA)


def compute_filter_responses(wavelengths=WAVELENGTHS):
    """
    Compute the transmission function T_i(lambda) for each polychromator channel.

    Returns
    -------
    filters : np.ndarray, shape (N_CHANNELS, len(wavelengths))
        Filter transmission values for each channel at each wavelength.
    """
    filters = np.zeros((N_CHANNELS, len(wavelengths)))
    for i in range(N_CHANNELS):
        delta = wavelengths - FILTER_CENTERS[i]
        filters[i] = np.exp(-0.5 * (delta / FILTER_SIGMA) ** 2)
    return filters


# Pre-compute filter responses (constant across all evaluations)
FILTER_RESPONSES = compute_filter_responses()


def thomson_spectrum(wavelengths, Te_eV):
    """
    Compute the non-relativistic Thomson scattering spectral shape.

    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelength array in nm.
    Te_eV : float
        Electron temperature in eV.

    Returns
    -------
    spectrum : np.ndarray
        Normalized spectral density (1/nm) at each wavelength.
    """
    if Te_eV <= 0:
        return np.zeros_like(wavelengths)

    # Doppler width: sigma_lambda = lambda_0 * sqrt(2 * Te / (m_e c^2))
    sigma_lambda = LAMBDA_LASER * np.sqrt(2.0 * Te_eV / ME_C2_EV)

    delta_lambda = wavelengths - LAMBDA_LASER
    spectrum = np.exp(-0.5 * (delta_lambda / sigma_lambda) ** 2) / (
        sigma_lambda * np.sqrt(2.0 * np.pi)
    )
    return spectrum


def compute_channel_signals(Te_eV, ne_1e19):
    """
    Compute expected photoelectron signals in each polychromator channel.

    Parameters
    ----------
    Te_eV : float
        Electron temperature in eV.
    ne_1e19 : float
        Electron density in units of 1e19 m^-3.

    Returns
    -------
    signals : np.ndarray, shape (N_CHANNELS,)
        Expected signal in each channel (arbitrary units, proportional to ne).
    """
    spectrum = thomson_spectrum(WAVELENGTHS, Te_eV)
    signals = np.zeros(N_CHANNELS)
    for i in range(N_CHANNELS):
        integrand = spectrum * FILTER_RESPONSES[i]
        signals[i] = ne_1e19 * np.trapz(integrand, WAVELENGTHS)
    return signals


def make_objective_function(observed_signals, sigma):
    """
    Create a chi-squared objective function for ODAT-SE.

    Parameters
    ----------
    observed_signals : np.ndarray, shape (N_CHANNELS,)
        Observed (synthetic or experimental) signals.
    sigma : np.ndarray, shape (N_CHANNELS,)
        Measurement uncertainties for each channel.

    Returns
    -------
    objective : callable
        Function f(x) -> float where x = [Te_eV, ne_1e19].
        Returns the chi-squared value.
    """
    def objective(x):
        Te_eV = x[0]
        ne_1e19 = x[1]

        # Physical bounds check
        if Te_eV <= 0 or ne_1e19 <= 0:
            return 1.0e10

        model_signals = compute_channel_signals(Te_eV, ne_1e19)
        residuals = (observed_signals - model_signals) / sigma
        chi2 = np.sum(residuals ** 2)
        return chi2

    return objective


def make_kappa_objective_function(observed_signals, sigma):
    """
    Create a chi-squared objective function for the Kappa distribution model.

    The Kappa distribution has power-law tails and reduces to Maxwellian
    as kappa -> infinity. It introduces one additional parameter.

    Parameters
    ----------
    observed_signals : np.ndarray
        Observed signals.
    sigma : np.ndarray
        Measurement uncertainties.

    Returns
    -------
    objective : callable
        Function f(x) -> float where x = [Te_eV, ne_1e19, kappa].
    """
    from scipy.special import gamma as gamma_func

    def kappa_spectrum(wavelengths, Te_eV, kappa):
        """Kappa distribution Thomson scattering spectrum."""
        if Te_eV <= 0 or kappa <= 1.5:
            return np.zeros_like(wavelengths)

        sigma_lambda = LAMBDA_LASER * np.sqrt(2.0 * Te_eV / ME_C2_EV)
        delta_lambda = wavelengths - LAMBDA_LASER
        u2 = (delta_lambda / sigma_lambda) ** 2

        # Kappa distribution spectral shape
        norm = (
            gamma_func(kappa + 1)
            / (gamma_func(kappa - 0.5) * np.sqrt(np.pi * kappa) * sigma_lambda)
        )
        spectrum = norm * (1.0 + u2 / kappa) ** (-(kappa + 1))
        return spectrum

    def objective(x):
        Te_eV = x[0]
        ne_1e19 = x[1]
        kappa = x[2]

        if Te_eV <= 0 or ne_1e19 <= 0 or kappa <= 1.5:
            return 1.0e10

        spectrum = kappa_spectrum(WAVELENGTHS, Te_eV, kappa)
        signals = np.zeros(N_CHANNELS)
        for i in range(N_CHANNELS):
            integrand = spectrum * FILTER_RESPONSES[i]
            signals[i] = ne_1e19 * np.trapz(integrand, WAVELENGTHS)

        residuals = (observed_signals - signals) / sigma
        chi2 = np.sum(residuals ** 2)
        return chi2

    return objective


if __name__ == "__main__":
    # Quick validation of the forward model
    Te_test = 500.0  # eV
    ne_test = 3.0    # 1e19 m^-3

    print("=== Thomson Scattering Forward Model Validation ===")
    print(f"Te = {Te_test} eV, ne = {ne_test} x 1e19 m^-3")
    print(f"Laser wavelength = {LAMBDA_LASER} nm")

    sigma_lambda = LAMBDA_LASER * np.sqrt(2.0 * Te_test / ME_C2_EV)
    print(f"Spectral width (sigma_lambda) = {sigma_lambda:.1f} nm")

    signals = compute_channel_signals(Te_test, ne_test)
    print(f"\nChannel centers (nm): {FILTER_CENTERS}")
    print(f"Channel signals:      {signals}")
    print(f"Signal sum:           {np.sum(signals):.6f}")
    print(f"Max signal:           {np.max(signals):.6f} (channel {np.argmax(signals)+1})")
