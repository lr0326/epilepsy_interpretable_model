"""
Handcrafted EEG Feature Extraction
====================================
Extracts 15 physiologically meaningful features per channel per window.

Feature list (``FEATURE_NAMES``):
  0  log_delta_power      – log band power  0.5–4  Hz
  1  log_theta_power      – log band power  4–8    Hz
  2  log_alpha_power      – log band power  8–13   Hz
  3  log_beta_power       – log band power  13–30  Hz
  4  log_gamma_power      – log band power  30–70  Hz
  5  theta_alpha_ratio    – theta / alpha power ratio
  6  delta_alpha_ratio    – delta / alpha power ratio
  7  slow_fast_ratio      – (delta+theta) / (alpha+beta) ratio
  8  gamma_total_ratio    – gamma / total power ratio
  9  spectral_entropy     – entropy of the normalised PSD
  10 sample_entropy       – SampEn(m=2, r=0.2*std)
  11 higuchi_fd           – Higuchi Fractal Dimension
  12 mean_plv_broadband   – mean PLV of this channel with all others (broadband)
  13 mean_plv_alpha       – mean PLV of this channel with all others (alpha band)
  14 mean_plv_gamma       – mean PLV of this channel with all others (gamma band)

Public API:
  ``extract_handcrafted_features(eeg_window, fs)``  →  [C, 15]
  ``extract_features_batch(eeg_sequences, fs)``     →  [S, N, C, 15]
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.signal import hilbert

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ─── Frequency band definitions ──────────────────────────────────────────────

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 70.0),
}

FEATURE_NAMES: List[str] = [
    "log_delta_power",
    "log_theta_power",
    "log_alpha_power",
    "log_beta_power",
    "log_gamma_power",
    "theta_alpha_ratio",
    "delta_alpha_ratio",
    "slow_fast_ratio",
    "gamma_total_ratio",
    "spectral_entropy",
    "sample_entropy",
    "higuchi_fd",
    "mean_plv_broadband",
    "mean_plv_alpha",
    "mean_plv_gamma",
]

N_FEATURES: int = len(FEATURE_NAMES)  # 15


# ─── Low-level helpers ────────────────────────────────────────────────────────

def _bandpass_filter(
    x: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter for a 1-D signal."""
    nyq = fs / 2.0
    low  = lowcut  / nyq
    high = min(highcut / nyq, 0.99)
    if low >= high or low <= 0:
        return x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b, a = signal.butter(order, [low, high], btype="band")
        return signal.filtfilt(b, a, x)


def _band_power(
    x: np.ndarray,
    fs: float,
    band: Tuple[float, float],
) -> float:
    """Power in *band* estimated via Welch's method (trapezoidal integration)."""
    nperseg = min(256, len(x) // 2)
    if nperseg < 4:
        return 1e-10
    f, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    mask = (f >= band[0]) & (f <= band[1])
    if mask.sum() == 0:
        return 1e-10
    return float(_trapz(psd[mask], f[mask]))


def _spectral_entropy(x: np.ndarray, fs: float) -> float:
    """Spectral (Shannon) entropy of the normalised PSD."""
    nperseg = min(256, len(x) // 2)
    if nperseg < 4:
        return 0.0
    _, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    psd_norm = psd / (psd.sum() + 1e-10)
    return float(-np.sum(psd_norm * np.log(psd_norm + 1e-10)))


def _sample_entropy(
    x: np.ndarray,
    m: int = 2,
    r: Optional[float] = None,
    n_max: int = 128,
) -> float:
    """
    Sample Entropy (SampEn) via vectorised Chebyshev-distance counting.

    Limited to *n_max* samples for efficiency.
    SampEn = −log(A / B) where B = template matches at length m,
    A = template matches at length m+1.
    """
    x = np.asarray(x[:n_max], dtype=np.float64)
    N = len(x)
    if N < m + 2:
        return 0.0
    if r is None:
        std = float(np.std(x))
        r = 0.2 * std if std > 0 else 0.1

    Nm  = N - m
    Nm1 = N - m - 1

    # Build template matrices  [Nm, m] and [Nm1, m+1]
    tmp_m  = np.stack([x[i : i + m]     for i in range(Nm)],  axis=0)
    tmp_m1 = np.stack([x[i : i + m + 1] for i in range(Nm1)], axis=0)

    # Chebyshev distances and counts
    dist_m  = np.max(np.abs(tmp_m[:, None]  - tmp_m[None, :]),  axis=2)
    B = int((dist_m  <= r).sum()) - Nm    # subtract self-matches

    dist_m1 = np.max(np.abs(tmp_m1[:, None] - tmp_m1[None, :]), axis=2)
    A = int((dist_m1 <= r).sum()) - Nm1

    if B <= 0:
        return 0.0
    return float(-np.log((A + 1e-10) / (B + 1e-10)))


def _higuchi_fd(x: np.ndarray, kmax: int = 8) -> float:
    """Higuchi Fractal Dimension of a 1-D time series."""
    N = len(x)
    x = np.asarray(x, dtype=np.float64)
    L_vals: List[float] = []
    for k in range(1, kmax + 1):
        Lk: List[float] = []
        for m in range(1, k + 1):
            idxs = np.arange(m - 1, N, k)
            if len(idxs) < 2:
                continue
            Lmk = (
                np.sum(np.abs(np.diff(x[idxs])))
                * (N - 1)
                / (k * (len(idxs) - 1))
            )
            Lk.append(Lmk)
        if Lk:
            L_vals.append(float(np.mean(Lk)))

    if len(L_vals) < 2:
        return 1.0
    ln_k = np.log(np.arange(1, len(L_vals) + 1, dtype=float))
    ln_L = np.log(np.array(L_vals) + 1e-10)
    slope = float(np.polyfit(ln_k, ln_L, 1)[0])
    return -slope


def _plv_matrix(
    eeg: np.ndarray,
    fs: float,
    band: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Compute the [C, C] Phase-Locking Value matrix.

    PLV[i,j] = |mean_t( exp(i·(φ_i(t) − φ_j(t))) )|

    Uses the Hilbert transform to obtain instantaneous phases.
    Optionally bandpass-filters before PLV computation.
    """
    C, T = eeg.shape
    if band is not None:
        filtered = np.stack(
            [_bandpass_filter(eeg[c], band[0], band[1], fs) for c in range(C)]
        )
    else:
        filtered = eeg

    analytic  = hilbert(filtered, axis=1)         # [C, T]
    phase_exp = np.exp(1j * np.angle(analytic))   # unit complex vectors

    # Vectorised PLV: [C, C] = |mean over T of outer product|
    cross = (phase_exp @ phase_exp.conj().T) / T  # [C, C]
    return np.abs(cross).astype(np.float32)


# ─── Main public functions ───────────────────────────────────────────────────

def extract_handcrafted_features(
    eeg_window: np.ndarray,
    fs: float = 256.0,
    compute_plv: bool = True,
) -> np.ndarray:
    """
    Extract 15 handcrafted features for every channel in a single EEG window.

    Args:
        eeg_window: [C, T] float32/64 array (channels × time samples).
        fs:         Sampling frequency in Hz.
        compute_plv: Whether to compute PLV connectivity features.
                     Can be disabled to speed up preprocessing.

    Returns:
        features: [C, N_FEATURES] float32 array.
    """
    C, T = eeg_window.shape
    features = np.zeros((C, N_FEATURES), dtype=np.float32)

    # ── Band powers ──────────────────────────────────────────────────
    band_powers: Dict[str, np.ndarray] = {}
    for i, (name, brange) in enumerate(FREQ_BANDS.items()):
        bp = np.array(
            [_band_power(eeg_window[c], fs, brange) for c in range(C)],
            dtype=np.float32,
        )
        band_powers[name] = bp
        features[:, i] = np.log(bp + 1e-10)

    # ── Band-power ratios ────────────────────────────────────────────
    total_pw = sum(band_powers.values()) + 1e-10
    features[:, 5] = band_powers["theta"] / (band_powers["alpha"] + 1e-10)
    features[:, 6] = band_powers["delta"] / (band_powers["alpha"] + 1e-10)
    features[:, 7] = (
        (band_powers["delta"] + band_powers["theta"])
        / (band_powers["alpha"] + band_powers["beta"] + 1e-10)
    )
    features[:, 8] = band_powers["gamma"] / total_pw

    # ── Spectral entropy ─────────────────────────────────────────────
    for c in range(C):
        features[c, 9] = _spectral_entropy(eeg_window[c], fs)

    # ── Sample entropy (limited to 128 pts for speed) ────────────────
    for c in range(C):
        features[c, 10] = _sample_entropy(eeg_window[c], n_max=128)

    # ── Higuchi Fractal Dimension ────────────────────────────────────
    for c in range(C):
        features[c, 11] = _higuchi_fd(eeg_window[c])

    # ── PLV connectivity (per-channel mean) ──────────────────────────
    if compute_plv and C > 1:
        plv_bb    = _plv_matrix(eeg_window, fs, band=None)
        plv_alpha = _plv_matrix(eeg_window, fs, band=FREQ_BANDS["alpha"])
        plv_gamma = _plv_matrix(eeg_window, fs, band=FREQ_BANDS["gamma"])

        # Zero out diagonal, then take row mean
        np.fill_diagonal(plv_bb,    0.0)
        np.fill_diagonal(plv_alpha, 0.0)
        np.fill_diagonal(plv_gamma, 0.0)

        features[:, 12] = plv_bb.sum(axis=1)    / (C - 1)
        features[:, 13] = plv_alpha.sum(axis=1) / (C - 1)
        features[:, 14] = plv_gamma.sum(axis=1) / (C - 1)

    return features


def extract_features_batch(
    eeg_sequences: np.ndarray,
    fs: float = 256.0,
    compute_plv: bool = True,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract handcrafted features from a batch of EEG window sequences.

    Args:
        eeg_sequences: [S, N, C, T]  –  S samples, N windows, C channels, T pts.
        fs:            Sampling frequency in Hz.
        compute_plv:   Whether to compute PLV features (slower).
        verbose:       Show tqdm progress bar if available.

    Returns:
        features: [S, N, C, N_FEATURES] float32 array.
    """
    S, N, C, T = eeg_sequences.shape
    features = np.zeros((S, N, C, N_FEATURES), dtype=np.float32)

    try:
        from tqdm import tqdm
        iterator = tqdm(range(S), desc="Extracting features") if verbose else range(S)
    except ImportError:
        iterator = range(S)
        if verbose:
            print(f"Extracting features for {S} samples…")

    for s in iterator:
        for n in range(N):
            features[s, n] = extract_handcrafted_features(
                eeg_sequences[s, n], fs=fs, compute_plv=compute_plv
            )

    return features
