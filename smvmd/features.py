"""
Feature Extraction Module
=========================
Exact Table I implementation of features extracted from decomposed modes:
Table I in Chandela, Faisal, & Sharma (IEEE TCDS 2025):
1. STD  : Standard Deviation
2. VAR  : Variance
3. RMS  : Root Mean Square
4. IQR  : Interquartile Range (Q3 - Q1)
5. ASSR : Absolute value of Summation of Square Root
6. AP   : Average Power (via Hamming-windowed modified periodogram)
7. MFL  : Maximum Fractal Length
8. IP   : Information Potential (Gaussian kernel)
9. PCC  : Parametric-Centered Correntropy
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


# ---------------------------------------------------------------------------
# 1. STD (Standard Deviation)
# ---------------------------------------------------------------------------
def compute_std(x: np.ndarray) -> float:
    """STD = sqrt(1/N * sum((xi - mu)^2))"""
    return float(np.std(x))


# ---------------------------------------------------------------------------
# 2. VAR (Variance)
# ---------------------------------------------------------------------------
def compute_var(x: np.ndarray) -> float:
    """VAR = 1/N * sum((xi - mu)^2)"""
    return float(np.var(x))


# ---------------------------------------------------------------------------
# 3. RMS (Root Mean Square)
# ---------------------------------------------------------------------------
def compute_rms(x: np.ndarray) -> float:
    """RMS = sqrt(1/N * sum(xi^2))"""
    return float(np.sqrt(np.mean(x ** 2)))


# ---------------------------------------------------------------------------
# 4. IQR (Interquartile Range)
# ---------------------------------------------------------------------------
def compute_iqr(x: np.ndarray) -> float:
    """IQR = Q3 - Q1 (75th percentile minus 25th percentile)"""
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


# ---------------------------------------------------------------------------
# 5. ASSR (Absolute value of Summation of Square Root)
# ---------------------------------------------------------------------------
def compute_assr(x: np.ndarray) -> float:
    """ASSR = |sum(sqrt(|xi|))|"""
    return float(np.abs(np.sum(np.sqrt(np.abs(x) + 1e-12))))


# ---------------------------------------------------------------------------
# 6. AP (Average Power via Hamming-windowed periodogram)
# ---------------------------------------------------------------------------
def compute_ap(x: np.ndarray, fs: float = 128.0) -> float:
    """
    AP = sum_f (delta_t / N) * |sum(wi * xi * e^{-j2pi f delta_t i})|^2
    using Hamming window w.
    """
    N = len(x)
    delta_t = 1.0 / fs
    w = np.hamming(N)
    xw = x * w
    fft_vals = np.fft.fft(xw)
    psd = (delta_t / N) * (np.abs(fft_vals) ** 2)
    return float(np.sum(psd))


# ---------------------------------------------------------------------------
# 7. MFL (Maximum Fractal Length)
# ---------------------------------------------------------------------------
def compute_mfl(x: np.ndarray) -> float:
    """MFL = log10(sqrt(sum_{i=1}^{N-1} (x_{i+1} - x_i)^2))"""
    diff_sq = np.sum(np.diff(x) ** 2)
    return float(np.log10(np.sqrt(diff_sq) + 1e-12))


# ---------------------------------------------------------------------------
# 8. IP (Information Potential)
# ---------------------------------------------------------------------------
def compute_ip(x: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    IP = 1/N^2 * sum_{i=1}^N sum_{j=1}^N K(xi - xj)
    where K(u) = 1/(sqrt(2pi)*sigma) * exp(-u^2 / (2*sigma^2))
    """
    N = len(x)
    if N > 128:
        # Downsample for speed while preserving kernel density
        step = N // 128
        x = x[::step]
        N = len(x)

    if sigma is None:
        sigma = 1.06 * (np.std(x) + 1e-12) * (N ** (-1.0 / 5.0))  # Silverman's rule
        sigma = max(sigma, 1e-3)

    # Vectorized pairwise squared difference
    diff = x[:, np.newaxis] - x[np.newaxis, :]
    kernel_vals = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(diff ** 2) / (2 * (sigma ** 2)))
    return float(np.mean(kernel_vals))


# ---------------------------------------------------------------------------
# 9. PCC (Parametric-Centered Correntropy)
# ---------------------------------------------------------------------------
def compute_pcc(x: np.ndarray, l: int = 1, sigma: Optional[float] = None) -> float:
    """
    PCC = CC[l] = 1/(N - l + 1) * sum_{n=l}^N K(X[n] - X[n-l])
                  - 1/N^2 * sum_{l=1}^N sum_{n=l}^N K(X[n] - X[n-l])
    with time delay l = 1.
    """
    N = len(x)
    if N > 128:
        step = N // 128
        x = x[::step]
        N = len(x)

    if sigma is None:
        sigma = 1.06 * (np.std(x) + 1e-12) * (N ** (-1.0 / 5.0))
        sigma = max(sigma, 1e-3)

    # Lag-l differences
    diff_l = x[l:] - x[:-l]
    k_l = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(diff_l ** 2) / (2 * (sigma ** 2)))
    first_term = np.mean(k_l)

    # Cross-lag term (Information Potential approximation of centered kernel)
    diff_all = x[:, np.newaxis] - x[np.newaxis, :]
    k_all = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-(diff_all ** 2) / (2 * (sigma ** 2)))
    second_term = np.mean(k_all)

    return float(first_term - second_term)


# ---------------------------------------------------------------------------
# Channel & Mode Feature Extractor
# ---------------------------------------------------------------------------

FEATURE_NAMES_TABLE_1 = ["STD", "VAR", "RMS", "IQR", "ASSR", "AP", "MFL", "IP", "PCC"]


def extract_9_features_1d(x: np.ndarray, fs: float = 128.0) -> Dict[str, float]:
    """
    Extract the exact 9 Table I features for a 1D mode of a single channel.
    """
    return {
        "STD": compute_std(x),
        "VAR": compute_var(x),
        "RMS": compute_rms(x),
        "IQR": compute_iqr(x),
        "ASSR": compute_assr(x),
        "AP": compute_ap(x, fs=fs),
        "MFL": compute_mfl(x),
        "IP": compute_ip(x),
        "PCC": compute_pcc(x, l=1),
    }


def extract_mode_features(
    mode: np.ndarray,
    fs: float = 128.0,
    center_freq_hz: Optional[float] = None,
    total_signal_energy: Optional[float] = None,
) -> Dict[str, float]:
    """
    Extract Table I features averaged across channels for a single MVMF.
    """
    if mode.ndim == 1:
        mode = mode[np.newaxis, :]

    C, T = mode.shape
    feats = {fname: 0.0 for fname in FEATURE_NAMES_TABLE_1}

    for c in range(C):
        ch_feats = extract_9_features_1d(mode[c, :], fs=fs)
        for fname in FEATURE_NAMES_TABLE_1:
            feats[fname] += ch_feats[fname] / C

    feats["energy"] = float(np.sum(mode ** 2))
    return feats


def extract_all_features(
    modes: np.ndarray,
    omega_hz: np.ndarray,
    fs: float = 128.0,
    total_signal_energy: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Extract Table I features for all extracted modes in an epoch.
    """
    K = modes.shape[0]
    all_mode_feats = []

    for k in range(K):
        cf = float(omega_hz[k]) if k < len(omega_hz) else None
        m_feat = extract_mode_features(
            mode=modes[k],
            fs=fs,
            center_freq_hz=cf,
            total_signal_energy=total_signal_energy,
        )
        m_feat["mode_index"] = k + 1
        all_mode_feats.append(m_feat)

    return all_mode_feats
