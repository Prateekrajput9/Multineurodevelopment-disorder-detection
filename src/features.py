"""
features.py - Feature Extraction from SMVMD Modes

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

Features extracted (Table I, paper explicit):
  1. STD  - Standard Deviation
  2. VAR  - Variance
  3. RMS  - Root Mean Square
  4. IQR  - Interquartile Range
  5. ASSR - Absolute Summation of Square Root
  6. AP   - Average Power (modified periodogram, Hamming window)
  7. MFL  - Maximum Fractal Length
  8. IP   - Information Potential (Gaussian kernel)
  9. PCC  - Parametric-Centered Correntropy (l=1 delay)

For each segment:
  Input: K MIMFs, each shape (C, T)
  Output: feature[c][k][f] for each channel c, mode k, feature f
  Dimensionality before energy integration: C × K × 9

Note: K is variable (different segments have different numbers of modes).
This is resolved by the energy-based feature integration step (energy_integration.py).

Mathematical definitions follow Table I of the paper exactly.
Where the paper references MATLAB toolboxes (ITL, EEG, EMG),
the mathematical formula is implemented directly from Table I.
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
from scipy.signal import periodogram

logger = logging.getLogger(__name__)

# Feature names in fixed order (matches paper Table I)
FEATURE_NAMES = ['STD', 'VAR', 'RMS', 'IQR', 'ASSR', 'AP', 'MFL', 'IP', 'PCC']
N_FEATURES = 9  # Paper explicitly lists 9 features


# ─────────────────────────────────────────────────────────────────────────────
# Individual Feature Functions (Paper Table I)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Shared Gaussian-kernel machinery for IP and PCC
# ─────────────────────────────────────────────────────────────────────────────

def silverman_sigma(x: np.ndarray) -> Optional[float]:
    """Silverman's rule-of-thumb kernel bandwidth: sigma = 1.06 * std(x) * N^(-1/5).

    [Paper does not specify sigma for IP/PCC; implementation choice]

    Returns None for a flat signal (std ~ 0), for which IP and PCC are
    defined to be 0.0 by convention here.
    """
    N = len(x)
    if N == 0:
        return None
    std_x = np.std(x)
    if std_x < 1e-10:
        return None
    sigma = 1.06 * std_x * (N ** (-0.2))
    return float(sigma) if sigma >= 1e-10 else None


def gaussian_pair_sum(x: np.ndarray, sigma: float) -> float:
    """Full double sum S = sum_i sum_j K(x_i - x_j) with a Gaussian kernel.

    K(u) = (1 / (sqrt(2*pi) * sigma)) * exp(-u^2 / (2 * sigma^2))

    This single O(N^2) quantity serves BOTH features:
      - IP  = S / N^2                        (Paper Table I, directly)
      - PCC centering term = (S - N*K(0)) / (2*N^2)

    The PCC identity holds because K depends only on |x_i - x_j|, so the
    double sum over all lags l = 1..N-1 of K(x[n] - x[n-l]) equals the sum
    over all unordered pairs i != j, which is (S - N*K(0)) / 2.
    Computing it this way is exact over every lag AND costs nothing extra.
    """
    diff = x[:, np.newaxis] - x[np.newaxis, :]
    kernel = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * (diff / sigma) ** 2)
    return float(np.sum(kernel))


def compute_std(x: np.ndarray) -> float:
    """Standard Deviation (STD).

    Paper Table I:
      STD = sqrt( (1/N) * Σ(x_i - μ)² )

    where μ = mean(x), N = len(x).
    Note: Population std (ddof=0).

    Args:
        x: 1D signal array, N samples.

    Returns:
        STD value.
    """
    return float(np.std(x, ddof=0))


def compute_var(x: np.ndarray) -> float:
    """Variance (VAR).

    Paper Table I:
      VAR = (1/N) * Σ(x_i - μ)²

    Note: Population variance (ddof=0).

    Args:
        x: 1D signal array.

    Returns:
        VAR value.
    """
    return float(np.var(x, ddof=0))


def compute_rms(x: np.ndarray) -> float:
    """Root Mean Square (RMS).

    Paper Table I:
      RMS = sqrt( (1/N) * Σ x_i² )

    Args:
        x: 1D signal array.

    Returns:
        RMS value.
    """
    return float(np.sqrt(np.mean(x ** 2)))


def compute_iqr(x: np.ndarray) -> float:
    """Interquartile Range (IQR).

    Paper Table I:
      IQR = Q3 - Q1

    where Q3 = 75th percentile, Q1 = 25th percentile.

    Args:
        x: 1D signal array.

    Returns:
        IQR value.
    """
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def compute_assr(x: np.ndarray) -> float:
    """Absolute Summation of Square Root (ASSR).

    Paper Table I:
      ASSR = |Σ sqrt(x_i)|

    INTERPRETATION NOTE:
    EEG signals contain negative values. The paper formula √(x_i) is
    ambiguous for x_i < 0. We interpret this as √|x_i| to handle
    negative amplitudes, consistent with the EEG toolbox [41] convention.

    [Paper does not specify handling of negative values; implementation choice]

    Args:
        x: 1D signal array (may contain negative values).

    Returns:
        ASSR value.
    """
    # Sum of square roots of absolute values, then take absolute value of sum
    return float(np.abs(np.sum(np.sqrt(np.abs(x)))))


def compute_ap(x: np.ndarray,
               fs: float = 128.0,
               window: str = 'hamming') -> float:
    """Average Power (AP) via modified periodogram.

    Paper Table I:
      AP = Σ_f |Σ_{i=1}^N w_i * x_i * e^{-j2πfΔt*i}|²

    where w is a Hamming window of N samples.

    Uses the modified periodogram (Welch-style single segment).
    "It is computed using the modified periodogram technique, which estimates
    the power spectral density. This involves multiplying the input EEG time
    series by a non-negative window function to reduce leakage."

    [Paper: Hamming window, explicit]
    [Paper does not specify nfft; implementation choice: nfft = N]
    [Paper does not specify normalization; implementation: mean of PSD]

    Args:
        x: 1D signal array, N samples.
        fs: Sampling frequency in Hz.
        window: Window function name (paper: 'hamming').

    Returns:
        Average power value.
    """
    N = len(x)
    if N == 0:
        return 0.0

    # Create Hamming window [Paper explicit]
    if window == 'hamming':
        w = np.hamming(N)
    elif window == 'hann':
        w = np.hanning(N)
    else:
        w = np.ones(N)

    # Apply window
    x_windowed = x * w

    # Compute periodogram (PSD)
    # scipy.signal.periodogram normalizes by window power
    freqs, psd = periodogram(x_windowed, fs=fs, window='boxcar',
                              nfft=N, scaling='density')

    # Average power = mean of PSD across all frequencies
    ap = float(np.mean(psd))
    return ap


def compute_mfl(x: np.ndarray) -> float:
    """Maximum Fractal Length (MFL).

    Paper Table I:
      MFL = log10( sqrt(Σ_{i=1}^{N-1} (x_{i+1} - x_i)²) )

    Also known as Higuchi Fractal Dimension variant.
    Captures scale-dependent complexity of EEG signals.

    [Paper Table I formula is explicit; standard MFL definition from EMG toolbox [42]]

    Edge case: If signal has zero variation (flat line), log10(0) = -inf.
    We return 0.0 in this case (flat signal → no fractal complexity).
    [Paper does not specify handling; implementation choice]

    Args:
        x: 1D signal array, N samples.

    Returns:
        MFL value (0.0 if signal is flat).
    """
    diff = np.diff(x)  # x_{i+1} - x_i, length N-1
    sum_sq = np.sum(diff ** 2)

    if sum_sq <= 0:
        return 0.0  # Flat signal: return 0 instead of -inf

    mfl = np.log10(np.sqrt(sum_sq))
    return float(mfl)


def compute_ip(x: np.ndarray,
               sigma: Optional[float] = None,
               pair_sum: Optional[float] = None) -> float:
    """Information Potential (IP).

    Paper Table I:
      IP = (1/N^2) * sum_i sum_j K(x_i, x_j)

    where K(x_i, x_j) = (1/(sqrt(2*pi)*sigma)) * exp(-(x_i - x_j)^2 / (2*sigma^2))

    (Gaussian Parzen kernel density estimator)

    IP measures the amount of information contained in a signal.
    Originates from Information Theoretic Learning (ITL) toolbox [40].

    Bandwidth sigma: Paper does not specify.
    Implementation choice: Silverman's rule of thumb:
      sigma = 1.06 * std(x) * N^(-1/5)

    [Silverman's rule is the standard KDE bandwidth; consistent with ITL toolbox default]

    Args:
        x: 1D signal array, N samples.
        sigma: Kernel bandwidth. If None, use Silverman's rule.
        pair_sum: Precomputed sum_i sum_j K(x_i - x_j), to avoid recomputing
            the O(N^2) kernel sum when IP and PCC are extracted together.

    Returns:
        IP value (0.0 for an empty or flat signal).
    """
    N = len(x)
    if N == 0:
        return 0.0

    if sigma is None:
        sigma = silverman_sigma(x)
    if sigma is None or sigma < 1e-10:
        return 0.0

    if pair_sum is None:
        pair_sum = gaussian_pair_sum(x, sigma)

    return float(pair_sum / (N ** 2))


def compute_pcc(x: np.ndarray,
                sigma: Optional[float] = None,
                time_delay: int = 1,
                pair_sum: Optional[float] = None) -> float:
    """Parametric-Centered Correntropy (PCC).

    Paper Table I (with l = time_delay = 1):
      CC[l] = (1/N) * sum_{n=l}^{N} K(X[n] - X[n-l])
               - (1/N^2) * sum_{l=1}^{N} sum_{n=l}^{N} K(X[n] - X[n-l])

    where l = 1 (paper explicitly states this).
    K is the same Gaussian kernel as in IP.

    PCC measures similarity between two time-shifted versions of the signal.
    Captures nonlinear temporal dependencies.

    [Paper: l=1 explicit; sigma not specified; implementation choice: Silverman's rule]

    Formula interpretation:
    - First term: correntropy at lag l = 1.
    - Second term: the centering term, a double sum over EVERY lag l = 1..N.
      Because the Gaussian kernel depends only on |X[n] - X[n-l]|, that double
      sum equals the sum over all unordered sample pairs, i.e.
      (S - N*K(0)) / 2 where S = sum_i sum_j K(x_i - x_j). This closed form is
      exact over all lags and reuses the same kernel sum that IP needs.

    Args:
        x: 1D signal array, N samples.
        sigma: Kernel bandwidth. If None, use Silverman's rule.
        time_delay: Time delay l (paper: l=1, explicit).
        pair_sum: Precomputed sum_i sum_j K(x_i - x_j) (shared with IP).

    Returns:
        PCC value.
    """
    N = len(x)
    if N <= time_delay:
        return 0.0

    if sigma is None:
        sigma = silverman_sigma(x)
    if sigma is None or sigma < 1e-10:
        return 0.0

    norm = 1.0 / (np.sqrt(2 * np.pi) * sigma)

    # First term: CC at lag l, mean of K(x[n] - x[n-l])
    l = time_delay  # l = 1 from paper
    lag_diff = x[l:] - x[:-l]
    first_term = float(np.mean(norm * np.exp(-0.5 * (lag_diff / sigma) ** 2)))

    # Second term: centering over ALL lags, in closed form (see docstring)
    if pair_sum is None:
        pair_sum = gaussian_pair_sum(x, sigma)
    k_zero = norm  # K(0)
    second_term = (pair_sum - N * k_zero) / (2.0 * N ** 2)

    return float(first_term - second_term)


# ─────────────────────────────────────────────────────────────────────────────
# Per-Mode Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_features_from_mode(mode: np.ndarray,
                                channel_idx: int,
                                fs: float = 128.0,
                                sigma: Optional[float] = None) -> np.ndarray:
    """Extract all 9 features from one channel of one MIMF mode.

    For each mode k and channel c:
    Computes: [STD, VAR, RMS, IQR, ASSR, AP, MFL, IP, PCC]

    Args:
        mode: 1D signal array for one channel, one mode (shape: T,).
        channel_idx: Channel index (for logging/debugging).
        fs: Sampling frequency.
        sigma: Kernel bandwidth for IP and PCC (None = Silverman's rule).

    Returns:
        Feature vector of length 9.
    """
    features = np.zeros(N_FEATURES)

    x = mode.astype(np.float64)

    try:
        features[0] = compute_std(x)
        features[1] = compute_var(x)
        features[2] = compute_rms(x)
        features[3] = compute_iqr(x)
        features[4] = compute_assr(x)
        features[5] = compute_ap(x, fs=fs)
        features[6] = compute_mfl(x)

        # IP and PCC share one O(N^2) Gaussian kernel sum (the hot path).
        sigma_k = sigma if sigma is not None else silverman_sigma(x)
        pair_sum = gaussian_pair_sum(x, sigma_k) if sigma_k is not None else None
        features[7] = compute_ip(x, sigma=sigma_k, pair_sum=pair_sum)
        features[8] = compute_pcc(x, sigma=sigma_k, time_delay=1,
                                  pair_sum=pair_sum)  # l=1 from paper
    except Exception as e:
        logger.warning(f"Feature extraction error (channel {channel_idx}): {e}")
        features = np.zeros(N_FEATURES)

    return features


def extract_features_from_segment(mimfs: List[np.ndarray],
                                   fs: float = 128.0,
                                   sigma: Optional[float] = None
                                   ) -> np.ndarray:
    """Extract features from all modes and channels of one segment.

    For a segment with K modes and C channels:
    Output shape: (C, K, 9)

    This is the pre-integration feature array.
    Energy integration (in energy_integration.py) collapses K dimension.

    Args:
        mimfs: List of K MIMFs, each shape (C, T).
        fs: Sampling frequency (128 Hz).
        sigma: Kernel bandwidth (None = Silverman's rule).

    Returns:
        Feature array of shape (C, K, 9).
        feature[c, k, f] = fth feature of kth mode of cth channel.
    """
    if not mimfs:
        return np.empty((0, 0, N_FEATURES))

    K = len(mimfs)
    C = mimfs[0].shape[0]

    feature_array = np.zeros((C, K, N_FEATURES))

    for k, mimf in enumerate(mimfs):
        for c in range(C):
            feature_array[c, k, :] = extract_features_from_mode(
                mode=mimf[c],
                channel_idx=c,
                fs=fs,
                sigma=sigma
            )

    # Check for NaN/Inf
    if np.any(np.isnan(feature_array)):
        n_nan = np.sum(np.isnan(feature_array))
        logger.warning(f"Feature array contains {n_nan} NaN values. Replacing with 0.")
        feature_array = np.nan_to_num(feature_array, nan=0.0)

    if np.any(np.isinf(feature_array)):
        n_inf = np.sum(np.isinf(feature_array))
        logger.warning(f"Feature array contains {n_inf} Inf values. Replacing with 0.")
        feature_array = np.nan_to_num(feature_array, posinf=0.0, neginf=0.0)

    return feature_array


def _extract_one(args):
    """Worker for parallel feature extraction from a single SMVMDResult."""
    result, fs, sigma = args
    if result.n_modes == 0 or not result.mimfs:
        return np.empty((0, 0, N_FEATURES))
    return extract_features_from_segment(mimfs=result.mimfs, fs=fs, sigma=sigma)


def extract_features_batch(smvmd_results: List,
                            fs: float = 128.0,
                            sigma: Optional[float] = None,
                            verbose: bool = True,
                            n_jobs: int = -1
                            ) -> List[np.ndarray]:
    """Extract features from all SMVMD results (parallelised).

    Args:
        smvmd_results: List of SMVMDResult objects.
        fs: Sampling frequency.
        sigma: Kernel bandwidth.
        verbose: Show progress bar.
        n_jobs: Number of parallel workers (-1 = all cores).

    Returns:
        List of feature arrays, one per segment.
        Each array has shape (C, K_i, 9) where K_i varies.
    """
    from tqdm import tqdm
    from joblib import Parallel, delayed

    n = len(smvmd_results)
    logger.info(f"Extracting features from {n} segments (parallel, n_jobs={n_jobs})...")

    args_list = [(r, fs, sigma) for r in smvmd_results]

    feature_list = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_extract_one)(args)
        for args in tqdm(args_list, desc="Feature extraction", disable=not verbose)
    )

    n_empty = sum(1 for f in feature_list if f.size == 0)
    if n_empty:
        logger.warning(f"{n_empty} segments had 0 modes and returned empty features.")

    return feature_list


def get_feature_summary(feature_list: List[np.ndarray]) -> Dict[str, Any]:
    """Compute summary statistics over extracted features.

    Args:
        feature_list: List of per-segment feature arrays (C, K, 9).

    Returns:
        Summary dict with statistics.
    """
    n_segments = len(feature_list)
    k_values = [f.shape[1] for f in feature_list if f.ndim == 3]

    summary = {
        'n_segments': n_segments,
        'n_with_modes': sum(1 for f in feature_list if f.ndim == 3 and f.shape[1] > 0),
        'k_stats': {
            'min': int(np.min(k_values)) if k_values else 0,
            'max': int(np.max(k_values)) if k_values else 0,
            'mean': float(np.mean(k_values)) if k_values else 0,
        },
        'feature_names': FEATURE_NAMES,
        'n_features_per_channel_per_mode': N_FEATURES,
    }

    return summary
