"""
preprocessing.py - EEG Signal Preprocessing

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

DS-1 Preprocessing (MODE A - requires MATLAB EEGLAB):
  1. Band-pass filter: 1–30 Hz [Paper explicit]
  2. CleanLine for 50/60 Hz line noise [Paper explicit; MATLAB-only]
  3. ICA via runica (logistic infomax) [Paper explicit; MATLAB-only]
  4. ADJUST artifact identification [Paper explicit; MATLAB-only]
  5. Remove 2 most problematic ICs [Paper explicit]
  6. Reconstruct EEG [Paper explicit]

  DEFAULT: MODE B – use dataset-provided preprocessed files
  (Paper does not specify which mode to use for reproduction;
   both are implemented here for completeness)

DS-2 Preprocessing [Paper explicit]:
  1. Butterworth band-pass filter: 0.5–60 Hz
  2. 50 Hz notch filter
  3. Coiflet-3 wavelet filtering (ocular noise removal)
  4. SURE (Stein's Unbiased Risk Estimation) denoising, level 6

All Python implementations are as faithful as possible to the paper's
specifications. MATLAB-only steps are noted.
"""

import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# DS-1 Preprocessing (Python approximation of MATLAB EEGLAB pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ds1_mode_a(eeg: np.ndarray,
                           fs: float = 128.0,
                           bandpass_low: float = 1.0,
                           bandpass_high: float = 30.0,
                           line_noise_freqs: List[float] = None,
                           n_ics_to_remove: int = 2
                           ) -> np.ndarray:
    """DS-1 preprocessing pipeline approximation (MODE A).

    This approximates the MATLAB EEGLAB pipeline described in the paper.
    EXACT reproduction is NOT possible without MATLAB + EEGLAB.

    Paper pipeline:
    1. Band-pass filter: 1-30 Hz
    2. CleanLine: 50/60 Hz line noise removal
    3. ICA: runica (logistic infomax with natural gradient)
    4. ADJUST: artifact IC identification
    5. Remove 2 most problematic ICs
    6. Reconstruct EEG

    Python approximation:
    1. Butterworth band-pass filter (same cutoffs)
    2. Zapline-like notch filter for line noise (APPROXIMATE - not CleanLine)
    3. FastICA (APPROXIMATE - not logistic infomax)
    4. Automatic IC rejection (APPROXIMATE - not ADJUST plugin)
    5. Reconstruct

    RECOMMENDATION: Use MODE B (preprocessed files from Mendeley dataset)
    for faithful reproduction.

    [Paper does not specify: exact EEGLAB version, ADJUST thresholds, CleanLine params]

    Args:
        eeg: Raw EEG array (n_channels, n_samples).
        fs: Sampling frequency (default 128 Hz).
        bandpass_low: Lower cutoff (paper: 1 Hz).
        bandpass_high: Upper cutoff (paper: 30 Hz).
        line_noise_freqs: Frequencies for line noise removal (paper: [50, 60] Hz).
        n_ics_to_remove: Number of artifact ICs to remove (paper: 2).

    Returns:
        Preprocessed EEG array (n_channels, n_samples).

    Notes:
        This is a Python APPROXIMATION. The MATLAB EEGLAB pipeline produces
        different results. For faithful reproduction, use MODE B.
    """
    if line_noise_freqs is None:
        line_noise_freqs = [50.0, 60.0]

    logger.warning(
        "DS-1 MODE A: Applying Python approximation of MATLAB EEGLAB pipeline. "
        "This does NOT exactly replicate runica ICA, CleanLine, or ADJUST. "
        "For faithful reproduction, use MODE B (preprocessed files)."
    )

    # Step 1: Band-pass filter [Paper: 1-30 Hz]
    eeg_filtered = bandpass_filter(eeg, fs=fs, low=bandpass_low, high=bandpass_high)

    # Step 2: Line noise removal [Paper: CleanLine at 50/60 Hz; approx with notch]
    for freq in line_noise_freqs:
        if freq < fs / 2:
            eeg_filtered = notch_filter(eeg_filtered, fs=fs, notch_freq=freq)
    logger.info("Applied notch filter (APPROXIMATION of CleanLine)")

    # Step 3-5: ICA artifact removal [Paper: runica + ADJUST; approx with FastICA]
    eeg_clean = _ica_artifact_removal_approx(eeg_filtered, n_remove=n_ics_to_remove)

    return eeg_clean


def _ica_artifact_removal_approx(eeg: np.ndarray,
                                  n_remove: int = 2) -> np.ndarray:
    """Approximate ICA artifact removal.

    APPROXIMATION of runica + ADJUST pipeline.
    Uses scikit-learn FastICA with heuristic artifact IC detection.

    [Paper: MATLAB runica (logistic infomax) + ADJUST plugin]
    [This is NOT equivalent to the paper's method]

    Args:
        eeg: EEG array (n_channels, n_samples).
        n_remove: Number of artifact ICs to remove.

    Returns:
        Cleaned EEG (n_channels, n_samples).
    """
    logger.warning(
        "ICA: Using scikit-learn FastICA as APPROXIMATION of MATLAB runica. "
        "Results will differ from the paper. Use MODE B for reproduction."
    )

    try:
        from sklearn.decomposition import FastICA
    except ImportError:
        logger.error("scikit-learn required for ICA approximation")
        return eeg

    n_channels, n_samples = eeg.shape

    # Fit FastICA
    ica = FastICA(n_components=n_channels, random_state=42, max_iter=1000)
    sources = ica.fit_transform(eeg.T).T  # shape: (n_components, n_samples)
    mixing = ica.mixing_  # shape: (n_channels, n_components)

    # Heuristic: detect artifact ICs by high kurtosis (eye/muscle artifacts)
    # [Paper: ADJUST plugin uses spatial + temporal features; not reproduced here]
    kurtosis_vals = np.array([
        _kurtosis(sources[i]) for i in range(n_channels)
    ])
    artifact_idx = np.argsort(np.abs(kurtosis_vals))[-n_remove:]
    logger.info(
        f"ICA: Removing {n_remove} high-kurtosis components (indices {artifact_idx}). "
        f"[APPROXIMATION: paper uses ADJUST plugin criteria]"
    )

    # Zero out artifact components
    sources_clean = sources.copy()
    sources_clean[artifact_idx, :] = 0.0

    # Reconstruct
    eeg_clean = (mixing @ sources_clean).T
    eeg_clean = eeg_clean.T  # back to (n_channels, n_samples)

    return eeg_clean.astype(np.float64)


def _kurtosis(x: np.ndarray) -> float:
    """Compute excess kurtosis of 1D signal."""
    x_centered = x - np.mean(x)
    std = np.std(x_centered)
    if std < 1e-10:
        return 0.0
    return float(np.mean((x_centered / std) ** 4) - 3)


# ─────────────────────────────────────────────────────────────────────────────
# DS-2 Preprocessing (Fully Python-reproducible)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ds2(eeg: np.ndarray,
                   fs: float = 128.0,
                   bandpass_low: float = 0.5,
                   bandpass_high: float = 60.0,
                   butterworth_order: int = 4,
                   notch_freq: float = 50.0,
                   notch_quality: float = 30.0,
                   wavelet: str = 'coif3',
                   wavelet_level: int = 6,
                   denoising_method: str = 'SURE'
                   ) -> np.ndarray:
    """DS-2 preprocessing pipeline (fully Python-reproducible).

    Paper [explicit]:
    1. Butterworth band-pass filter: 0.5-60 Hz
       "eliminates high-frequency muscle artifacts (65-100 Hz)"
    2. 50 Hz notch filter (power line interference)
    3. Coiflet-3 wavelet filtering (ocular noise removal)
    4. SURE denoising up to wavelet level 6

    [Butterworth filter order not specified; implementation choice: order=4]
    [Notch filter Q factor not specified; implementation choice: Q=30]

    Args:
        eeg: Raw EEG array (n_channels, n_samples).
        fs: Sampling frequency (paper: 128 Hz).
        bandpass_low: Lower cutoff (paper: 0.5 Hz).
        bandpass_high: Upper cutoff (paper: 60 Hz).
        butterworth_order: Filter order [paper does not specify; default 4].
        notch_freq: Notch frequency (paper: 50 Hz).
        notch_quality: Notch Q factor [paper does not specify; default 30].
        wavelet: Wavelet type (paper: 'coif3' = Coiflet-3).
        denoising_method: Thresholding rule (paper: 'SURE'). Accepted so the
            whole `ds2.preprocessing` config block can be splatted in directly;
            'SURE' is the only rule implemented and anything else is rejected.
        wavelet_level: Decomposition level (paper: 6).

    Returns:
        Preprocessed EEG array (n_channels, n_samples).
    """
    logger.info(f"DS-2 preprocessing: shape={eeg.shape}, fs={fs} Hz")

    # Step 1: Butterworth band-pass filter [Paper: 0.5-60 Hz]
    logger.info(f"Step 1: Butterworth bandpass [{bandpass_low}-{bandpass_high} Hz], "
                f"order={butterworth_order}")
    eeg = bandpass_filter(eeg, fs=fs, low=bandpass_low, high=bandpass_high,
                          order=butterworth_order, filter_type='butterworth')

    # Step 2: 50 Hz notch filter [Paper explicit]
    logger.info(f"Step 2: Notch filter at {notch_freq} Hz")
    eeg = notch_filter(eeg, fs=fs, notch_freq=notch_freq, quality=notch_quality)

    # Step 3: Coiflet-3 wavelet filtering for ocular noise [Paper explicit]
    logger.info(f"Step 3: Wavelet filtering (Coiflet-3, level {wavelet_level})")
    eeg = wavelet_filter_ocular(eeg, wavelet=wavelet, level=wavelet_level, fs=fs)

    # Step 4: SURE denoising up to level 6 [Paper explicit]
    if str(denoising_method).upper() != 'SURE':
        raise ValueError(
            f"Unsupported denoising_method={denoising_method!r}. "
            f"The paper specifies Stein's SURE, which is the only rule implemented."
        )
    logger.info(f"Step 4: SURE denoising (level {wavelet_level})")
    eeg = sure_wavelet_denoise(eeg, wavelet=wavelet, level=wavelet_level)

    # Final sanity: replace any residual NaN/Inf with 0
    # (NaN can arise from SURE thresholding zero-valued coefficients)
    nan_count = np.sum(~np.isfinite(eeg))
    if nan_count > 0:
        logger.warning(
            f"Found {nan_count} NaN/Inf values after preprocessing. "
            f"Replacing with 0. [Implementation note: pywt soft threshold "
            f"can produce NaN when coefficient magnitude = 0]"
        )
        eeg = np.where(np.isfinite(eeg), eeg, 0.0)

    logger.info(f"DS-2 preprocessing complete: shape={eeg.shape}")
    return eeg


# ─────────────────────────────────────────────────────────────────────────────
# Individual Filter Functions
# ─────────────────────────────────────────────────────────────────────────────

def bandpass_filter(eeg: np.ndarray,
                    fs: float,
                    low: float,
                    high: float,
                    order: int = 4,
                    filter_type: str = 'butterworth') -> np.ndarray:
    """Apply band-pass filter to multichannel EEG.

    DS-1: Band-pass 1-30 Hz [Paper explicit]
    DS-2: Band-pass 0.5-60 Hz, Butterworth [Paper explicit]

    Args:
        eeg: EEG array (n_channels, n_samples).
        fs: Sampling frequency.
        low: Lower cutoff frequency (Hz).
        high: Upper cutoff frequency (Hz).
        order: Filter order [not specified in paper for DS-1].
        filter_type: 'butterworth' (for DS-2) or 'iir' (general).

    Returns:
        Filtered EEG (n_channels, n_samples).
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = fs / 2.0

    # Validate cutoffs
    if low >= nyquist or high >= nyquist:
        logger.warning(f"Cutoff frequency exceeds Nyquist ({nyquist} Hz). Clipping.")
        high = min(high, nyquist - 0.5)
        low = max(low, 0.1)

    sos = butter(order, [low / nyquist, high / nyquist], btype='band', output='sos')

    # Apply filter to each channel
    eeg_filtered = np.zeros_like(eeg)
    for ch in range(eeg.shape[0]):
        eeg_filtered[ch] = sosfiltfilt(sos, eeg[ch])

    return eeg_filtered


def notch_filter(eeg: np.ndarray,
                 fs: float,
                 notch_freq: float = 50.0,
                 quality: float = 30.0) -> np.ndarray:
    """Apply notch filter to remove power line interference.

    DS-2: 50 Hz notch [Paper explicit]
    DS-1: CleanLine at 50/60 Hz [approximated here with IIR notch]

    [Paper does not specify Q factor; implementation choice: Q=30]

    Args:
        eeg: EEG array (n_channels, n_samples).
        fs: Sampling frequency.
        notch_freq: Center frequency to remove (paper DS-2: 50 Hz).
        quality: Q factor of notch filter.

    Returns:
        Filtered EEG (n_channels, n_samples).
    """
    from scipy.signal import iirnotch, sosfilt
    from scipy.signal import zpk2sos

    nyquist = fs / 2.0
    if notch_freq >= nyquist:
        logger.warning(f"Notch freq {notch_freq} Hz >= Nyquist {nyquist} Hz. Skipping.")
        return eeg

    b, a = iirnotch(notch_freq / nyquist, quality)

    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    from scipy.signal import filtfilt
    eeg_filtered = np.zeros_like(eeg)
    for ch in range(eeg.shape[0]):
        eeg_filtered[ch] = filtfilt(b, a, eeg[ch])

    return eeg_filtered


def wavelet_filter_ocular(eeg: np.ndarray,
                           wavelet: str = 'coif3',
                           level: int = 6,
                           fs: float = 128.0) -> np.ndarray:
    """Wavelet-based ocular noise removal using Coiflet-3.

    DS-2: "Coiflet-3 wavelet filtering for ocular noise removal" [Paper explicit]

    The paper uses the Coiflet-3 wavelet with SURE thresholding to remove
    ocular artifacts. This is applied BEFORE the main SURE denoising.

    Implementation: Decompose each channel with DWT, zero out low-frequency
    detail coefficients (where ocular artifacts predominate), then reconstruct.

    [Paper does not specify exactly which levels to zero; implementation choice:
     zero out highest-level approximation coefficients where eye artifacts live]

    Args:
        eeg: EEG array (n_channels, n_samples).
        wavelet: Wavelet name (paper: coif3).
        level: Decomposition level (paper: 6).
        fs: Sampling frequency.

    Returns:
        Cleaned EEG (n_channels, n_samples).
    """
    try:
        import pywt
    except ImportError:
        logger.error("PyWavelets required: pip install pywavelets")
        return eeg

    eeg_clean = np.zeros_like(eeg)

    for ch in range(eeg.shape[0]):
        # Decompose
        coeffs = pywt.wavedec(eeg[ch], wavelet, level=level)

        # Apply SURE threshold to each level (this is the ocular noise removal)
        # Approximation + all detail levels
        thresholded_coeffs = [coeffs[0]]  # Keep approximation (low freq)

        for i, detail in enumerate(coeffs[1:]):
            # Apply soft threshold using SURE
            threshold = _sure_threshold(detail)
            if threshold == 0.0:
                thresholded_coeffs.append(detail)
                continue
            thresholded = pywt.threshold(detail, threshold, mode='soft')
            # Guard against NaN/Inf from soft thresholding (occurs when |coef| == 0)
            thresholded = np.where(np.isfinite(thresholded), thresholded, 0.0)
            thresholded_coeffs.append(thresholded)

        # Reconstruct
        reconstructed = pywt.waverec(thresholded_coeffs, wavelet)[:eeg.shape[1]]
        # Final NaN guard
        reconstructed = np.where(np.isfinite(reconstructed), reconstructed, 0.0)
        eeg_clean[ch] = reconstructed

    return eeg_clean


def sure_wavelet_denoise(eeg: np.ndarray,
                          wavelet: str = 'coif3',
                          level: int = 6) -> np.ndarray:
    """Apply Stein's Unbiased Risk Estimation (SURE) wavelet denoising.

    DS-2: "Stein's adaptive unbiased risk estimation technique for denoising
           by removing unwanted signal coefficients up to sixth level" [Paper explicit]

    SURE selects the optimal threshold λ* that minimizes an unbiased estimate
    of the mean squared error.

    For each detail level d and coefficient set c:
    SURE(λ) = n - 2·|{i: |c_i| ≤ λ}| + Σ min(|c_i|, λ)²
    λ* = argmin SURE(λ)

    [Paper cites SURE; exact implementation follows standard wavelet denoising]

    Args:
        eeg: EEG array (n_channels, n_samples).
        wavelet: Wavelet name (paper: coif3).
        level: Decomposition levels (paper: 6).

    Returns:
        Denoised EEG (n_channels, n_samples).
    """
    try:
        import pywt
    except ImportError:
        logger.error("PyWavelets required: pip install pywavelets")
        return eeg

    eeg_denoised = np.zeros_like(eeg)

    for ch in range(eeg.shape[0]):
        coeffs = pywt.wavedec(eeg[ch], wavelet, level=level)

        # Apply SURE threshold to detail coefficients at each level
        denoised_coeffs = [coeffs[0]]  # approximation - keep as is
        for detail in coeffs[1:]:
            threshold = _sure_threshold(detail)
            if threshold == 0.0:
                denoised_coeffs.append(detail)
                continue
            denoised_detail = pywt.threshold(detail, threshold, mode='soft')
            # Guard against NaN/Inf (pywt soft threshold divides by magnitude)
            denoised_detail = np.where(np.isfinite(denoised_detail), denoised_detail, 0.0)
            denoised_coeffs.append(denoised_detail)

        reconstructed = pywt.waverec(denoised_coeffs, wavelet)
        reconstructed = np.where(np.isfinite(reconstructed), reconstructed, 0.0)
        eeg_denoised[ch] = reconstructed[:eeg.shape[1]]

    return eeg_denoised


def _sure_threshold(coefficients: np.ndarray) -> float:
    """Compute SURE (Stein's Unbiased Risk Estimation) threshold.

    Minimizes unbiased estimate of MSE:
    SURE(λ) = n - 2·#{|c_i| ≤ λ} + Σ min(|c_i|, λ)²

    The optimal threshold is found by evaluating SURE at each |c_i| value.

    Reference: Donoho & Johnstone (1995), "Adapting to unknown smoothness
    via wavelet shrinkage", JASA.

    [Paper references SURE; standard implementation used]

    Args:
        coefficients: 1D array of wavelet detail coefficients.

    Returns:
        Optimal SURE threshold value.
    """
    n = len(coefficients)
    if n == 0:
        return 0.0

    # Estimate noise sigma from median absolute deviation
    sigma = np.median(np.abs(coefficients)) / 0.6745

    if sigma < 1e-10:
        return 0.0

    # Normalize
    abs_coefs = np.sort(np.abs(coefficients / sigma))

    # Compute SURE for each possible threshold
    sure_values = np.zeros(n)
    for i, threshold in enumerate(abs_coefs):
        c2 = np.minimum(abs_coefs, threshold) ** 2
        n_below = np.sum(abs_coefs <= threshold)
        sure_values[i] = n - 2 * n_below + np.sum(c2)

    # Return optimal threshold (rescaled by sigma)
    optimal_idx = np.argmin(sure_values)
    return float(abs_coefs[optimal_idx] * sigma)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing Pipeline Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_ds1(eeg: np.ndarray,
                   fs: float = 128.0,
                   mode: str = 'B',
                   config: Optional[dict] = None) -> np.ndarray:
    """Preprocess DS-1 EEG data.

    Args:
        eeg: Raw or preprocessed EEG (n_channels, n_samples).
        fs: Sampling frequency (128 Hz).
        mode: 'A' = full pipeline approximation, 'B' = pass-through.
        config: Configuration dictionary (optional).

    Returns:
        Preprocessed EEG.
    """
    if mode == 'B':
        # MODE B: Use dataset's provided preprocessed data
        # No additional processing needed
        logger.info(
            "DS-1 MODE B: Using dataset-provided preprocessed EEG. "
            "No additional preprocessing applied."
        )
        return eeg
    elif mode == 'A':
        cfg = config or {}
        return preprocess_ds1_mode_a(
            eeg=eeg,
            fs=fs,
            bandpass_low=cfg.get('bandpass_low', 1.0),
            bandpass_high=cfg.get('bandpass_high', 30.0),
            line_noise_freqs=cfg.get('line_noise_freq', [50.0, 60.0]),
            n_ics_to_remove=cfg.get('n_artifact_ics_to_remove', 2)
        )
    else:
        raise ValueError(f"Unknown preprocessing mode: {mode}. Use 'A' or 'B'.")
