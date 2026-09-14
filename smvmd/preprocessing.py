"""
EEG Signal Preprocessing Module
===============================
Standard pediatric EEG preprocessing pipelines including:
- Zero-phase Butterworth bandpass filtering (0.5 - 45 Hz)
- Powerline notch filtering (50 Hz / 60 Hz)
- Baseline drift correction and robust artifact normalization
- Multi-channel epoch segmentation
"""

from typing import Tuple, List, Optional
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, detrend


def bandpass_filter(
    data: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = 128.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter across EEG channels.

    Parameters
    ----------
    data : np.ndarray, shape (C, T) or (T,)
        EEG signal array.
    lowcut : float
        Lower cutoff frequency in Hz.
    highcut : float
        Upper cutoff frequency in Hz.
    fs : float
        Sampling rate in Hz.
    order : int
        Filter order.

    Returns
    -------
    filtered : np.ndarray of same shape
    """
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.999, highcut / nyq)
    b, a = butter(order, [low, high], btype="band")

    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.array([filtfilt(b, a, data[c, :]) for c in range(data.shape[0])])


def notch_filter(
    data: np.ndarray,
    freq: float = 50.0,
    q: float = 30.0,
    fs: float = 128.0,
) -> np.ndarray:
    """
    Apply IIR notch filter to suppress powerline interference.
    """
    nyq = 0.5 * fs
    if freq >= nyq:
        return data  # Notch frequency above Nyquist
    w0 = freq / nyq
    b, a = iirnotch(w0, q)

    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.array([filtfilt(b, a, data[c, :]) for c in range(data.shape[0])])


def normalize_eeg(
    data: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """
    Normalize EEG channels.

    Parameters
    ----------
    data : np.ndarray, shape (C, T) or (T,)
    method : str, 'zscore' or 'robust' or 'minmax'

    Returns
    -------
    normalized : np.ndarray
    """
    if data.ndim == 1:
        data_2d = data[np.newaxis, :]
        was_1d = True
    else:
        data_2d = data
        was_1d = False

    C, T = data_2d.shape
    out = np.zeros_like(data_2d)

    for c in range(C):
        ch = data_2d[c, :]
        if method == "zscore":
            std = np.std(ch) + 1e-8
            out[c, :] = (ch - np.mean(ch)) / std
        elif method == "robust":
            med = np.median(ch)
            mad = np.median(np.abs(ch - med)) + 1e-8
            out[c, :] = (ch - med) / (1.4826 * mad)
        elif method == "minmax":
            ptp = np.ptp(ch) + 1e-8
            out[c, :] = (ch - np.min(ch)) / ptp
        else:
            out[c, :] = ch

    return out[0, :] if was_1d else out


def preprocess_eeg(
    data: np.ndarray,
    fs: float = 128.0,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    notch_freq: Optional[float] = 50.0,
    normalize: bool = True,
    norm_method: str = "zscore",
) -> np.ndarray:
    """
    Full standard EEG preprocessing pipeline:
    1. Detrending (baseline drift removal)
    2. Bandpass filtering (0.5 - 45 Hz)
    3. Notch filtering (50/60 Hz)
    4. Channel normalization
    """
    # 1. Detrend
    if data.ndim == 1:
        cleaned = detrend(data, type="linear")
    else:
        cleaned = np.array([detrend(data[c, :], type="linear") for c in range(data.shape[0])])

    # 2. Bandpass
    cleaned = bandpass_filter(cleaned, lowcut=lowcut, highcut=highcut, fs=fs)

    # 3. Notch
    if notch_freq is not None and notch_freq < (fs / 2.0):
        cleaned = notch_filter(cleaned, freq=notch_freq, fs=fs)

    # 4. Normalization
    if normalize:
        cleaned = normalize_eeg(cleaned, method=norm_method)

    return cleaned


def create_epochs(
    data: np.ndarray,
    epoch_len_sec: float = 2.0,
    overlap_ratio: float = 0.5,
    fs: float = 128.0,
) -> np.ndarray:
    """
    Segment continuous multi-channel EEG into overlapping epochs.

    Parameters
    ----------
    data : np.ndarray, shape (C, T) or (T,)
    epoch_len_sec : float
        Length of each epoch in seconds (e.g. 2.0s = 256 samples at 128 Hz).
    overlap_ratio : float
        Ratio of overlap between consecutive windows (e.g. 0.5 = 50% overlap).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    epochs : np.ndarray of shape (num_epochs, C, epoch_samples) or (num_epochs, epoch_samples)
    """
    was_1d = (data.ndim == 1)
    if was_1d:
        data_2d = data[np.newaxis, :]
    else:
        data_2d = data

    C, T = data_2d.shape
    epoch_samples = int(epoch_len_sec * fs)
    step_samples = int(epoch_samples * (1.0 - overlap_ratio))
    step_samples = max(1, step_samples)

    num_epochs = (T - epoch_samples) // step_samples + 1
    if num_epochs <= 0:
        # Signal is shorter than epoch length, pad or return single epoch
        padded = np.zeros((C, epoch_samples))
        padded[:, :T] = data_2d
        epochs = padded[np.newaxis, :, :]
    else:
        epochs = np.zeros((num_epochs, C, epoch_samples))
        for i in range(num_epochs):
            start = i * step_samples
            end = start + epoch_samples
            epochs[i, :, :] = data_2d[:, start:end]

    if was_1d:
        return epochs[:, 0, :]
    return epochs
