"""
utils.py - Utility functions for EEG-Based Neurodevelopmental Disorder Detection

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025
"""

import os
import yaml
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(name: str, log_file: Optional[str] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Configure logging for an experiment run.

    Handlers are attached to the ROOT logger, not just to `name`. Every module
    under src/ logs through its own `logging.getLogger(__name__)`, and those
    records propagate to root. Attaching handlers only to the experiment's own
    logger left root bare, so all of that output — per-fold cross-validation
    progress, SMVMD detail, stage timings, cache hits — was silently discarded.
    During the longest stages a run then looked like it had hung.

    Calling this more than once in a process replaces the handlers it installed
    previously rather than stacking duplicates.

    Args:
        name: Logger name for the experiment's own messages.
        log_file: Optional path; the whole run is also written there.
        level: Threshold for our own code.

    Returns:
        The named logger.
    """
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Drop handlers a previous setup_logger call installed, so repeated calls
    # (e.g. importing two experiment modules) do not duplicate every line.
    for h in list(root.handlers):
        if getattr(h, '_eeg_pipeline_handler', False):
            root.removeHandler(h)
            h.close()

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch._eeg_pipeline_handler = True
    root.addHandler(ch)

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(formatter)
        fh._eeg_pipeline_handler = True
        root.addHandler(fh)

    # Third-party libraries are chatty at DEBUG and drown out the pipeline.
    for noisy in ('matplotlib', 'PIL', 'fontTools'):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        Dictionary with all configuration parameters.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Checks
# ─────────────────────────────────────────────────────────────────────────────

def check_array_health(arr: np.ndarray, name: str = "array",
                       logger: Optional[logging.Logger] = None) -> bool:
    """Check for NaN and Inf values in an array.

    Args:
        arr: NumPy array to check.
        name: Name for logging purposes.
        logger: Logger instance.

    Returns:
        True if array is healthy, False otherwise.
    """
    log = logger or logging.getLogger(__name__)
    has_nan = np.any(np.isnan(arr))
    has_inf = np.any(np.isinf(arr))

    if has_nan:
        log.warning(f"[HEALTH CHECK FAILED] {name} contains NaN values: "
                    f"{np.sum(np.isnan(arr))} NaNs out of {arr.size}")
    if has_inf:
        log.warning(f"[HEALTH CHECK FAILED] {name} contains Inf values: "
                    f"{np.sum(np.isinf(arr))} Infs out of {arr.size}")

    healthy = not has_nan and not has_inf
    if healthy:
        log.debug(f"[HEALTH CHECK OK] {name}: shape={arr.shape}, "
                  f"min={arr.min():.4f}, max={arr.max():.4f}")
    return healthy


def verify_segment_count(actual: int, expected: int, dataset_name: str,
                         logger: Optional[logging.Logger] = None,
                         tolerance: int = 0) -> bool:
    """Verify that segment count matches paper-reported value.

    Args:
        actual: Actual number of segments produced.
        expected: Expected number from paper.
        dataset_name: Name of the dataset for logging.
        logger: Logger instance.
        tolerance: Allowable difference (default 0 = exact match required).

    Returns:
        True if within tolerance, False otherwise.
    """
    log = logger or logging.getLogger(__name__)
    diff = abs(actual - expected)

    if diff <= tolerance:
        log.info(f"[SEGMENT COUNT OK] {dataset_name}: {actual} segments "
                 f"(expected {expected})")
        return True
    else:
        log.warning(f"[SEGMENT COUNT MISMATCH] {dataset_name}: "
                    f"actual={actual}, expected={expected}, diff={diff}")
        log.warning(f"Investigate: different from paper's reported count.")
        return False


def verify_feature_dimensions(feature_matrix: np.ndarray,
                               expected_n_segments: int,
                               expected_n_features: int,
                               dataset_name: str,
                               logger: Optional[logging.Logger] = None) -> bool:
    """Verify that the integrated feature matrix has correct dimensions.

    For DS-2: expected (6588, 171)
    For DS-1: expected (1288, 126)

    Args:
        feature_matrix: The integrated feature matrix (n_segments, n_features).
        expected_n_segments: Expected first dimension.
        expected_n_features: Expected second dimension.
        dataset_name: For logging.
        logger: Logger instance.

    Returns:
        True if dimensions match exactly.
    """
    log = logger or logging.getLogger(__name__)
    actual_shape = feature_matrix.shape

    seg_ok = actual_shape[0] == expected_n_segments
    feat_ok = actual_shape[1] == expected_n_features

    log.info(f"[FEATURE DIM CHECK] {dataset_name}: "
             f"actual={actual_shape}, expected=({expected_n_segments}, {expected_n_features})")

    if seg_ok and feat_ok:
        log.info(f"[FEATURE DIM OK] {dataset_name}")
        return True
    else:
        if not seg_ok:
            log.warning(f"Segment count mismatch: {actual_shape[0]} vs {expected_n_segments}")
        if not feat_ok:
            log.warning(f"Feature count mismatch: {actual_shape[1]} vs {expected_n_features}")
        return False


def verify_smvmd_reconstruction(original: np.ndarray,
                                mimfs: List[np.ndarray],
                                residual: Optional[np.ndarray] = None,
                                tolerance: float = 1e-6,
                                logger: Optional[logging.Logger] = None) -> Tuple[bool, float]:
    """Verify that SMVMD reconstruction ≈ original signal.

    Checks: |original - sum(MIMFs)| / |original| < tolerance

    Args:
        original: Original signal, shape (C, T).
        mimfs: List of K MIMFs, each shape (C, T).
        residual: Optional residual signal, shape (C, T).
        tolerance: Relative error threshold.
        logger: Logger instance.

    Returns:
        Tuple of (passes_check, relative_error).
    """
    log = logger or logging.getLogger(__name__)

    reconstruction = np.sum(np.array(mimfs), axis=0)
    if residual is not None:
        reconstruction = reconstruction + residual

    error = np.linalg.norm(original - reconstruction) / (np.linalg.norm(original) + 1e-12)

    passes = error < tolerance
    level = log.info if passes else log.warning
    level(f"[SMVMD RECONSTRUCTION] Relative error: {error:.2e} "
          f"(tolerance: {tolerance:.2e}) -> {'PASS' if passes else 'FAIL'}")

    return passes, float(error)


def verify_sampling_rate(fs: float, expected_fs: float = 128.0,
                         logger: Optional[logging.Logger] = None) -> bool:
    """Verify sampling rate matches paper specification (128 Hz).

    Args:
        fs: Actual sampling frequency.
        expected_fs: Expected frequency from paper (default 128 Hz).
        logger: Logger instance.

    Returns:
        True if match.
    """
    log = logger or logging.getLogger(__name__)
    ok = abs(fs - expected_fs) < 0.1
    level = log.info if ok else log.error
    level(f"[SAMPLING RATE] actual={fs} Hz, expected={expected_fs} Hz -> "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def verify_segment_shape(segment: np.ndarray,
                         n_channels: int,
                         n_samples: int = 640,
                         logger: Optional[logging.Logger] = None) -> bool:
    """Verify a segment has correct shape (n_channels × 640).

    Paper specifies: 5s × 128 Hz = 640 samples.

    Args:
        segment: Array of shape (n_channels, n_samples).
        n_channels: Expected channel count (14 for DS-1, 19 for DS-2).
        n_samples: Expected sample count (640 per paper).
        logger: Logger instance.

    Returns:
        True if shape matches.
    """
    log = logger or logging.getLogger(__name__)
    expected = (n_channels, n_samples)
    ok = segment.shape == expected
    level = log.info if ok else log.error
    level(f"[SEGMENT SHAPE] actual={segment.shape}, expected={expected} -> "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_raw_eeg(eeg: np.ndarray,
                channel_names: List[str],
                fs: float = 128.0,
                title: str = "Raw EEG",
                save_path: Optional[str] = None,
                figsize: Tuple[int, int] = (16, 10)) -> None:
    """Plot multichannel EEG signal.

    Args:
        eeg: EEG array of shape (n_channels, n_samples).
        channel_names: List of channel name strings.
        fs: Sampling frequency in Hz.
        title: Figure title.
        save_path: If provided, save figure to this path.
        figsize: Figure size tuple.
    """
    n_channels, n_samples = eeg.shape
    time = np.arange(n_samples) / fs

    fig, ax = plt.subplots(figsize=figsize)

    # Offset channels for visibility
    offsets = np.arange(n_channels) * 2 * np.max(np.abs(eeg))
    for i, (ch_data, offset) in enumerate(zip(eeg, offsets)):
        # Normalize per channel
        norm = np.max(np.abs(ch_data)) + 1e-10
        ax.plot(time, ch_data / norm + i * 2, linewidth=0.5, color='steelblue')

    ax.set_yticks(np.arange(n_channels) * 2)
    ax.set_yticklabels(channel_names, fontsize=8)
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_smvmd_decomposition(original: np.ndarray,
                              mimfs: List[np.ndarray],
                              center_freqs: List[float],
                              channel_idx: int = 0,
                              fs: float = 128.0,
                              channel_name: str = "Ch0",
                              save_path: Optional[str] = None) -> None:
    """Plot SMVMD decomposition for one channel.

    Plots:
    1. Original signal
    2. Each MIMF with its center frequency label
    3. Reconstruction (sum of MIMFs) vs original
    4. Residual

    Args:
        original: Original signal (C, T).
        mimfs: List of K MIMFs, each (C, T).
        center_freqs: List of K center frequencies.
        channel_idx: Which channel to plot.
        fs: Sampling frequency.
        channel_name: Channel name for title.
        save_path: Save path for the figure.
    """
    K = len(mimfs)
    n_plots = K + 3  # original + K modes + reconstruction + residual
    time = np.arange(original.shape[1]) / fs

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 2 * n_plots), sharex=True)

    # Original
    axes[0].plot(time, original[channel_idx], color='black', linewidth=0.8)
    axes[0].set_ylabel("Original", fontsize=8)
    axes[0].set_title(f"SMVMD Decomposition - {channel_name}", fontsize=12)

    # MIMFs
    reconstruction = np.zeros_like(original[channel_idx])
    for k, (mimf, wk) in enumerate(zip(mimfs, center_freqs)):
        axes[k + 1].plot(time, mimf[channel_idx], linewidth=0.8,
                         color=f'C{k % 10}')
        axes[k + 1].set_ylabel(f"MIMF{k+1}\n(ω={wk*fs/2:.2f} Hz)", fontsize=7)
        reconstruction += mimf[channel_idx]

    # Reconstruction
    axes[K + 1].plot(time, original[channel_idx], color='black',
                     linewidth=1, label='Original', alpha=0.7)
    axes[K + 1].plot(time, reconstruction, color='red',
                     linewidth=0.8, linestyle='--', label='Reconstruction')
    axes[K + 1].set_ylabel("Reconstruction", fontsize=8)
    axes[K + 1].legend(fontsize=7)

    # Residual
    residual = original[channel_idx] - reconstruction
    axes[K + 2].plot(time, residual, color='gray', linewidth=0.6)
    axes[K + 2].set_ylabel("Residual", fontsize=8)
    axes[K + 2].set_xlabel("Time (s)", fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: np.ndarray,
                          class_names: List[str],
                          title: str = "Confusion Matrix",
                          save_path: Optional[str] = None) -> None:
    """Plot confusion matrix with annotations.

    Args:
        cm: Confusion matrix array, shape (n_classes, n_classes).
        class_names: List of class label names.
        title: Figure title.
        save_path: Save path for the figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           ylabel='True Label',
           xlabel='Predicted Label',
           title=title)

    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_features(n_features_list: List[int],
                               accuracies: List[float],
                               optimal_n: int,
                               dataset_name: str,
                               save_path: Optional[str] = None) -> None:
    """Plot accuracy vs number of selected mRMR features.

    Reproduces Fig. 5 from the paper.

    Args:
        n_features_list: List of feature counts evaluated.
        accuracies: Corresponding accuracy values.
        optimal_n: The selected optimal feature count.
        dataset_name: Dataset label for the title.
        save_path: Save path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(n_features_list, [a * 100 for a in accuracies],
            color='steelblue', linewidth=2, marker='o', markersize=4)
    ax.axvline(x=optimal_n, color='red', linestyle='--',
               label=f'Selected: {optimal_n} features')
    ax.set_xlabel("Number of Features (mRMR Ranked)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(f"Accuracy vs. Number of Features – {dataset_name}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Metadata Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_segment_metadata(subject_id: str,
                            label: int,
                            dataset: str,
                            condition: str,
                            segment_idx: int,
                            start_sample: int,
                            end_sample: int,
                            fs: float = 128.0) -> Dict[str, Any]:
    """Build a metadata dictionary for a single EEG segment.

    CRITICAL: Never lose subject ID - this function ensures full traceability.

    Args:
        subject_id: Unique subject identifier string.
        label: Class label (e.g., 1=IDD/ADHD, 0=TDC/NC).
        dataset: Dataset name ("DS1" or "DS2").
        condition: Recording condition ("rest", "music", "visual_task").
        segment_idx: Index of this segment within subject's recording.
        start_sample: Start sample index in original recording.
        end_sample: End sample index (exclusive) in original recording.
        fs: Sampling frequency.

    Returns:
        Dictionary with all required metadata fields.
    """
    return {
        "subject_id": subject_id,
        "label": label,
        "dataset": dataset,
        "recording_condition": condition,
        "segment_index": segment_idx,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "start_time_sec": start_sample / fs,
        "end_time_sec": end_sample / fs,
    }


def save_metadata_csv(metadata_list: List[Dict[str, Any]],
                       save_path: str) -> None:
    """Save segment metadata to CSV.

    Args:
        metadata_list: List of metadata dictionaries.
        save_path: Output CSV file path.
    """
    df = pd.DataFrame(metadata_list)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


def load_metadata_csv(path: str) -> pd.DataFrame:
    """Load segment metadata from CSV.

    Args:
        path: CSV file path.

    Returns:
        DataFrame with segment metadata.
    """
    df = pd.read_csv(path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_random_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Note: Paper does not specify random seed.
    Implementation choice: seed = 42 (configurable via config.yaml).

    Args:
        seed: Random seed value.
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass  # PyTorch not required


# ─────────────────────────────────────────────────────────────────────────────
# Frequency Domain Helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_analytical_signal(x: np.ndarray) -> np.ndarray:
    """Compute analytical signal via Hilbert transform.

    Used internally by SMVMD.

    Args:
        x: Real-valued signal array.

    Returns:
        Complex analytical signal.
    """
    from scipy.signal import hilbert
    return hilbert(x)


def normalized_frequency_axis(n: int) -> np.ndarray:
    """Generate normalized frequency axis [0, 0.5] for n samples.

    Args:
        n: Number of samples.

    Returns:
        Frequency axis array (only positive frequencies, length n//2+1).
    """
    return np.linspace(0, 0.5, n // 2 + 1)


# ─────────────────────────────────────────────────────────────────────────────
# GDR and G-mean (DS-2 metrics)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gdr(cm: np.ndarray) -> float:
    """Compute Good Detection Rate (GDR) from confusion matrix.

    GDR is used by the paper (ref [46]) for imbalanced classification.
    GDR = geometric mean of per-class sensitivities.

    For 2-class problem:
    GDR = √(sensitivity_class0 × sensitivity_class1) × 100

    Note: paper reports GDR ≈ 99.239 for DS-2.

    Args:
        cm: Confusion matrix, shape (2, 2).
            cm[i,j] = count predicted as j when true label is i.

    Returns:
        GDR value (as percentage, 0-100).
    """
    # Sensitivity per class = TP / (TP + FN)
    sensitivities = cm.diagonal() / cm.sum(axis=1)
    gdr = np.sqrt(np.prod(sensitivities)) * 100
    return float(gdr)


def compute_gmean(cm: np.ndarray) -> float:
    """Compute G-mean from confusion matrix.

    G-mean = √(sensitivity × specificity)
    For binary case:
    - sensitivity = TP / (TP + FN) = recall for positive class
    - specificity = TN / (TN + FP) = recall for negative class

    Note: paper reports G-mean ≈ 0.9915 for DS-2.

    Args:
        cm: 2×2 confusion matrix.

    Returns:
        G-mean value (0 to 1).
    """
    # cm[0,0] = TN, cm[0,1] = FP, cm[1,0] = FN, cm[1,1] = TP
    sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0] + 1e-10)  # recall of positive
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-10)  # recall of negative
    gmean = np.sqrt(sensitivity * specificity)
    return float(gmean)
