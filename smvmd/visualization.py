"""
Visualization and Diagnostic Plotting Module
============================================
Publication-grade visualization utilities for EEG analysis,
SMVMD decomposition modes, power spectra, and clinical classification results.
"""

from typing import List, Dict, Tuple, Optional, Any
import os
import numpy as np


def plot_eeg_channels(
    signal: np.ndarray,
    fs: float = 128.0,
    channel_names: Optional[List[str]] = None,
    title: str = "Multi-Channel Pediatric EEG",
    save_path: Optional[str] = None,
):
    """
    Plot multi-channel EEG signals as a stacked montage trace.
    """
    import matplotlib.pyplot as plt

    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    C, T = signal.shape
    time_axis = np.arange(T) / fs
    ch_names = channel_names or [f"Ch {i+1}" for i in range(C)]

    fig, ax = plt.subplots(figsize=(12, max(4, C * 0.45)), dpi=150)
    spacing = 3.0 * np.std(signal) if np.std(signal) > 0 else 1.0

    for c in range(C):
        offset = (C - 1 - c) * spacing
        ax.plot(time_axis, signal[c, :] + offset, color="#1f77b4", lw=1.0)

    ax.set_yticks([(C - 1 - c) * spacing for c in range(C)])
    ax.set_yticklabels(ch_names, fontsize=9, fontweight="bold")
    ax.set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim([0, time_axis[-1]])
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_raw_vs_modes(
    raw_signal: np.ndarray,
    modes: np.ndarray,
    omega_hz: np.ndarray,
    fs: float = 128.0,
    channel_idx: int = 0,
    channel_name: str = "Fp1",
    title: str = "SMVMD Decomposition (Raw Signal & MVMFs)",
    save_path: Optional[str] = None,
):
    """
    Plot raw EEG channel and its decomposed Multivariate Variational Mode Functions (MVMFs).
    """
    import matplotlib.pyplot as plt

    if raw_signal.ndim == 1:
        raw_ch = raw_signal
    else:
        raw_ch = raw_signal[channel_idx, :]

    K = modes.shape[0]
    T = raw_ch.shape[0]
    time_axis = np.arange(T) / fs

    fig, axes = plt.subplots(K + 1, 1, figsize=(12, 2.0 * (K + 1)), sharex=True, dpi=150)
    if K == 0:
        axes = [axes]

    # Plot Raw Signal
    axes[0].plot(time_axis, raw_ch, color="#2b2b2b", lw=1.2)
    axes[0].set_ylabel(f"Raw ({channel_name})", fontweight="bold", fontsize=10)
    axes[0].set_title(title, fontweight="bold", fontsize=12, pad=10)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Plot Modes
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    for k in range(K):
        m_ch = modes[k] if modes[k].ndim == 1 else modes[k, channel_idx, :]
        cf = omega_hz[k] if k < len(omega_hz) else 0.0
        color = colors[k % len(colors)]

        axes[k + 1].plot(time_axis, m_ch, color=color, lw=1.1)
        axes[k + 1].set_ylabel(f"MVMF {k+1}\n(fc={cf:.1f}Hz)", fontweight="bold", fontsize=9)
        axes[k + 1].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (seconds)", fontweight="bold", fontsize=11)
    axes[-1].set_xlim([0, time_axis[-1]])
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_mode_spectra(
    modes: np.ndarray,
    omega_hz: np.ndarray,
    fs: float = 128.0,
    channel_idx: int = 0,
    title: str = "Power Spectra of Extracted MVMFs",
    save_path: Optional[str] = None,
):
    """
    Plot power spectral density of extracted modes showing band compactness.
    """
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    K = modes.shape[0]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for k in range(K):
        m_ch = modes[k] if modes[k].ndim == 1 else modes[k, channel_idx, :]
        f, pxx = welch(m_ch, fs=fs, nperseg=min(len(m_ch), 128))
        cf = omega_hz[k] if k < len(omega_hz) else 0.0
        color = colors[k % len(colors)]

        ax.plot(f, pxx, label=f"MVMF {k+1} (fc = {cf:.1f} Hz)", color=color, lw=1.8)
        ax.axvline(cf, color=color, linestyle=":", alpha=0.7)

    ax.set_xlabel("Frequency (Hz)", fontweight="bold", fontsize=11)
    ax.set_ylabel("Power Spectral Density (V²/Hz)", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set_xlim([0, fs / 2.0])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, fontsize=9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_energy_distribution(
    modes: np.ndarray,
    omega_hz: np.ndarray,
    title: str = "Energy Distribution Across Decomposed Modes",
    save_path: Optional[str] = None,
):
    """
    Plot bar chart of mode relative energy contributions.
    """
    import matplotlib.pyplot as plt

    K = modes.shape[0]
    energies = [np.sum(modes[k] ** 2) for k in range(K)]
    total_e = sum(energies) + 1e-12
    rel_energies = [e / total_e * 100 for e in energies]

    labels = [f"MVMF {k+1}\n({omega_hz[k]:.1f} Hz)" for k in range(K)]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(labels, rel_energies, color=colors[:K], edgecolor="black", alpha=0.85, width=0.55)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    ax.set_ylabel("Relative Energy (%)", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.set_ylim([0, max(rel_energies) * 1.25])
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
):
    """
    Plot formatted confusion matrix with cell annotations.
    """
    import matplotlib.pyplot as plt

    n_classes = cm.shape[0]
    names = class_names or [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.8)

    for i in range(n_classes):
        for j in range(n_classes):
            val = cm[i, j]
            row_sum = np.sum(cm[i, :])
            pct = (val / row_sum * 100) if row_sum > 0 else 0
            text_color = "white" if val > np.max(cm) / 2 else "black"
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color, fontweight="bold", fontsize=10,
            )

    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(names, fontweight="bold", fontsize=10)
    ax.set_yticklabels(names, fontweight="bold", fontsize=10)
    ax.set_xlabel("Predicted Label", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_ylabel("True Label", fontweight="bold", fontsize=11, labelpad=10)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Receiver Operating Characteristic (ROC)",
    save_path: Optional[str] = None,
):
    """
    Plot multi-class or binary ROC curves with AUC annotations.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    classes = np.unique(y_true)
    n_classes = len(classes)
    names = class_names or [f"Class {i}" for i in range(n_classes)]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if n_classes == 2:
        prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
        fpr, tpr, _ = roc_curve(y_true, prob_pos)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[0], lw=2.0, label=f"ROC (AUC = {roc_auc:.4f})")
    else:
        y_bin = label_binarize(y_true, classes=classes)
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.0, label=f"{names[i]} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontweight="bold", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.legend(loc="lower right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    top_n: int = 15,
    title: str = "Top Discriminative Features",
    save_path: Optional[str] = None,
):
    """
    Plot top discriminative features based on importance scores.
    """
    import matplotlib.pyplot as plt

    sorted_indices = np.argsort(importance_scores)[::-1][:top_n]
    top_names = [feature_names[i] for i in sorted_indices][::-1]
    top_scores = importance_scores[sorted_indices][::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)), dpi=150)
    ax.barh(top_names, top_scores, color="#2b5c8f", edgecolor="black", alpha=0.85, height=0.6)
    ax.set_xlabel("Relative Importance / Discriminability", fontweight="bold", fontsize=11)
    ax.set_title(title, fontweight="bold", fontsize=12, pad=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    return fig
