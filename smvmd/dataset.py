"""
EEG Dataset Module for Pediatric IDD (DS-1) and ADHD (DS-2)
============================================================
Accurately reflects the dataset specifications from Chandela et al. (2025):
- DS-1: 14-Channel EMOTIV EPOC+ montage (IDD vs TDC, Rest & Music states, 5s windows, 50% overlap).
- DS-2: 19-Channel 10-20 montage (ADHD vs NC, 61 ADHD + 60 NC, 5s windows, 50% overlap).
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import os
import numpy as np
from scipy.io import loadmat


# 14-channel EMOTIV EPOC+ montage used in DS-1 (Fig. 2a)
DS1_14_CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"
]

# 19-channel 10-20 montage used in DS-2 (Fig. 2b)
DS2_19_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2"
]


class PediatricEEGDataset:
    """
    Container for pediatric EEG dataset (DS-1 and DS-2).
    """

    def __init__(
        self,
        signals: List[np.ndarray],  # list of (C, T) arrays per epoch
        labels: List[int],          # 0: Control, 1: ADHD / IDD
        subject_ids: Optional[List[str]] = None,
        fs: float = 128.0,
        channel_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        state: Optional[str] = None,  # 'rest', 'music', 'combined' for DS-1
    ):
        self.signals = signals
        self.labels = np.array(labels, dtype=int)
        self.fs = fs
        self.channel_names = channel_names or (DS1_14_CHANNELS if signals and signals[0].shape[0] == 14 else DS2_19_CHANNELS)
        self.class_names = class_names or ["Control", "Disorder"]
        self.state = state
        if subject_ids is None:
            self.subject_ids = [f"sub_{i:03d}" for i in range(len(signals))]
        else:
            self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        return self.signals[idx], self.labels[idx], self.subject_ids[idx]

    def filter_classes(self, target_labels: List[int]) -> "PediatricEEGDataset":
        mask = [lbl in target_labels for lbl in self.labels]
        new_signals = [self.signals[i] for i, m in enumerate(mask) if m]
        new_labels = [self.labels[i] for i, m in enumerate(mask) if m]
        new_sub_ids = [self.subject_ids[i] for i, m in enumerate(mask) if m]
        new_class_names = [self.class_names[lbl] for lbl in target_labels]

        label_map = {old: new for new, old in enumerate(target_labels)}
        remapped_labels = [label_map[l] for l in new_labels]

        return PediatricEEGDataset(
            signals=new_signals,
            labels=remapped_labels,
            subject_ids=new_sub_ids,
            fs=self.fs,
            channel_names=self.channel_names,
            class_names=new_class_names,
            state=self.state,
        )


def generate_synthetic_eeg_subject(
    label: int,  # 0: Control, 1: ADHD, 2: IDD
    duration_sec: float = 10.0,
    fs: float = 128.0,
    n_channels: int = 19,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Generate realistic pediatric EEG based on DS-1 / DS-2 electrophysiological profiles.
    """
    if rng is None:
        rng = np.random.RandomState()

    T = int(duration_sec * fs)
    t = np.linspace(0, duration_sec, T, endpoint=False)
    signal = np.zeros((n_channels, T))

    # 1. Scale-free 1/f neural noise
    white = rng.randn(n_channels, T)
    fft_white = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(T, 1.0 / fs)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    pink_filter = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(fft_white * pink_filter[np.newaxis, :], n=T, axis=1)
    signal += 0.8 * pink

    for c in range(n_channels):
        pj = rng.uniform(0, 2 * np.pi)

        if label == 0:  # Typical Control
            # Healthy Alpha (9.5-11.5 Hz) & Beta (16-22 Hz)
            a_amp, a_f = rng.uniform(1.2, 1.8), rng.uniform(9.5, 11.0)
            b_amp, b_f = rng.uniform(0.6, 1.0), rng.uniform(16.0, 20.0)
            th_amp, th_f = rng.uniform(0.3, 0.6), rng.uniform(5.0, 6.5)
            d_amp, d_f = rng.uniform(0.3, 0.6), rng.uniform(1.5, 2.5)

        elif label == 1:  # ADHD (Elevated frontal theta, high TBR)
            is_frontal = c in (0, 1, 2, 3, 4, 5, 6)
            mult = 2.4 if is_frontal else 1.6
            th_amp, th_f = rng.uniform(1.6, 2.4) * mult, rng.uniform(4.5, 6.5)
            b_amp, b_f = rng.uniform(0.2, 0.45), rng.uniform(15.0, 19.0)
            a_amp, a_f = rng.uniform(0.5, 0.9), rng.uniform(9.0, 10.5)
            d_amp, d_f = rng.uniform(0.4, 0.8), rng.uniform(1.5, 3.0)

        elif label == 2:  # IDD (Diffuse slowing, low alpha frequency)
            d_amp, d_f = rng.uniform(1.8, 2.8), rng.uniform(1.0, 2.8)
            th_amp, th_f = rng.uniform(1.5, 2.2), rng.uniform(4.0, 6.0)
            a_amp, a_f = rng.uniform(0.2, 0.5), rng.uniform(7.0, 8.5)
            b_amp, b_f = rng.uniform(0.1, 0.3), rng.uniform(14.0, 18.0)

        mod_a = 1.0 + 0.3 * np.sin(2 * np.pi * 0.2 * t + pj)
        mod_th = 1.0 + 0.4 * np.sin(2 * np.pi * 0.15 * t + pj + 1.0)

        sig_a = a_amp * mod_a * np.sin(2 * np.pi * a_f * t + pj)
        sig_b = b_amp * np.sin(2 * np.pi * b_f * t + pj * 1.5)
        sig_th = th_amp * mod_th * np.sin(2 * np.pi * th_f * t + pj * 0.7)
        sig_d = d_amp * np.sin(2 * np.pi * d_f * t + pj * 0.3)

        signal[c, :] += (sig_a + sig_b + sig_th + sig_d)

    # Spatial scalp conduction
    coupling = np.eye(n_channels)
    for i in range(n_channels):
        for j in range(n_channels):
            if i != j:
                coupling[i, j] = 0.25 * np.exp(-abs(i - j) / 3.0)
    coupling /= np.sum(coupling, axis=1, keepdims=True)
    signal = np.dot(coupling, signal) + 0.1 * rng.randn(n_channels, T)

    return signal


def generate_synthetic_eeg_cohort(
    n_controls: int = 30,
    n_adhd: int = 30,
    n_idd: int = 0,
    duration_sec: float = 10.0,
    epoch_len_sec: float = 5.0,  # 5s window from Section II-A3
    overlap_ratio: float = 0.5,  # 50% overlap from Section II-A3
    fs: float = 128.0,
    n_channels: int = 19,
    random_state: int = 42,
) -> PediatricEEGDataset:
    """
    Generate pediatric cohort segmented into 5-second epochs with 50% overlap
    as specified in Section II-A3 of Chandela et al. (2025).
    """
    from .preprocessing import create_epochs

    rng = np.random.RandomState(random_state)
    all_epochs: List[np.ndarray] = []
    all_labels: List[int] = []
    all_sub_ids: List[str] = []

    cohort = [
        (0, n_controls, "NC"),
        (1, n_adhd, "ADHD"),
        (2, n_idd, "IDD"),
    ]

    for label, count, prefix in cohort:
        if count == 0:
            continue
        for i in range(count):
            sub_id = f"{prefix}_{i+1:03d}"
            raw = generate_synthetic_eeg_subject(
                label=label,
                duration_sec=duration_sec,
                fs=fs,
                n_channels=n_channels,
                rng=rng,
            )
            epochs = create_epochs(
                raw,
                epoch_len_sec=epoch_len_sec,
                overlap_ratio=overlap_ratio,
                fs=fs,
            )
            for ep in epochs:
                all_epochs.append(ep)
                all_labels.append(label)
                all_sub_ids.append(sub_id)

    ch_names = DS1_14_CHANNELS if n_channels == 14 else DS2_19_CHANNELS[:n_channels]
    return PediatricEEGDataset(
        signals=all_epochs,
        labels=all_labels,
        subject_ids=all_sub_ids,
        fs=fs,
        channel_names=ch_names,
        class_names=["Control", "ADHD", "IDD"],
    )


def load_eeg_file(
    file_path: str,
    fs: float = 128.0,
    label: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[int]]:
    """Load EEG from .mat (e.g. IEEE Dataport DS-2 or Mendeley DS-1), .npy, .npz, .csv."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".npy":
        data = np.load(file_path)
    elif ext == ".npz":
        npz = np.load(file_path)
        data = npz["data"] if "data" in npz else npz[list(npz.keys())[0]]
        if "label" in npz:
            label = int(npz["label"])
    elif ext == ".mat":
        mat = loadmat(file_path)
        candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
        preferred = ["data", "eeg", "signal", "x", "val"]
        chosen_key = candidate_keys[0]
        for pref in preferred:
            for k in candidate_keys:
                if pref in k.lower():
                    chosen_key = k
                    break
        data = mat[chosen_key]
    elif ext == ".csv":
        data = np.loadtxt(file_path, delimiter=",")
    else:
        raise ValueError(f"Unsupported format: {ext}")

    if data.ndim == 1:
        data = data[np.newaxis, :]
    elif data.ndim == 2 and data.shape[0] > data.shape[1]:
        data = data.T

    return data, label
