"""
Ablation Study: Validating Key Novelties of Chandela et al. (2025)
==================================================================
Systematic ablation comparing:
1. SMVMD + Energy-Based Feature Integration (Proposed Method)
2. Fixed Multivariate VMD (MVMD, K=5) + Feature Extraction
3. Raw Multichannel EEG + Feature Extraction (No Decomposition)
4. Univariate VMD per channel (No Joint Multivariate Alignment)
5. SMVMD with Naive Zero-Padding Concatenation (No Energy Integration)
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smvmd.smvmd import smvmd_decompose, mvmd_decompose, vmd_decompose
from smvmd.preprocessing import preprocess_eeg
from smvmd.features import extract_9_features_1d
from smvmd.feature_integration import EnergyBasedFeatureIntegrator
from smvmd.models import evaluate_cross_validation
from smvmd.dataset import generate_synthetic_eeg_cohort


def run_ablation_study(
    n_samples_per_class: int = 15,
    fs: float = 128.0,
    output_dir: str = "results/ablation_study",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print(" ABLATION STUDY: SYSTEMATIC ARCHITECTURE COMPARISON")
    print("=" * 75)

    dataset = generate_synthetic_eeg_cohort(
        n_controls=n_samples_per_class,
        n_adhd=n_samples_per_class,
        n_idd=0,
        duration_sec=6.0,
        epoch_len_sec=5.0,
        overlap_ratio=0.5,
        fs=fs,
        n_channels=14,
        random_state=42,
    ).filter_classes([0, 1])

    y = np.array(dataset.labels)
    clean_epochs = [
        preprocess_eeg(sig, fs=fs, lowcut=0.5, highcut=45.0)
        for sig in dataset.signals
    ]

    ablation_results = {}
    integrator = EnergyBasedFeatureIntegrator()

    # 1. Proposed (SMVMD + EBFI)
    print("\n[1/5] Evaluating Variant 1: Proposed (SMVMD + EBFI)...")
    t0 = time.time()
    X_v1 = []
    for epoch in clean_epochs:
        modes, omega_hz, _ = smvmd_decompose(epoch, fs=fs, alpha=2000.0, max_modes=6)
        f_vec, _ = integrator.integrate_multichannel_modes(modes, fs=fs)
        X_v1.append(f_vec)
    time_v1 = time.time() - t0
    m_v1 = evaluate_cross_validation(np.array(X_v1), y, classifier_type="knn", preset="ds2_adhd", n_splits=5)
    ablation_results["1. Proposed (SMVMD + EBFI)"] = (m_v1, time_v1)
    print(f"      Accuracy: {m_v1['accuracy']*100:.2f}% +/- {m_v1['accuracy_std']*100:.2f}% | F1: {m_v1['f1_score']*100:.2f}% | Time: {time_v1:.1f}s")

    # 2. Fixed MVMD (K=5) + Feature Extraction
    print("\n[2/5] Evaluating Variant 2: Fixed MVMD (K=5)...")
    t0 = time.time()
    X_v2 = []
    for epoch in clean_epochs:
        modes, omega_hz, _ = mvmd_decompose(epoch, K=5, alpha=2000.0, fs=fs)
        f_vec, _ = integrator.integrate_multichannel_modes(modes, fs=fs)
        X_v2.append(f_vec)
    time_v2 = time.time() - t0
    m_v2 = evaluate_cross_validation(np.array(X_v2), y, classifier_type="knn", preset="ds2_adhd", n_splits=5)
    ablation_results["2. Fixed MVMD (K=5) + EBFI"] = (m_v2, time_v2)
    print(f"      Accuracy: {m_v2['accuracy']*100:.2f}% +/- {m_v2['accuracy_std']*100:.2f}% | F1: {m_v2['f1_score']*100:.2f}% | Time: {time_v2:.1f}s")

    # 3. Raw Multichannel EEG (No Decomposition)
    print("\n[3/5] Evaluating Variant 3: Raw Multichannel EEG (No Decomposition)...")
    t0 = time.time()
    X_v3 = []
    for epoch in clean_epochs:
        C = epoch.shape[0]
        f_raw = []
        for c in range(C):
            f_dict = extract_9_features_1d(epoch[c, :], fs=fs)
            f_raw.extend(list(f_dict.values()))
        X_v3.append(np.array(f_raw))
    time_v3 = time.time() - t0
    m_v3 = evaluate_cross_validation(np.array(X_v3), y, classifier_type="knn", preset="ds2_adhd", n_splits=5)
    ablation_results["3. Raw EEG (No Decomposition)"] = (m_v3, time_v3)
    print(f"      Accuracy: {m_v3['accuracy']*100:.2f}% +/- {m_v3['accuracy_std']*100:.2f}% | F1: {m_v3['f1_score']*100:.2f}% | Time: {time_v3:.1f}s")

    # 4. Univariate VMD per Channel
    print("\n[4/5] Evaluating Variant 4: Univariate VMD per Channel...")
    t0 = time.time()
    X_v4 = []
    for epoch in clean_epochs:
        ch_modes_list = []
        for c in range(min(4, epoch.shape[0])):
            m_c, _, _ = vmd_decompose(epoch[c, :], K=3, fs=fs)
            ch_modes_list.append(m_c)
        stacked_modes = np.concatenate(ch_modes_list, axis=0)
        feats_uni = [extract_9_features_1d(stacked_modes[k], fs=fs) for k in range(len(stacked_modes))]
        flat_vec = np.concatenate([np.array(list(f.values())) for f in feats_uni])
        X_v4.append(flat_vec)
    time_v4 = time.time() - t0
    m_v4 = evaluate_cross_validation(np.array(X_v4), y, classifier_type="knn", preset="ds2_adhd", n_splits=5)
    ablation_results["4. Univariate VMD (No MVMD Alignment)"] = (m_v4, time_v4)
    print(f"      Accuracy: {m_v4['accuracy']*100:.2f}% +/- {m_v4['accuracy_std']*100:.2f}% | F1: {m_v4['f1_score']*100:.2f}% | Time: {time_v4:.1f}s")

    # 5. SMVMD with Naive Zero-Padding
    print("\n[5/5] Evaluating Variant 5: SMVMD with Naive Zero-Padding (No EBFI)...")
    t0 = time.time()
    X_v5 = []
    max_fixed_modes = 6
    for epoch in clean_epochs:
        modes, omega_hz, _ = smvmd_decompose(epoch, fs=fs, alpha=2000.0, max_modes=max_fixed_modes)
        K, C, _ = modes.shape
        feats_naive = []
        for k in range(max_fixed_modes):
            if k < K:
                for c in range(C):
                    f_dict = extract_9_features_1d(modes[k, c, :], fs=fs)
                    feats_naive.extend(list(f_dict.values()))
            else:
                sample_len = C * 9
                feats_naive.extend([0.0] * sample_len)
        X_v5.append(np.array(feats_naive))
    time_v5 = time.time() - t0
    m_v5 = evaluate_cross_validation(np.array(X_v5), y, classifier_type="knn", preset="ds2_adhd", n_splits=5)
    ablation_results["5. SMVMD + Naive Zero-Padding (No EBFI)"] = (m_v5, time_v5)
    print(f"      Accuracy: {m_v5['accuracy']*100:.2f}% +/- {m_v5['accuracy_std']*100:.2f}% | F1: {m_v5['f1_score']*100:.2f}% | Time: {time_v5:.1f}s")

    # Comparative Table
    print("\n" + "=" * 75)
    print(" ABLATION STUDY COMPARISON TABLE")
    print("=" * 75)
    print("| Method / Pipeline Architecture | Accuracy (%) | Sensitivity (%) | Specificity (%) | F1-Score (%) | Execution Time |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for name, (metrics, exec_t) in ablation_results.items():
        acc = metrics["accuracy"] * 100
        sens = metrics["sensitivity"] * 100
        spec = metrics["specificity"] * 100
        f1 = metrics["f1_score"] * 100
        print(f"| **{name}** | **{acc:.2f}%** | {sens:.2f}% | {spec:.2f}% | **{f1:.2f}%** | {exec_t:.1f}s |")

    return ablation_results


if __name__ == "__main__":
    run_ablation_study()
