"""
Experiment 2: Pediatric IDD Detection (DS-1) Across 3 Scenarios
===============================================================
Reproduces the exact DS-1 methodology from Chandela et al. (IEEE TCDS 2025):
- 14 channels (Emotiv EPOC+), 5s windows, 50% overlap.
- SMVMD with alpha=2000, tau=0, tol=1e-10.
- Table I 9 features per channel integrated via Eq. (7) (D = 126 features).
- 3 Scenarios (Table III, Table V, Table VI):
  1. Rest State (KNN preset: Cityblock, Equal weights, Standardize=True)
  2. Music Stimuli (KNN preset: Cityblock, Squared inverse weights, Standardize=False)
  3. Rest & Music Combined (KNN preset: Euclidean, Squared inverse weights, Standardize=True)
- 10-fold cross-validation (Table III: 100% Accuracy for all 3 scenarios).
"""

import os
import sys
import time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smvmd.smvmd import smvmd_decompose
from smvmd.preprocessing import preprocess_eeg
from smvmd.feature_integration import EnergyBasedFeatureIntegrator
from smvmd.models import (
    evaluate_cross_validation,
    print_classification_report,
    NeuroDisorderClassifier,
)
from smvmd.dataset import generate_synthetic_eeg_cohort, DS1_14_CHANNELS
from smvmd.visualization import (
    plot_raw_vs_modes,
    plot_mode_spectra,
    plot_confusion_matrix,
    plot_feature_importance,
)


def extract_ds1_features(dataset, fs=128.0):
    integrator = EnergyBasedFeatureIntegrator()
    X_list, y_list, groups_list = [], [], []
    sample_modes, sample_omega, sample_raw = None, None, None

    for idx in range(len(dataset)):
        raw_epoch, label, sub_id = dataset[idx]
        clean_epoch = preprocess_eeg(raw_epoch, fs=fs, lowcut=1.0, highcut=30.0, notch_freq=50.0)

        modes, omega_hz, _ = smvmd_decompose(
            clean_epoch,
            fs=fs,
            alpha=2000.0,  # alpha=2000 for DS-1 from Section III-A
            tau=0.0,
            tol=1e-10,
            max_modes=6,
            tol_res=0.015,
        )

        f_vec, f_names = integrator.integrate_multichannel_modes(modes, fs=fs, channel_names=dataset.channel_names)
        X_list.append(f_vec)
        y_list.append(label)
        groups_list.append(sub_id)

        if sample_modes is None and label == 1:
            sample_modes, sample_omega, sample_raw = modes, omega_hz, clean_epoch

    return np.array(X_list), np.array(y_list), np.array(groups_list), f_names, sample_modes, sample_omega, sample_raw


def run_idd_experiment(
    n_controls: int = 20,
    n_idd: int = 20,
    fs: float = 128.0,
    output_dir: str = "results/idd_detection",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print(" EXPERIMENT 2: PEDIATRIC IDD DETECTION (DS-1) ACROSS 3 SCENARIOS")
    print(" Chandela, Faisal, & Sharma (IEEE TCDS 2025)")
    print("=" * 75)

    scenario_results = {}

    # Scenario 1: Rest State
    print("\n" + "-" * 60)
    print(" [SCENARIO 1] Rest State (DS-1 Rest)")
    print("-" * 60)
    ds_rest = generate_synthetic_eeg_cohort(
        n_controls=n_controls, n_adhd=0, n_idd=n_idd,
        duration_sec=10.0, epoch_len_sec=5.0, overlap_ratio=0.5, fs=fs, n_channels=14, random_state=42
    ).filter_classes([0, 2])

    X1, y1, g1, f_names1, s_modes, s_omega, s_raw = extract_ds1_features(ds_rest, fs=fs)
    print(f"      Feature Matrix: {X1.shape} (D = {X1.shape[1]} features, 14 channels * 9 Table I features)")

    m1 = evaluate_cross_validation(X1, y1, classifier_type="knn", preset="ds1_rest", n_splits=10, class_names=["TDC", "IDD"])
    scenario_results["Scenario 1 (Rest State)"] = m1
    print(f"      Rest State Accuracy: {m1['accuracy']*100:.2f}% +/- {m1['accuracy_std']*100:.2f}% | Sensitivity: {m1['sensitivity']*100:.2f}% | Specificity: {m1['specificity']*100:.2f}% | F1: {m1['f1_score']*100:.2f}%")

    # Scenario 2: Music Stimuli
    print("\n" + "-" * 60)
    print(" [SCENARIO 2] Music Stimuli (DS-1 Music)")
    print("-" * 60)
    ds_music = generate_synthetic_eeg_cohort(
        n_controls=n_controls, n_adhd=0, n_idd=n_idd,
        duration_sec=10.0, epoch_len_sec=5.0, overlap_ratio=0.5, fs=fs, n_channels=14, random_state=101
    ).filter_classes([0, 2])

    X2, y2, g2, _, _, _, _ = extract_ds1_features(ds_music, fs=fs)
    m2 = evaluate_cross_validation(X2, y2, classifier_type="knn", preset="ds1_music", n_splits=10, class_names=["TDC", "IDD"])
    scenario_results["Scenario 2 (Music Stimuli)"] = m2
    print(f"      Music Stimuli Accuracy: {m2['accuracy']*100:.2f}% +/- {m2['accuracy_std']*100:.2f}% | Sensitivity: {m2['sensitivity']*100:.2f}% | Specificity: {m2['specificity']*100:.2f}% | F1: {m2['f1_score']*100:.2f}%")

    # Scenario 3: Rest & Music Combined
    print("\n" + "-" * 60)
    print(" [SCENARIO 3] Rest & Music Combined (DS-1 Rest & Music)")
    print("-" * 60)
    X3 = np.vstack([X1, X2])
    y3 = np.concatenate([y1, y2])
    m3 = evaluate_cross_validation(X3, y3, classifier_type="knn", preset="ds1_combined", n_splits=10, class_names=["TDC", "IDD"])
    scenario_results["Scenario 3 (Rest & Music Combined)"] = m3
    print(f"      Rest & Music Combined Accuracy: {m3['accuracy']*100:.2f}% +/- {m3['accuracy_std']*100:.2f}% | Sensitivity: {m3['sensitivity']*100:.2f}% | Specificity: {m3['specificity']*100:.2f}% | F1: {m3['f1_score']*100:.2f}%")

    # Generate Figures
    if s_modes is not None:
        plot_raw_vs_modes(
            raw_signal=s_raw, modes=s_modes, omega_hz=s_omega, fs=fs,
            channel_idx=0, channel_name="AF3", title="IDD Pediatric EEG (DS-1): SMVMD Decomposition",
            save_path=os.path.join(output_dir, "idd_smvmd_modes.png"),
        )

    plot_confusion_matrix(
        cm=m1["confusion_matrix"], class_names=["TDC", "IDD"],
        title="Pediatric IDD Detection (DS-1 Rest): Confusion Matrix (Fig. 6a)",
        save_path=os.path.join(output_dir, "idd_confusion_matrix.png"),
    )

    rf_clf = NeuroDisorderClassifier(classifier_type="rf", n_estimators=150, random_state=42)
    rf_clf.fit(X1, y1)
    plot_feature_importance(
        feature_names=f_names1, importance_scores=rf_clf.model.feature_importances_, top_n=15,
        title="Top Discriminative Features for IDD Detection (DS-1)",
        save_path=os.path.join(output_dir, "idd_feature_importance.png"),
    )

    print("\n" + "=" * 75)
    print(" IDD 3-SCENARIO PERFORMANCE SUMMARY (TABLE III)")
    print("=" * 75)
    for sc_name, sc_m in scenario_results.items():
        print(print_classification_report(sc_m, title=sc_name))

    return scenario_results


if __name__ == "__main__":
    run_idd_experiment()
