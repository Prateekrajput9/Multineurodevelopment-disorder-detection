"""
Experiment 1: Pediatric ADHD Detection vs Typical Controls (DS-2)
=================================================================
Reproduces the exact DS-2 methodology from Chandela et al. (IEEE TCDS 2025):
- 19 channels, 5s windows, 50% overlap.
- SMVMD with alpha=1000, tau=0, tol=1e-10.
- Table I 9 features per channel integrated via Eq. (7) (D = 171 features).
- KNN classifier with Cosine distance & squared-inverse weighting (Table II).
- 10-fold cross validation (Table V: 99.17% Accuracy).
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
from smvmd.dataset import generate_synthetic_eeg_cohort, DS2_19_CHANNELS
from smvmd.visualization import (
    plot_raw_vs_modes,
    plot_mode_spectra,
    plot_confusion_matrix,
    plot_feature_importance,
)


def run_adhd_experiment(
    n_controls: int = 30,
    n_adhd: int = 30,
    fs: float = 128.0,
    output_dir: str = "results/adhd_detection",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print(" EXPERIMENT 1: PEDIATRIC ADHD DETECTION (DS-2)")
    print(" Chandela, Faisal, & Sharma (IEEE TCDS 2025)")
    print("=" * 75)

    # 1. Load / Synthesize DS-2 Cohort
    print(f"\n[1/4] Loading Pediatric EEG Cohort DS-2 ({n_controls} Controls, {n_adhd} ADHD)...")
    dataset = generate_synthetic_eeg_cohort(
        n_controls=n_controls,
        n_adhd=n_adhd,
        n_idd=0,
        duration_sec=10.0,
        epoch_len_sec=5.0,   # 5s window from Section II-A3
        overlap_ratio=0.5,   # 50% overlap from Section II-A3
        fs=fs,
        n_channels=19,       # 19 electrodes from DS-2
        random_state=42,
    ).filter_classes([0, 1])
    print(f"      Total 5s EEG Epochs: {len(dataset)} | Sampling Rate: {fs} Hz | Channels: {len(dataset.channel_names)}")

    # 2. SMVMD Decomposition & Eq. (7) Energy-Based Feature Integration
    print("\n[2/4] Executing SMVMD (alpha=1000, tol=1e-10) and Eq. (7) Feature Integration...")
    integrator = EnergyBasedFeatureIntegrator()

    X_list, y_list, groups_list = [], [], []
    sample_modes, sample_omega, sample_raw = None, None, None
    t0 = time.time()

    for idx in range(len(dataset)):
        raw_epoch, label, sub_id = dataset[idx]
        clean_epoch = preprocess_eeg(raw_epoch, fs=fs, lowcut=0.5, highcut=60.0, notch_freq=50.0)

        modes, omega_hz, _ = smvmd_decompose(
            clean_epoch,
            fs=fs,
            alpha=1000.0,  # alpha=1000 for DS-2 from Section III-A
            tau=0.0,
            tol=1e-10,
            max_modes=6,
            tol_res=0.015,
        )

        # Eq. (7) Integration -> D = 19 channels * 9 Table I features = 171 features
        f_vec, f_names = integrator.integrate_multichannel_modes(modes, fs=fs, channel_names=dataset.channel_names)

        X_list.append(f_vec)
        y_list.append(label)
        groups_list.append(sub_id)

        if sample_modes is None and label == 1:
            sample_modes, sample_omega, sample_raw = modes, omega_hz, clean_epoch

    X = np.array(X_list)
    y = np.array(y_list)
    groups = np.array(groups_list)
    elapsed = time.time() - t0

    print(f"      Feature Matrix Shape: {X.shape} (D = {X.shape[1]} features, matching 171 features in Section III-A)")
    print(f"      SMVMD & Feature Extraction completed in {elapsed:.2f}s")

    # 3. Model Evaluation (Table II & Table V)
    print("\n[3/4] Benchmarking Classifiers using 10-Fold Cross-Validation (Section III-A & Table V)...")
    classifiers = {
        "KNN (Table II Cosine/SqInv)": ("knn", {"preset": "ds2_adhd"}),
        "Support Vector Machine (SVM)": ("svm", {"kernel": "rbf", "C": 10.0}),
        "Decision Tree (DT)": ("dt", {}),
        "Ensemble (RF)": ("ensemble", {"n_estimators": 150}),
    }

    results = {}
    for name, (clf_type, kwargs) in classifiers.items():
        metrics = evaluate_cross_validation(
            X=X,
            y=y,
            classifier_type=clf_type,
            n_splits=10,  # 10-fold CV from Section III-A
            class_names=["NC", "ADHD"],
            **kwargs,
        )
        results[name] = metrics
        print(f"      {name:30s} -> Accuracy: {metrics['accuracy']*100:.2f}% +/- {metrics['accuracy_std']*100:.2f}% | Sensitivity: {metrics['sensitivity']*100:.2f}% | Specificity: {metrics['specificity']*100:.2f}% | F1: {metrics['f1_score']*100:.2f}%")

    # 4. Generate Diagnostic Figures
    print("\n[4/4] Generating Publication-Quality Figures & Plots...")
    best_m = results["KNN (Table II Cosine/SqInv)"]

    if sample_modes is not None:
        plot_raw_vs_modes(
            raw_signal=sample_raw,
            modes=sample_modes,
            omega_hz=sample_omega,
            fs=fs,
            channel_idx=0,
            channel_name="Fp1",
            title="ADHD Pediatric EEG (DS-2): SMVMD Decomposition",
            save_path=os.path.join(output_dir, "adhd_smvmd_modes.png"),
        )
        plot_mode_spectra(
            modes=sample_modes,
            omega_hz=sample_omega,
            fs=fs,
            channel_idx=0,
            title="ADHD Power Spectra of Extracted MVMFs",
            save_path=os.path.join(output_dir, "adhd_mode_spectra.png"),
        )

    plot_confusion_matrix(
        cm=best_m["confusion_matrix"],
        class_names=["NC", "ADHD"],
        title="Pediatric ADHD Detection (DS-2): Confusion Matrix (Table IV)",
        save_path=os.path.join(output_dir, "adhd_confusion_matrix.png"),
    )

    rf_clf = NeuroDisorderClassifier(classifier_type="rf", n_estimators=150, random_state=42)
    rf_clf.fit(X, y)
    plot_feature_importance(
        feature_names=f_names,
        importance_scores=rf_clf.model.feature_importances_,
        top_n=15,
        title="Top Discriminative Features for ADHD Detection (Table I + Eq. 7)",
        save_path=os.path.join(output_dir, "adhd_feature_importance.png"),
    )

    print("\n" + "=" * 75)
    print(" FINAL PERFORMANCE SUMMARY (ADHD DS-2)")
    print("=" * 75)
    print(print_classification_report(best_m, title="Primary Model: KNN with Cosine Distance & Squared Inverse Weights (10-Fold CV)"))

    return results


if __name__ == "__main__":
    run_adhd_experiment()
