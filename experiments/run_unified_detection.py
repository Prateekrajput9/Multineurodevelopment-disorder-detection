"""
Experiment 3: Unified Multi-Disorder Diagnosis (Control vs ADHD vs IDD)
=======================================================================
Multi-disorder classification combining DS-1 and DS-2 methodologies
via SMVMD and Table I feature integration.
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
from smvmd.dataset import generate_synthetic_eeg_cohort
from smvmd.visualization import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_feature_importance,
)


def run_unified_experiment(
    n_controls: int = 20,
    n_adhd: int = 20,
    n_idd: int = 20,
    fs: float = 128.0,
    output_dir: str = "results/unified_detection",
):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print(" EXPERIMENT 3: UNIFIED MULTI-DISORDER DIAGNOSIS (CONTROL vs ADHD vs IDD)")
    print("=" * 75)

    dataset = generate_synthetic_eeg_cohort(
        n_controls=n_controls,
        n_adhd=n_adhd,
        n_idd=n_idd,
        duration_sec=10.0,
        epoch_len_sec=5.0,
        overlap_ratio=0.5,
        fs=fs,
        n_channels=14,
        random_state=42,
    )
    print(f"      Total 5s Epochs: {len(dataset)} | Classes: {dataset.class_names}")

    integrator = EnergyBasedFeatureIntegrator()
    X_list, y_list = [], []

    for idx in range(len(dataset)):
        raw_epoch, label, _ = dataset[idx]
        clean_epoch = preprocess_eeg(raw_epoch, fs=fs, lowcut=0.5, highcut=45.0)

        modes, omega_hz, _ = smvmd_decompose(clean_epoch, fs=fs, alpha=1500.0, max_modes=6)
        f_vec, f_names = integrator.integrate_multichannel_modes(modes, fs=fs, channel_names=dataset.channel_names)

        X_list.append(f_vec)
        y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    print("\n[2/3] Benchmarking Multi-Class Classifiers (10-Fold CV)...")
    classifiers = {
        "KNN (Table II Preset)": ("knn", {"preset": "ds2_adhd"}),
        "Support Vector Machine (SVM)": ("svm", {"kernel": "rbf", "C": 10.0}),
        "Random Forest": ("ensemble", {"n_estimators": 150}),
    }

    results = {}
    for name, (clf_type, kwargs) in classifiers.items():
        metrics = evaluate_cross_validation(
            X=X,
            y=y,
            classifier_type=clf_type,
            n_splits=10,
            class_names=dataset.class_names,
            **kwargs,
        )
        results[name] = metrics
        print(f"      {name:30s} -> Macro F1: {metrics['f1_score']*100:.2f}% | Accuracy: {metrics['accuracy']*100:.2f}% +/- {metrics['accuracy_std']*100:.2f}% | Kappa: {metrics['cohen_kappa']:.4f} | ROC-AUC: {metrics['roc_auc']*100:.2f}%")

    # Generate Figures
    best_m = results["KNN (Table II Preset)"]
    plot_confusion_matrix(
        cm=best_m["confusion_matrix"],
        class_names=dataset.class_names,
        title="Unified 3-Class Diagnosis: Confusion Matrix (SMVMD + KNN)",
        save_path=os.path.join(output_dir, "unified_confusion_matrix.png"),
    )

    clf_full = NeuroDisorderClassifier(classifier_type="knn", preset="ds2_adhd")
    clf_full.fit(X, y)
    probs = clf_full.predict_proba(X)
    plot_roc_curves(
        y_true=y,
        y_prob=probs,
        class_names=dataset.class_names,
        title="Multi-Class ROC Curves (Unified Diagnosis)",
        save_path=os.path.join(output_dir, "unified_roc_curves.png"),
    )

    print("\n" + "=" * 75)
    print(" UNIFIED 3-CLASS CLASSIFICATION REPORT")
    print("=" * 75)
    print(print_classification_report(best_m, title="Unified Multi-Class Diagnosis (10-Fold CV)"))

    return results


if __name__ == "__main__":
    run_unified_experiment()
