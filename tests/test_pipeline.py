"""
Integration Tests for End-to-End Classification Pipeline
"""

import unittest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smvmd.dataset import generate_synthetic_eeg_cohort
from smvmd.preprocessing import preprocess_eeg
from smvmd.smvmd import smvmd_decompose
from smvmd.feature_integration import EnergyBasedFeatureIntegrator
from smvmd.models import (
    NeuroDisorderClassifier,
    evaluate_cross_validation,
    compute_metrics,
)


class TestPipeline(unittest.TestCase):

    def test_end_to_end_pipeline(self):
        fs = 128.0
        # Generate small cohort
        dataset = generate_synthetic_eeg_cohort(
            n_controls=6,
            n_adhd=6,
            n_idd=0,
            duration_sec=5.0,
            epoch_len_sec=2.5,
            fs=fs,
            n_channels=4,
            random_state=42,
        )

        self.assertGreater(len(dataset), 5)
        integrator = EnergyBasedFeatureIntegrator()

        X_list, y_list, groups_list = [], [], []

        for idx in range(min(10, len(dataset))):
            raw_sig, label, sub_id = dataset[idx]
            clean = preprocess_eeg(raw_sig, fs=fs)
            modes, omega_hz, _ = smvmd_decompose(clean, fs=fs, max_modes=3)
            vec, _ = integrator.integrate_multichannel_modes(modes, fs=fs)

            X_list.append(vec)
            y_list.append(label)
            groups_list.append(sub_id)

        X = np.array(X_list)
        y = np.array(y_list)

        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(X.shape[1], 4 * 9)  # 4 channels * 9 Table I features

        # Classifier fitting and prediction with Table II preset
        clf = NeuroDisorderClassifier(classifier_type="knn", preset="ds2_adhd")
        clf.fit(X, y)
        preds = clf.predict(X)
        probs = clf.predict_proba(X)

        self.assertEqual(len(preds), len(y))
        self.assertEqual(probs.shape[0], len(y))

        m = compute_metrics(y, preds, probs)
        self.assertIn("accuracy", m)
        self.assertIn("f1_score", m)
        self.assertIn("cohen_kappa", m)


if __name__ == "__main__":
    unittest.main()
