"""
Unit Tests for Table I Feature Extraction and Equation 7 Integration
"""

import unittest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smvmd.features import (
    extract_9_features_1d,
    extract_mode_features,
    extract_all_features,
    compute_std,
    compute_var,
    compute_rms,
    compute_iqr,
    compute_assr,
    compute_ap,
    compute_mfl,
    compute_ip,
    compute_pcc,
    FEATURE_NAMES_TABLE_1,
)
from smvmd.feature_integration import EnergyBasedFeatureIntegrator, integrate_mode_features


class TestFeaturesAndIntegration(unittest.TestCase):

    def setUp(self):
        self.fs = 128.0
        self.C = 4
        self.T = 256
        rng = np.random.RandomState(42)
        self.sig_1d = rng.randn(self.T)
        self.modes_3 = rng.randn(3, self.C, self.T)
        self.omega_3 = np.array([5.5, 10.2, 22.0])

    def test_table_1_features_1d(self):
        feats = extract_9_features_1d(self.sig_1d, fs=self.fs)
        for fname in FEATURE_NAMES_TABLE_1:
            self.assertIn(fname, feats)
            self.assertFalse(np.isnan(feats[fname]))
            self.assertFalse(np.isinf(feats[fname]))

        self.assertGreater(feats["STD"], 0)
        self.assertGreater(feats["VAR"], 0)
        self.assertGreater(feats["RMS"], 0)
        self.assertGreater(feats["AP"], 0)

    def test_eq_7_multichannel_integration(self):
        integrator = EnergyBasedFeatureIntegrator()
        vec, names = integrator.integrate_multichannel_modes(self.modes_3, fs=self.fs)

        expected_dim = self.C * len(FEATURE_NAMES_TABLE_1)  # 4 * 9 = 36
        self.assertEqual(len(vec), expected_dim)
        self.assertEqual(len(names), expected_dim)
        self.assertFalse(np.isnan(vec).any())
        self.assertFalse(np.isinf(vec).any())

        # Test with 2 modes vs 4 modes - dimension must always be C * 9 = 36!
        modes_2 = self.modes_3[:2]
        vec_2, names_2 = integrator.integrate_multichannel_modes(modes_2, fs=self.fs)
        self.assertEqual(len(vec_2), expected_dim)
        self.assertEqual(names_2, names)


if __name__ == "__main__":
    unittest.main()
