"""
Unit Tests for SMVMD
"""

import unittest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from smvmd.smvmd import SuccessiveMVMD, smvmd_decompose, mvmd_decompose, svmd_decompose


class TestSMVMD(unittest.TestCase):

    def setUp(self):
        self.fs = 128.0
        self.T = 256
        self.t = np.arange(self.T) / self.fs
        self.C = 3
        self.sig = np.zeros((self.C, self.T))
        for c in range(self.C):
            self.sig[c, :] = (
                1.5 * np.sin(2 * np.pi * 6.0 * self.t + c * 0.2)
                + 1.0 * np.sin(2 * np.pi * 10.0 * self.t + c * 0.4)
                + 0.6 * np.sin(2 * np.pi * 20.0 * self.t + c * 0.1)
                + 0.05 * np.random.randn(self.T)
            )

    def test_smvmd_decompose_multichannel(self):
        modes, omega_hz, info = smvmd_decompose(
            self.sig,
            fs=self.fs,
            alpha=2000.0,
            max_modes=4,
            tol_res=0.01,
        )

        K, C, T = modes.shape
        self.assertEqual(C, self.C)
        self.assertEqual(T, self.T)
        self.assertGreater(K, 0)
        self.assertEqual(len(omega_hz), K)
        for cf in omega_hz:
            self.assertGreater(cf, 0.0)
            self.assertLessEqual(cf, self.fs / 2.0)

    def test_smvmd_1d_input(self):
        sig_1d = self.sig[0, :]
        modes, omega_hz, info = smvmd_decompose(
            sig_1d,
            fs=self.fs,
            max_modes=3,
        )
        self.assertEqual(modes.ndim, 2)
        self.assertEqual(modes.shape[1], self.T)

    def test_mvmd_fixed_k(self):
        K_fixed = 3
        modes, omega_hz, info = mvmd_decompose(
            self.sig,
            K=K_fixed,
            fs=self.fs,
        )
        self.assertEqual(modes.shape[0], K_fixed)
        self.assertEqual(modes.shape[1], self.C)
        self.assertEqual(modes.shape[2], self.T)


if __name__ == "__main__":
    unittest.main()
