"""
Energy-Based Feature Integration (EBFI) Module
==============================================
Exact implementation of Equation (7) from:
Chandela, Faisal, & Sharma (IEEE Transactions on Cognitive and Developmental Systems, 2025).

Equation (7):
f_{c,m} = [(f_{c,m,1} * e_{c,1}) + (f_{c,m,2} * e_{c,2}) + ... + (f_{c,m,n} * e_{c,n})] /
          [e_{c,1} + e_{c,2} + ... + e_{c,n}]

Where:
- f_{c,m}   : m-th integrated feature of c-th channel
- f_{c,m,n} : m-th feature from n-th decomposed mode of c-th channel
- e_{c,n}   : energy of the n-th decomposed mode of c-th channel
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from .features import FEATURE_NAMES_TABLE_1, extract_9_features_1d


class EnergyBasedFeatureIntegrator:
    """
    Implements Equation (7) of Chandela et al. (2025).
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_NAMES_TABLE_1

    def integrate_multichannel_modes(
        self,
        modes: np.ndarray,  # shape: (K, C, T)
        fs: float = 128.0,
        channel_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Apply Equation (7) across all channels and modes.

        Parameters
        ----------
        modes : np.ndarray, shape (K, C, T)
            Extracted MVMF modes.
        fs : float
            Sampling frequency in Hz.
        channel_names : list of str, optional

        Returns
        -------
        feature_vector : np.ndarray, shape (C * 9,)
            Integrated feature vector.
        names : list of str (length C * 9)
        """
        if modes.ndim == 2:
            modes = modes[:, np.newaxis, :]

        K, C, T = modes.shape
        ch_labels = channel_names or [f"Ch{c+1}" for c in range(C)]

        integrated_features = []
        feature_vector_names = []

        # For each channel c
        for c in range(C):
            ch_name = ch_labels[c] if c < len(ch_labels) else f"Ch{c+1}"

            # Compute energies e_{c,n} for each mode n = 1 ... K
            mode_energies = np.array([float(np.sum(modes[n, c, :] ** 2)) for n in range(K)])
            total_ch_energy = float(np.sum(mode_energies)) + 1e-12
            weights = mode_energies / total_ch_energy

            # Extract 9 Table I features for each mode n of channel c
            mode_feat_dicts = [extract_9_features_1d(modes[n, c, :], fs=fs) for n in range(K)]

            # Apply Eq. (7) for each feature m
            for m_name in self.feature_names:
                vals = np.array([mode_feat_dicts[n][m_name] for n in range(K)])
                f_cm = float(np.sum(weights * vals))

                integrated_features.append(f_cm)
                feature_vector_names.append(f"{ch_name}_{m_name}")

        feat_vec = np.array(integrated_features, dtype=np.float64)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1e6, neginf=-1e6)

        return feat_vec, feature_vector_names

    def integrate(
        self,
        mode_features_list: List[Dict[str, float]],
        omega_hz: np.ndarray,
        total_energy: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Fallback interface when mode features are precomputed.
        """
        K = len(mode_features_list)
        if K == 0:
            raise ValueError("mode_features_list is empty")

        energies = np.array([m.get("energy", 1.0) for m in mode_features_list])
        total_e = np.sum(energies) + 1e-12
        weights = energies / total_e

        integrated_dict = {}
        for feat_name in self.feature_names:
            vals = np.array([m.get(feat_name, 0.0) for m in mode_features_list])
            integrated_dict[feat_name] = float(np.sum(weights * vals))

        names = sorted(list(integrated_dict.keys()))
        vec = np.array([integrated_dict[k] for k in names], dtype=np.float64)
        return vec, names


def integrate_mode_features(
    modes: np.ndarray,
    fs: float = 128.0,
    channel_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convenience function for Eq. (7) integration.
    """
    integrator = EnergyBasedFeatureIntegrator()
    return integrator.integrate_multichannel_modes(modes, fs=fs, channel_names=channel_names)
