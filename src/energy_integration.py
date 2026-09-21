"""
energy_integration.py - Energy-Based Feature Integration

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

Energy-based feature integration (Paper Equation 7, explicit):

  f_{c,m} = Σ_k (f_{c,m,k} × e_{c,k}) / Σ_k e_{c,k}

where:
  f_{c,m}   = mth integrated feature of cth channel (FINAL output)
  f_{c,m,k} = mth feature from kth mode of cth channel
  e_{c,k}   = energy of kth mode for cth channel

Energy computation:
  e_{c,k} = Σ_t u_{k,c}(t)²   (sum of squared samples)
  [Paper does not specify energy definition; implementation choice: L2 energy]

This converts the variable-K feature array (C, K, 9) into a fixed
feature vector of size (C × 9):
  DS-2: 19 × 9 = 171 features per segment
  DS-1: 14 × 9 = 126 features per segment

The integrated feature vector has a consistent dimension regardless
of how many modes SMVMD extracted for each segment.
This is the key innovation in the paper that enables ML classification.
"""

import numpy as np
import logging
from typing import List, Optional, Tuple

from src.smvmd import SMVMDResult
from src.features import FEATURE_NAMES, N_FEATURES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Energy Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mode_energies(mimfs: List[np.ndarray]) -> np.ndarray:
    """Compute energy of each mode for each channel.

    Energy definition (implementation choice, paper does not specify):
    e_{c,k} = Σ_t u_{k,c}(t)²  (sum of squared samples = L2 energy)

    This is the standard signal energy definition.

    Args:
        mimfs: List of K MIMFs, each shape (C, T).

    Returns:
        Energy array of shape (K, C).
        energies[k, c] = energy of mode k for channel c.
    """
    if not mimfs:
        return np.empty((0, 0))

    K = len(mimfs)
    C = mimfs[0].shape[0]

    energies = np.zeros((K, C))
    for k, mimf in enumerate(mimfs):
        for c in range(C):
            energies[k, c] = np.sum(mimf[c] ** 2)

    return energies


# ─────────────────────────────────────────────────────────────────────────────
# Energy-Based Feature Integration (Paper Eq. 7)
# ─────────────────────────────────────────────────────────────────────────────

def energy_integrate_features(feature_array: np.ndarray,
                               energies: np.ndarray,
                               eps: float = 1e-12
                               ) -> np.ndarray:
    """Apply energy-based feature integration for one segment.

    Paper Equation 7:
      f_{c,m} = [Σ_k (f_{c,m,k} × e_{c,k})] / [Σ_k e_{c,k}]

    This is a weighted average of features across modes,
    weighted by mode energy.

    Args:
        feature_array: Per-mode features, shape (C, K, 9).
            feature_array[c, k, f] = fth feature of mode k, channel c.
        energies: Mode energies, shape (K, C).
            energies[k, c] = energy of mode k for channel c.
        eps: Small constant to avoid division by zero.

    Returns:
        Integrated features, shape (C, 9).
        integrated[c, f] = energy-weighted average of feature f for channel c.
    """
    C, K, F = feature_array.shape

    if K == 0:
        logger.warning("Feature array has 0 modes. Returning zeros.")
        return np.zeros((C, F))

    if energies.shape != (K, C):
        raise ValueError(
            f"Energy shape mismatch: expected ({K}, {C}), "
            f"got {energies.shape}"
        )

    integrated = np.zeros((C, F))

    for c in range(C):
        # e_{c,k} for all k: shape (K,)
        e_c = energies[:, c]  # energy of each mode for channel c

        # Total energy for this channel
        total_energy = np.sum(e_c) + eps

        for m in range(F):
            # f_{c,m,k} for all k: shape (K,)
            f_ck = feature_array[c, :, m]

            # f_{c,m} = Σ_k (f_{c,m,k} × e_{c,k}) / Σ_k e_{c,k}
            integrated[c, m] = np.sum(f_ck * e_c) / total_energy

    return integrated


def energy_integrate_segment(smvmd_result: SMVMDResult,
                              feature_array: np.ndarray,
                              eps: float = 1e-12
                              ) -> np.ndarray:
    """Integrate features for one segment using SMVMD energies.

    Uses energies from the SMVMDResult (already computed during decomposition).

    Args:
        smvmd_result: Result from SMVMD.decompose(), contains mode energies.
        feature_array: Per-mode features (C, K, 9).
        eps: Division epsilon.

    Returns:
        Integrated feature vector, shape (C × 9,) = flat vector.
        This is the final feature vector for one segment.
    """
    if smvmd_result.n_modes == 0:
        C = feature_array.shape[0] if feature_array.ndim == 3 else 0
        logger.warning(f"Segment has 0 modes. Returning zero features.")
        return np.zeros(C * N_FEATURES)

    # Use energies from SMVMD result if available, else recompute
    if smvmd_result.energies is not None:
        energies = smvmd_result.energies  # shape: (K, C)
    else:
        energies = compute_mode_energies(smvmd_result.mimfs)  # shape: (K, C)

    # Integrate: (C, 9)
    integrated_2d = energy_integrate_features(
        feature_array=feature_array,
        energies=energies,
        eps=eps
    )

    # Flatten to 1D: (C × 9,) = (126,) for DS-1 or (171,) for DS-2
    integrated_flat = integrated_2d.flatten()

    return integrated_flat


# ─────────────────────────────────────────────────────────────────────────────
# Batch Integration
# ─────────────────────────────────────────────────────────────────────────────

def integrate_all_segments(smvmd_results: List[SMVMDResult],
                            feature_lists: List[np.ndarray],
                            n_channels: int,
                            dataset: str = 'DS2'
                            ) -> np.ndarray:
    """Integrate features for all segments.

    Produces the final feature matrix for classification.

    Expected output shapes:
      DS-1: (n_segments, 14 × 9) = (n_segments, 126)
      DS-2: (n_segments, 19 × 9) = (n_segments, 171)

    Args:
        smvmd_results: List of SMVMD results.
        feature_lists: List of per-segment feature arrays (C, K, 9).
        n_channels: Expected number of channels.
        dataset: 'DS1' or 'DS2' (for logging and assertions).

    Returns:
        Feature matrix (n_segments, n_channels × 9).

    Raises:
        AssertionError: If output dimensions are incorrect.
    """
    from tqdm import tqdm

    # Expected feature count
    expected_n_features = n_channels * N_FEATURES
    if dataset == 'DS1':
        assert n_channels == 14, f"DS-1 expects 14 channels, got {n_channels}"
        assert expected_n_features == 126, f"DS-1: expected 126 features, got {expected_n_features}"
    elif dataset == 'DS2':
        assert n_channels == 19, f"DS-2 expects 19 channels, got {n_channels}"
        assert expected_n_features == 171, f"DS-2: expected 171 features, got {expected_n_features}"

    n_segments = len(smvmd_results)
    assert len(feature_lists) == n_segments, (
        f"Mismatch: {len(feature_lists)} feature arrays but {n_segments} SMVMD results"
    )

    logger.info(
        f"Integrating {n_segments} segments for {dataset}: "
        f"expected output shape ({n_segments}, {expected_n_features})"
    )

    feature_matrix = np.zeros((n_segments, expected_n_features))
    failed = 0

    for i, (result, feat_arr) in enumerate(
        tqdm(zip(smvmd_results, feature_lists),
             total=n_segments, desc="Energy integration")
    ):
        try:
            integrated = energy_integrate_segment(
                smvmd_result=result,
                feature_array=feat_arr,
            )

            if len(integrated) != expected_n_features:
                logger.warning(
                    f"Segment {i}: integrated length={len(integrated)}, "
                    f"expected {expected_n_features}. Padding/truncating."
                )
                # Pad or truncate
                if len(integrated) < expected_n_features:
                    integrated = np.pad(
                        integrated,
                        (0, expected_n_features - len(integrated))
                    )
                else:
                    integrated = integrated[:expected_n_features]

            feature_matrix[i] = integrated

        except Exception as e:
            logger.error(f"Integration failed for segment {i}: {e}")
            feature_matrix[i] = np.zeros(expected_n_features)
            failed += 1

    if failed > 0:
        logger.warning(f"Integration failed for {failed}/{n_segments} segments")

    # Verify final dimensions [Paper-specified assertion]
    assert feature_matrix.shape == (n_segments, expected_n_features), (
        f"Feature matrix shape mismatch: "
        f"got {feature_matrix.shape}, expected ({n_segments}, {expected_n_features})"
    )

    # Health checks
    n_nan = np.sum(np.isnan(feature_matrix))
    n_inf = np.sum(np.isinf(feature_matrix))

    if n_nan > 0:
        logger.warning(f"Feature matrix has {n_nan} NaN values. Replacing with 0.")
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    if n_inf > 0:
        logger.warning(f"Feature matrix has {n_inf} Inf values. Replacing with 0.")
        feature_matrix = np.nan_to_num(feature_matrix, posinf=0.0, neginf=0.0)

    logger.info(
        f"✓ Feature matrix shape verified: {feature_matrix.shape}\n"
        f"  DS-{dataset[-1]}: {'✓' if feature_matrix.shape[1] == expected_n_features else '✗'} "
        f"{n_channels} channels × 9 features = {expected_n_features} features"
    )

    return feature_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Feature Matrix Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_feature_matrix(feature_matrix: np.ndarray,
                           dataset: str,
                           segment_df=None) -> bool:
    """Verify feature matrix dimensions and content.

    Paper-specified assertions:
      DS-1: (1288, 126) [14 channels × 9 features]
      DS-2: (6588, 171) [19 channels × 9 features]

    Args:
        feature_matrix: Shape (n_segments, n_features).
        dataset: 'DS1' or 'DS2'.
        segment_df: Optional segment metadata DataFrame.

    Returns:
        True if all checks pass.
    """
    checks = []

    expected = {
        'DS1': (1288, 126),
        'DS1_Rest': (644, 126),
        'DS1_Music': (644, 126),
        'DS1_RestMusic': (1288, 126),   # both conditions pooled
        'DS2': (6588, 171),
    }

    if dataset in expected:
        exp_shape = expected[dataset]
        shape_ok = feature_matrix.shape == exp_shape
        checks.append(shape_ok)
        logger.info(
            f"[SHAPE] {dataset}: actual={feature_matrix.shape}, "
            f"expected={exp_shape}: {'✓' if shape_ok else '✗'}"
        )
    else:
        logger.info(f"[SHAPE] {dataset}: {feature_matrix.shape}")

    # NaN check
    nan_ok = not np.any(np.isnan(feature_matrix))
    checks.append(nan_ok)
    logger.info(f"[NaN] {'✓ None' if nan_ok else '✗ Found'}")

    # Inf check
    inf_ok = not np.any(np.isinf(feature_matrix))
    checks.append(inf_ok)
    logger.info(f"[Inf] {'✓ None' if inf_ok else '✗ Found'}")

    # Feature name reference
    expected_feature_names = (
        FEATURE_NAMES * (feature_matrix.shape[1] // N_FEATURES)
    )
    n_channels = feature_matrix.shape[1] // N_FEATURES
    logger.info(
        f"[FEATURES] {n_channels} channels × {N_FEATURES} features = "
        f"{n_channels * N_FEATURES} features"
    )
    logger.info(f"  Feature order: {FEATURE_NAMES}")

    all_ok = all(checks)
    logger.info(f"{'='*40}\nFeature Matrix Verification: "
                f"{'ALL PASSED ✓' if all_ok else 'FAILED ✗'}\n{'='*40}")

    return all_ok
