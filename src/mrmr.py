"""
mrmr.py - Maximum Relevance Minimum Redundancy Feature Selection

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

mRMR [Paper explicit]:
  - Uses Maximum Relevance Minimum Redundancy algorithm
  - Features ranked by mRMR criterion
  - Classification accuracy computed for increasing feature counts (Fig. 5)
  - Optimal feature count = max accuracy achieved with minimum features

Paper-reported selected feature counts:
  DS-1 Rest:         80 features (from 126 total)
  DS-1 Music:        47 features (from 126 total)
  DS-1 Rest+Music:  103 features (from 126 total)
  DS-2:             171 features (all; no reduction)

[Paper does not specify mRMR variant: MI vs F-statistic]
[Implementation choice: mutual information-based mRMR]

[Paper does not specify if mRMR applied globally or per-fold]
[Implementation choice: per-fold on training data (methodologically strict)]
NOTE: Global mRMR (applied before CV) is common in the field and may
match the paper's MATLAB workflow more closely. Both modes implemented.
"""

import numpy as np
import logging
import warnings
from typing import List, Tuple, Optional, Dict, Any

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# mRMR Implementation
# ─────────────────────────────────────────────────────────────────────────────

def compute_mrmr_ranking(X: np.ndarray,
                          y: np.ndarray,
                          n_features: Optional[int] = None,
                          method: str = 'MI',
                          max_samples_redundancy: int = 2000,
                          random_seed: int = 42) -> np.ndarray:
    """Compute mRMR feature ranking.

    Maximum Relevance Minimum Redundancy criterion:
      min Σ_{i≠j} I(f_i; f_j) - Σ_i I(f_i; y)

    Greedy selection: At each step, select the feature that maximizes:
      score(f) = I(f; y) - (1/|S|) * Σ_{j∈S} I(f; f_j)

    where S is the set of already-selected features.

    [Paper does not specify variant (MI vs F-stat); implementation choice: MI]

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Class labels (n_samples,).
        n_features: Number of features to rank. If None, rank all.
        method: 'MI' (mutual information) or 'F' (F-statistic).
        max_samples_redundancy: Cap on the number of rows used to estimate the
            feature-feature redundancy matrix. The relevance term I(f; y) always
            uses every row; only the O(p^2) redundancy estimate is subsampled.
            See _mrmr_mi for why this matters.
        random_seed: Seed for that subsample.

    Returns:
        Ranked feature indices, length n_features.
        indices[0] = most relevant feature index.
    """
    if method == 'MI':
        return _mrmr_mi(X, y, n_features,
                        max_samples_redundancy=max_samples_redundancy,
                        random_seed=random_seed)
    elif method == 'F':
        return _mrmr_fstat(X, y, n_features)
    else:
        raise ValueError(f"Unknown mRMR method: {method}. Use 'MI' or 'F'.")


def _mrmr_mi(X: np.ndarray,
              y: np.ndarray,
              n_features: Optional[int] = None,
              max_samples_redundancy: int = 2000,
              random_seed: int = 42) -> np.ndarray:
    """Greedy mRMR using mutual information.

    Cost note. Relevance, I(f; y), is one call over the whole matrix and is
    cheap. Redundancy needs a p x p matrix of I(f_i; f_j), i.e. p calls to
    scikit-learn's kNN-based estimator, and that estimator scales with the
    number of rows. On the DS-2 matrix (6588 x 171) a single call takes ~54 s,
    so the full matrix would take about 2.5 hours -- longer than the entire
    rest of the pipeline.

    The kNN mutual-information estimate converges in the number of samples
    well before several thousand rows, so the redundancy matrix is estimated
    from a random subsample of at most `max_samples_redundancy` rows. The
    relevance term, which is what actually drives the top of the ranking, still
    uses every row. The subsample is seeded, so the ranking is reproducible.

    [Paper does not describe its mRMR implementation beyond naming the
     algorithm; this is an implementation choice, documented in
     docs/reproducibility_notes.md]

    Args:
        X: (n_samples, n_total_features)
        y: (n_samples,)
        n_features: How many features to rank.
        max_samples_redundancy: Row cap for the redundancy matrix.
        random_seed: Seed for the subsample.

    Returns:
        Ranked indices array.
    """
    try:
        from sklearn.feature_selection import mutual_info_classif
    except ImportError:
        logger.error("scikit-learn required for mRMR")
        return np.arange(X.shape[1])

    n_total = X.shape[1]
    if n_features is None:
        n_features = n_total
    n_features = min(n_features, n_total)

    logger.info(f"Computing mRMR ranking: {n_total} features → top {n_features}")

    # Relevance: I(f_i; y) for all features
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        relevance = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    # Precompute the feature-feature MI matrix for the redundancy term.
    # Subsample rows first: this is p separate kNN-MI estimates, and the
    # estimator's cost grows with the number of rows (see the docstring).
    from sklearn.feature_selection import mutual_info_regression

    n_samples = X.shape[0]
    if max_samples_redundancy and n_samples > max_samples_redundancy:
        rng = np.random.default_rng(random_seed)
        sub_idx = rng.choice(n_samples, max_samples_redundancy, replace=False)
        X_red = X[sub_idx]
        logger.info(
            f"Redundancy matrix estimated from a random {max_samples_redundancy} "
            f"of {n_samples} rows (seeded). Relevance uses all rows."
        )
    else:
        X_red = X

    logger.info(f"Precomputing {n_total}x{n_total} MI matrix for redundancy...")
    mi_matrix = np.zeros((n_total, n_total), dtype=np.float32)
    for j in range(n_total):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi_matrix[:, j] = mutual_info_regression(
                X_red, X_red[:, j], discrete_features=False,
                random_state=random_seed
            )
    mi_matrix = (mi_matrix + mi_matrix.T) / 2.0
    logger.info("MI matrix ready. Running greedy selection...")

    # Greedy mRMR selection using cached matrix (O(n) lookups per step)
    selected = []
    remaining = list(range(n_total))

    for step in range(n_features):
        if not remaining:
            break
        rem_arr = np.array(remaining)
        if step == 0:
            best_feat = int(rem_arr[np.argmax(relevance[rem_arr])])
        else:
            sel_arr = np.array(selected)
            redundancy = mi_matrix[np.ix_(rem_arr, sel_arr)].mean(axis=1)
            scores = relevance[rem_arr] - redundancy
            best_feat = int(rem_arr[np.argmax(scores)])

        selected.append(best_feat)
        remaining.remove(best_feat)
        if step % 10 == 0:
            logger.debug(f"mRMR step {step+1}/{n_features}: selected feature {best_feat}")

    logger.info(f"mRMR done: {len(selected)} features selected.")
    return np.array(selected)



def _mrmr_fstat(X: np.ndarray,
                 y: np.ndarray,
                 n_features: Optional[int] = None) -> np.ndarray:
    """Greedy mRMR using F-statistic for relevance.

    Alternative to MI-based mRMR.

    [Paper does not specify variant; this is an alternative implementation]
    """
    from sklearn.feature_selection import f_classif

    n_total = X.shape[1]
    if n_features is None:
        n_features = n_total

    n_features = min(n_features, n_total)

    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0)

    selected = []
    remaining = list(range(n_total))

    for step in range(n_features):
        if not remaining:
            break

        if step == 0:
            best_idx = np.argmax([f_scores[i] for i in remaining])
            selected.append(remaining[best_idx])
            remaining.pop(best_idx)
        else:
            best_score = -np.inf
            best_pos = None

            for pos, feat_idx in enumerate(remaining):
                rel = f_scores[feat_idx]

                # Pearson correlation redundancy
                red = 0.0
                for sel_idx in selected:
                    corr = np.abs(np.corrcoef(X[:, feat_idx], X[:, sel_idx])[0, 1])
                    red += corr
                red /= len(selected)

                score = rel - red
                if score > best_score:
                    best_score = score
                    best_pos = pos

            if best_pos is not None:
                selected.append(remaining[best_pos])
                remaining.pop(best_pos)

    return np.array(selected)


def _mutual_info_1d(x: np.ndarray, y: np.ndarray,
                     n_bins: int = 10) -> float:
    """Estimate mutual information between two continuous variables.

    Uses binned entropy estimation.

    [Implementation choice for mRMR redundancy computation]
    """
    x_binned = np.digitize(x, np.linspace(x.min(), x.max(), n_bins))
    y_binned = np.digitize(y, np.linspace(y.min(), y.max(), n_bins))

    # Joint probability
    joint = np.histogram2d(x_binned, y_binned,
                           bins=n_bins)[0] / len(x)
    p_x = joint.sum(axis=1)
    p_y = joint.sum(axis=0)

    # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
    outer = np.outer(p_x, p_y)
    mi = 0.0
    mask = (joint > 0) & (outer > 0)
    mi = np.sum(joint[mask] * np.log(joint[mask] / outer[mask]))

    return max(0.0, float(mi))


# ─────────────────────────────────────────────────────────────────────────────
# Feature Selection with Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

def select_features_by_accuracy(X: np.ndarray,
                                  y: np.ndarray,
                                  ranked_indices: np.ndarray,
                                  knn_params: Dict[str, Any],
                                  cv_folds: int = 10,
                                  random_seed: int = 42
                                  ) -> Tuple[int, np.ndarray, np.ndarray]:
    """Select optimal feature count by cross-validated accuracy.

    Paper: "Classification accuracy for the ranked features is computed,
    and the maximum accuracy for the lowest number of features is selected."
    (Fig. 5 in paper)

    Evaluates KNN with increasing numbers of mRMR-ranked features.
    Selects the smallest n_features achieving maximum accuracy.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Labels (n_samples,).
        ranked_indices: mRMR-ranked feature indices.
        knn_params: KNN parameters for evaluation.
        cv_folds: Number of CV folds.
        random_seed: Random seed.

    Returns:
        Tuple of:
        - optimal_n: Optimal number of features.
        - feature_counts: Array of feature counts evaluated.
        - accuracies: Array of corresponding accuracies.
    """
    from sklearn.model_selection import cross_val_score
    from src.knn_model import build_knn_classifier

    n_max = len(ranked_indices)
    # Evaluate at each count from 1 to n_max
    # For efficiency, use steps for large feature sets
    if n_max <= 50:
        eval_counts = np.arange(1, n_max + 1)
    else:
        # Evaluate at every feature, but group for speed
        eval_counts = np.concatenate([
            np.arange(1, min(50, n_max) + 1),
            np.arange(50, n_max + 1, 5)
        ])
        eval_counts = np.unique(eval_counts)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    accuracies = np.zeros(len(eval_counts))

    logger.info(f"Evaluating mRMR feature counts: 1 to {n_max}")

    for i, n in enumerate(eval_counts):
        selected = ranked_indices[:n]
        X_selected = X[:, selected]

        # Build KNN with paper-specified params
        knn = build_knn_classifier(knn_params)

        # Cross-validate
        scores = cross_val_score(knn, X_selected, y, cv=skf,
                                  scoring='accuracy', n_jobs=-1)
        accuracies[i] = np.mean(scores)

        if i % 20 == 0:
            logger.debug(f"  n_features={n}: accuracy={accuracies[i]:.4f}")

    # Select: maximum accuracy with minimum features
    max_acc = np.max(accuracies)
    # Find first occurrence of max accuracy (minimum features achieving it)
    optimal_idx = np.argmax(accuracies >= max_acc)
    optimal_n = int(eval_counts[optimal_idx])

    logger.info(
        f"mRMR optimal: n_features={optimal_n}, "
        f"accuracy={max_acc*100:.2f}%"
    )

    return optimal_n, eval_counts, accuracies


def apply_mrmr_pipeline(X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_test: np.ndarray,
                          n_features_target: int,
                          method: str = 'MI',
                          apply_globally: bool = False
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full mRMR pipeline: rank and select features.

    IMPORTANT: mRMR is fitted on training data ONLY (X_train).
    Test data (X_test) is transformed using the same ranking.
    This prevents data leakage.

    [Paper does not specify whether mRMR was applied globally or per-fold]
    [Implementation choice: per-fold (methodologically strict)]

    Args:
        X_train: Training features (n_train, n_total).
        y_train: Training labels.
        X_test: Test features (n_test, n_total).
        n_features_target: Number of features to select.
        method: mRMR method ('MI' or 'F').
        apply_globally: If True, rank is from pre-computed global ranking.

    Returns:
        Tuple of:
        - X_train_selected: Training features, shape (n_train, n_features_target)
        - X_test_selected: Test features, shape (n_test, n_features_target)
        - ranked_indices: The feature ranking indices
    """
    n_total = X_train.shape[1]

    if n_features_target >= n_total:
        # Selecting EVERY feature: the ranking cannot change which columns are
        # kept, and KNN is invariant to column order, so the expensive
        # n_total x n_total mutual-information matrix would be pure waste.
        # This is the DS-2 case (paper keeps all 171 features).
        logger.info(
            f"mRMR skipped: n_features_target={n_features_target} >= "
            f"n_total={n_total}, so all features are retained."
        )
        ranked_indices = np.arange(n_total)
    else:
        # Rank features on training data
        ranked_indices = compute_mrmr_ranking(
            X=X_train,
            y=y_train,
            n_features=n_features_target,
            method=method
        )

    # Select top-n features
    selected = ranked_indices[:n_features_target]

    X_train_selected = X_train[:, selected]
    X_test_selected = X_test[:, selected]

    return X_train_selected, X_test_selected, ranked_indices


# ─────────────────────────────────────────────────────────────────────────────
# Global mRMR (for paper-style reproduction where mRMR may be global)
# ─────────────────────────────────────────────────────────────────────────────

def global_mrmr_ranking(X: np.ndarray,
                          y: np.ndarray,
                          n_features_target: int,
                          method: str = 'MI') -> np.ndarray:
    """Compute mRMR ranking on entire dataset (global, potentially leaky).

    NOTE: This approach applies mRMR before cross-validation splits,
    which is technically data leakage. However, it may match the paper's
    MATLAB workflow where mRMR was likely applied globally.

    [Paper does not specify whether mRMR is global or per-fold]
    [This is the LEAKY version; use apply_mrmr_pipeline for strict version]

    Args:
        X: All features (n_samples, n_total).
        y: All labels.
        n_features_target: Target feature count.
        method: mRMR method.

    Returns:
        Ranked feature indices (all features ranked, use first n for selection).
    """
    logger.warning(
        "global_mrmr_ranking: Computing mRMR on entire dataset. "
        "This is potentially data-leaky. Intended for paper-style reproduction. "
        "See reproducibility_notes.md for details."
    )
    return compute_mrmr_ranking(X, y, n_features=n_features_target, method=method)
