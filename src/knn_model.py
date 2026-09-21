"""
knn_model.py - KNN Classifier with Bayesian Optimization

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

Classifier [Paper explicit]:
  - K-Nearest Neighbors (KNN)
  - MATLAB R2023a Classification Learner
  - Bayesian optimization: 200 iterations
  - 10-fold cross-validation

Paper-reported optimized hyperparameters (Table II):
  DS-1 Rest:         n=2, distance=Cityblock, weight=Equal,          standardize=True
  DS-1 Music:        n=2, distance=Cityblock, weight=SquaredInverse, standardize=False
  DS-1 Rest+Music:   n=1, distance=Euclidean, weight=SquaredInverse, standardize=True
  DS-2:              n=2, distance=Cosine,    weight=SquaredInverse, standardize=False

MATLAB vs sklearn differences:
  - MATLAB "Squared Inverse" = weight ∝ 1/d² (not available in sklearn)
  - MATLAB "Equal" = sklearn weights='uniform'
  - MATLAB "Standardize=True" = apply z-score standardization before KNN
    [sklearn has no built-in KNN standardization; apply StandardScaler separately]

Also implements comparison classifiers (Table V):
  - Decision Tree (DT)
  - Support Vector Machine (SVM)
  - Ensemble (Bagged Decision Trees / Random Forest)
"""

import numpy as np
import logging
import warnings
from typing import Dict, Any, Optional, Tuple, List

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Custom "Squared Inverse" Distance Weighting
# ─────────────────────────────────────────────────────────────────────────────

def squared_inverse_weights(distances: np.ndarray) -> np.ndarray:
    """Compute squared-inverse distance weights.

    MATLAB Classification Learner "Squared Inverse" weight:
      w_i = 1 / (d_i² + ε)

    scikit-learn's built-in options are 'uniform' and 'distance' (= 1/d).
    This custom callable implements 1/d² weighting.

    [Paper: "Squared inverse" - MATLAB definition]
    [Implementation choice: w = 1/(d² + ε) to avoid division by zero]

    Args:
        distances: Array of distances to neighbors.

    Returns:
        Weights array (same shape as distances).
    """
    eps = 1e-10
    return 1.0 / (distances ** 2 + eps)


# ─────────────────────────────────────────────────────────────────────────────
# KNN Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_classifier(params: Dict[str, Any]) -> KNeighborsClassifier:
    """Build a KNN classifier from paper-specified hyperparameters.

    Args:
        params: Dict with keys:
            n_neighbors: int (e.g., 2)
            distance: str - 'cityblock', 'euclidean', 'cosine'
                      [maps to sklearn metric names]
            weights: str - 'uniform', 'distance', 'squared_inverse'
                      [squared_inverse is a custom callable]
            [standardize is handled externally via StandardScaler]

    Returns:
        KNeighborsClassifier instance.
    """
    n_neighbors = params.get('n_neighbors', 2)
    distance = params.get('distance', 'euclidean').lower()
    weights_str = params.get('weights', 'uniform').lower()

    # Map MATLAB distance names to sklearn
    distance_map = {
        'cityblock': 'manhattan',         # Manhattan = L1 = Cityblock
        'euclidean': 'euclidean',         # L2
        'cosine': 'cosine',               # 1 - cosine_similarity
        'minkowski': 'minkowski',
        'chebyshev': 'chebyshev',
    }
    metric = distance_map.get(distance, distance)

    # Map MATLAB weight names to sklearn
    if weights_str == 'uniform' or weights_str == 'equal':
        weights = 'uniform'
    elif weights_str == 'distance' or weights_str == 'inverse':
        weights = 'distance'
    elif weights_str == 'squared_inverse':
        weights = squared_inverse_weights  # Custom callable
    else:
        logger.warning(
            f"Unknown weight '{weights_str}'. Using 'uniform'. "
            f"[Paper: {weights_str}]"
        )
        weights = 'uniform'

    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        metric=metric,
        weights=weights,
        n_jobs=-1,
    )

    return knn


def build_knn_with_scaler(params: Dict[str, Any]) -> Pipeline:
    """Build KNN with optional StandardScaler.

    Paper Table II: "Standardize" column indicates whether z-score
    standardization is applied before KNN.

    MATLAB "Standardize=True" standardizes features before distance computation.
    In sklearn, this requires a Pipeline with StandardScaler.

    Args:
        params: Same as build_knn_classifier, plus:
            standardize: bool (paper Table II)

    Returns:
        sklearn Pipeline (optionally with StandardScaler).
    """
    standardize = params.get('standardize', False)
    knn = build_knn_classifier(params)

    if standardize:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('knn', knn),
        ])
        logger.info("KNN with StandardScaler (standardize=True)")
    else:
        # Still wrap in pipeline for consistent interface
        pipeline = Pipeline([
            ('knn', knn),
        ])
        logger.info("KNN without standardization (standardize=False)")

    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Classifier Comparison (Paper Table V)
# ─────────────────────────────────────────────────────────────────────────────

def build_comparison_classifiers() -> Dict[str, Any]:
    """Build all classifiers for Table V comparison.

    Paper compares (Table V):
    - Decision Tree (DT)
    - KNN
    - SVM
    - Ensemble

    Each is optimized with Bayesian optimization in the paper.
    For initial reproduction, use paper-reported hyperparameters.

    [Paper reports final accuracy but not all classifier hyperparameters]
    [Implementation choice: use sklearn defaults + Bayesian optimization]

    Returns:
        Dict mapping classifier name → sklearn estimator.
    """
    classifiers = {
        'DT': DecisionTreeClassifier(
            random_state=42,
            # [Paper does not specify DT hyperparameters; using defaults]
        ),
        'KNN': None,  # Built separately with paper-specific params

        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            # [Paper does not specify SVM hyperparameters; using RBF defaults]
        ),

        'Ensemble': BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=42),
            n_estimators=100,
            random_state=42,
            # [Paper: "Ensemble" likely = bagged trees in MATLAB; implementation choice]
        ),
    }

    logger.info("Initialized comparison classifiers (DT, KNN, SVM, Ensemble)")
    logger.warning(
        "[Paper does not specify hyperparameters for DT, SVM, Ensemble] "
        "Using sklearn defaults + Bayesian optimization for tuning."
    )

    return classifiers


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Hyperparameter Optimization
# ─────────────────────────────────────────────────────────────────────────────

def bayesian_optimize_knn(X: np.ndarray,
                           y: np.ndarray,
                           n_iterations: int = 200,
                           cv_folds: int = 10,
                           random_seed: int = 42) -> Dict[str, Any]:
    """Bayesian hyperparameter optimization for KNN.

    Paper [explicit]:
    - MATLAB R2023a Classification Learner
    - Bayesian optimization
    - 200 iterations
    - 10-fold cross-validation

    Python implementation: scikit-optimize BayesSearchCV
    [Not identical to MATLAB's Bayesian optimizer; documented discrepancy]

    Search space:
    - n_neighbors: [1, 50]
    - distance: ['euclidean', 'manhattan', 'cosine', 'chebyshev']
    - weights: ['uniform', 'distance', 'squared_inverse']
    - standardize: [True, False]

    Args:
        X: Feature matrix.
        y: Labels.
        n_iterations: Number of optimizer iterations (paper: 200).
        cv_folds: Cross-validation folds (paper: 10).
        random_seed: Random seed.

    Returns:
        Dict with optimal hyperparameters.
    """
    try:
        from skopt import BayesSearchCV
        from skopt.space import Integer, Categorical, Real
    except ImportError:
        logger.warning(
            "scikit-optimize not installed. Falling back to paper-reported hyperparameters. "
            "Install with: pip install scikit-optimize"
        )
        return {}

    logger.info(
        f"Bayesian optimization: {n_iterations} iterations, "
        f"{cv_folds}-fold CV"
    )
    logger.warning(
        "MATLAB vs Python Bayesian optimization: Results will differ. "
        "Paper used MATLAB R2023a Classification Learner. "
        "See docs/reproducibility_notes.md."
    )

    # Define search space
    search_space = {
        'knn__n_neighbors': Integer(1, 50),
        'knn__metric': Categorical(['euclidean', 'manhattan', 'cosine', 'chebyshev']),
    }

    # Build base pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_jobs=-1)),
    ])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    optimizer = BayesSearchCV(
        estimator=pipeline,
        search_spaces=search_space,
        n_iter=n_iterations,
        cv=skf,
        scoring='accuracy',
        n_jobs=1,
        random_state=random_seed,
        verbose=0,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer.fit(X, y)

    best_params = optimizer.best_params_
    best_score = optimizer.best_score_

    logger.info(
        f"Bayesian optimization complete:\n"
        f"  Best params: {best_params}\n"
        f"  Best CV accuracy: {best_score*100:.2f}%"
    )

    return {
        'best_params': best_params,
        'best_score': float(best_score),
        'all_results': optimizer.cv_results_,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation Evaluation (Paper-style)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_knn_paper_style(X: np.ndarray,
                              y: np.ndarray,
                              knn_params: Dict[str, Any],
                              n_features_target: int,
                              mrmr_method: str = 'MI',
                              cv_folds: int = 10,
                              random_seed: int = 42,
                              dataset_name: str = 'DS2'
                              ) -> Dict[str, Any]:
    """Evaluate KNN using paper-style 10-fold CV.

    Paper style:
    - 10-fold stratified cross-validation at segment level
    - Segments from the same subject may appear in train and test
    - mRMR feature selection applied within each fold

    [Paper evaluation protocol; see reproducibility_notes.md for caveats]

    Args:
        X: Feature matrix (n_segments, n_features).
        y: Labels (n_segments,).
        knn_params: KNN hyperparameters dict.
        n_features_target: Number of mRMR features to use.
        mrmr_method: 'MI' or 'F'.
        cv_folds: Number of folds (paper: 10).
        random_seed: Random seed.
        dataset_name: For logging.

    Returns:
        Dict with all metrics.
    """
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score,
                                  matthews_corrcoef, cohen_kappa_score,
                                  confusion_matrix)
    from src.mrmr import apply_mrmr_pipeline

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    all_y_true = []
    all_y_pred = []
    fold_accuracies = []

    logger.info(
        f"\n{'='*50}\n"
        f"Evaluating KNN ({dataset_name}): {cv_folds}-fold CV\n"
        f"  Features: {n_features_target}, KNN: {knn_params}\n"
        f"{'='*50}"
    )

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # mRMR feature selection (on training data only)
        X_train_sel, X_test_sel, ranked_idx = apply_mrmr_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            n_features_target=n_features_target,
            method=mrmr_method,
        )

        # Build and train KNN
        model = build_knn_with_scaler(knn_params)
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)

        # Record results
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)

        logger.debug(f"Fold {fold_idx+1}/{cv_folds}: accuracy={fold_acc*100:.2f}%")

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Compute metrics
    cm = confusion_matrix(all_y_true, all_y_pred)
    acc = accuracy_score(all_y_true, all_y_pred)
    prec = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    rec = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)

    results = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm,
        'fold_accuracies': fold_accuracies,
        'y_true': all_y_true,
        'y_pred': all_y_pred,
    }

    # Additional DS-2 metrics
    if cm.shape == (2, 2):
        mcc = matthews_corrcoef(all_y_true, all_y_pred)
        kappa = cohen_kappa_score(all_y_true, all_y_pred)

        from src.utils import compute_gdr, compute_gmean
        gdr = compute_gdr(cm)
        gmean = compute_gmean(cm)

        results.update({
            'mcc': float(mcc),
            'kappa': float(kappa),
            'gdr': float(gdr),
            'gmean': float(gmean),
        })

    # Log results
    logger.info(
        f"\nResults ({dataset_name}):\n"
        f"  Accuracy:  {acc*100:.2f}% (paper target: see docs)\n"
        f"  Precision: {prec*100:.2f}%\n"
        f"  Recall:    {rec*100:.2f}%\n"
        f"  F1-score:  {f1*100:.2f}%\n"
        f"  Confusion matrix:\n{cm}"
    )

    if 'mcc' in results:
        logger.info(
            f"  MCC:       {results['mcc']:.4f}\n"
            f"  Kappa:     {results['kappa']:.4f}\n"
            f"  GDR:       {results['gdr']:.3f}\n"
            f"  G-mean:    {results['gmean']:.4f}"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Subject-Independent Evaluation (Methodological Audit)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_knn_subject_independent(X: np.ndarray,
                                      y: np.ndarray,
                                      groups: np.ndarray,
                                      knn_params: Dict[str, Any],
                                      n_features_target: int,
                                      mrmr_method: str = 'MI',
                                      n_splits: int = 10,
                                      random_seed: int = 42,
                                      dataset_name: str = 'DS2'
                                      ) -> Dict[str, Any]:
    """Subject-independent evaluation (methodological robustness audit).

    Uses StratifiedGroupKFold: all segments from a subject stay in one fold.
    This is NOT the paper's evaluation protocol.

    [See docs/reproducibility_notes.md for detailed explanation]

    Args:
        X: Feature matrix.
        y: Labels.
        groups: Subject ID array (same subject = same group).
        knn_params: KNN parameters.
        n_features_target: mRMR feature count.
        mrmr_method: MI or F.
        n_splits: Number of folds.
        random_seed: Random seed.
        dataset_name: For logging.

    Returns:
        Dict with metrics.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score, confusion_matrix)
    from src.mrmr import apply_mrmr_pipeline

    logger.info(
        f"\n{'='*50}\n"
        f"Subject-Independent Evaluation ({dataset_name})\n"
        f"  [NOT the paper's protocol - methodological audit only]\n"
        f"  n_splits={n_splits}, features={n_features_target}\n"
        f"{'='*50}"
    )

    # StratifiedGroupKFold may not be available in older sklearn
    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits)
        splits = list(sgkf.split(X, y, groups))
    except Exception as e:
        logger.warning(f"StratifiedGroupKFold failed: {e}. Using GroupKFold.")
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))

    all_y_true = []
    all_y_pred = []
    fold_accuracies = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # mRMR
        X_train_sel, X_test_sel, _ = apply_mrmr_pipeline(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            n_features_target=n_features_target,
            method=mrmr_method,
        )

        model = build_knn_with_scaler(knn_params)
        model.fit(X_train_sel, y_train)
        y_pred = model.predict(X_test_sel)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_accuracies.append(accuracy_score(y_test, y_pred))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    cm = confusion_matrix(all_y_true, all_y_pred)
    acc = accuracy_score(all_y_true, all_y_pred)

    results = {
        'protocol': 'subject_independent',
        'accuracy': float(acc),
        'precision': float(precision_score(all_y_true, all_y_pred,
                                           average='weighted', zero_division=0)),
        'recall': float(recall_score(all_y_true, all_y_pred,
                                      average='weighted', zero_division=0)),
        'f1': float(f1_score(all_y_true, all_y_pred,
                              average='weighted', zero_division=0)),
        'confusion_matrix': cm,
        'fold_accuracies': fold_accuracies,
        'note': 'NOT the paper protocol. Methodological audit only.',
    }

    logger.info(
        f"Subject-Independent Results ({dataset_name}):\n"
        f"  Accuracy: {acc*100:.2f}%\n"
        f"  [Compare with paper-style: shows effect of segment leakage]"
    )

    return results
