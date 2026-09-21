"""
results_io.py - Persistent result store for the SMVMD reproduction.

Each experiment is run MANUALLY and independently:

    python -m experiments.ds1_rest
    python -m experiments.ds1_music
    python -m experiments.ds1_rest_music
    python -m experiments.ds2_adhd

Every run drops a self-contained bundle on disk, and nothing a run writes
clobbers another run's output. Once the runs you want are on disk, build the
consolidated write-up with:

    python -m experiments.make_report

Layout produced
---------------
results/
  metrics/<exp>.json              Every metric, both CV protocols, the KNN
                                  hyperparameters used, paper targets and the
                                  deltas against them, plus run provenance.
  features/<exp>_features.npz     Cached feature matrix X, labels, subject
                                  groups. Lets you redo classification and
                                  reporting WITHOUT redoing SMVMD.
  confusion_matrices/cm_<exp>.png
  plots/<exp>_mrmr_accuracy.png   Accuracy vs. number of mRMR features (Fig. 5).
  tables/<exp>_*.csv              Report-ready tables.
  final_report.md                 Written by experiments/make_report.py.

Feature cache invalidation
--------------------------
The cache stores a fingerprint of every parameter that affects X (SMVMD alpha
/ tau / tol, window, overlap, preprocessing mode, sampling rate). If you change
any of them in configs/config.yaml, the fingerprint stops matching and the
cache is refused, so a stale matrix can never silently end up in your report.
"""

import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RESULTS_ROOT = 'results'
METRICS_DIR = os.path.join(RESULTS_ROOT, 'metrics')
FEATURES_DIR = os.path.join(RESULTS_ROOT, 'features')
CM_DIR = os.path.join(RESULTS_ROOT, 'confusion_matrices')
PLOTS_DIR = os.path.join(RESULTS_ROOT, 'plots')
TABLES_DIR = os.path.join(RESULTS_ROOT, 'tables')

# Human-readable titles, and the order sections appear in the final report.
EXPERIMENT_TITLES = {
    'DS1_Rest': 'DS-1: IDD vs TDC (rest state)',
    'DS1_Music': 'DS-1: IDD vs TDC (music stimulus)',
    'DS1_RestMusic': 'DS-1: IDD vs TDC (rest + music combined)',
    'DS2': 'DS-2: ADHD vs NC (visual attention task)',
}
EXPERIMENT_ORDER = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes into."""
    for d in (METRICS_DIR, FEATURES_DIR, CM_DIR, PLOTS_DIR, TABLES_DIR):
        os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy types/arrays into JSON-serialisable values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Feature cache (skip SMVMD on re-runs)
# ─────────────────────────────────────────────────────────────────────────────

def feature_fingerprint(config: Dict[str, Any], ds_key: str,
                        condition: str = '') -> str:
    """Fingerprint of every config value that changes the feature matrix.

    Args:
        config: The full loaded config dict.
        ds_key: 'ds1' or 'ds2'.
        condition: Free-form discriminator, e.g. 'rest', 'music', 'rest+music'.

    Returns:
        A short stable string. Changing SMVMD or segmentation parameters
        changes it, which invalidates any cached matrix.
    """
    ds = config[ds_key]
    seg = config['segmentation']
    smvmd = ds['smvmd']
    parts = [
        f"cond={condition}",
        f"fs={ds['sampling_freq']}",
        f"alpha={smvmd['alpha']}",
        f"tau={smvmd['tau']}",
        f"tol={smvmd['tol']}",
        f"max_iter={smvmd.get('max_iter', 500)}",
        f"win={seg['window_sec']}",
        f"ovl={seg['overlap']}",
    ]
    if ds_key == 'ds1':
        parts.append(f"mode={ds['preprocessing']['mode']}")
    else:
        pre = ds['preprocessing']
        parts.append(f"bp={pre['bandpass_low']}-{pre['bandpass_high']}")
        parts.append(f"wav={pre['wavelet']}L{pre['wavelet_level']}")
    return '|'.join(str(p) for p in parts)


def save_feature_cache(exp_key: str,
                       X: np.ndarray,
                       y: np.ndarray,
                       groups: np.ndarray,
                       fingerprint: str,
                       n_modes_per_segment: Optional[List[int]] = None) -> str:
    """Persist the feature matrix so later runs can skip SMVMD entirely."""
    ensure_dirs()
    path = os.path.join(FEATURES_DIR, f'{exp_key}_features.npz')
    np.savez_compressed(
        path,
        X=X,
        y=y,
        groups=np.asarray(groups, dtype=object).astype(str),
        fingerprint=np.array(fingerprint),
        n_modes=np.array(n_modes_per_segment if n_modes_per_segment else [],
                         dtype=np.int32),
        saved_at=np.array(datetime.now().isoformat(timespec='seconds')),
    )
    logger.info(f"Feature cache saved -> {path}  (X={X.shape})")
    return path


def load_feature_cache(exp_key: str,
                       fingerprint: str
                       ) -> Optional[Tuple[np.ndarray, np.ndarray,
                                           np.ndarray, np.ndarray]]:
    """Load a cached feature matrix, but only if the fingerprint still matches.

    Returns:
        (X, y, groups, n_modes) or None when there is no usable cache.
    """
    path = os.path.join(FEATURES_DIR, f'{exp_key}_features.npz')
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        cached_fp = str(data['fingerprint'])
    except Exception as e:  # corrupt or written by an older version
        logger.warning(f"Could not read feature cache {path}: {e}. Recomputing.")
        return None

    if cached_fp != fingerprint:
        logger.warning(
            "Feature cache IGNORED: pipeline parameters changed since it was "
            "written.\n"
            f"  cached : {cached_fp}\n"
            f"  current: {fingerprint}\n"
            "Recomputing from raw EEG."
        )
        return None

    X, y = data['X'], data['y']
    groups = data['groups']
    n_modes = data['n_modes']
    logger.info(
        f"Feature cache HIT -> {path}  (X={X.shape}, saved "
        f"{str(data['saved_at'])}). Skipping preprocessing/SMVMD/features."
    )
    return X, y, groups, n_modes


# ─────────────────────────────────────────────────────────────────────────────
# Per-experiment result bundle
# ─────────────────────────────────────────────────────────────────────────────

def save_experiment(exp_key: str,
                    metrics: Dict[str, Any],
                    knn_params: Dict[str, Any],
                    n_features_used: int,
                    n_features_total: int,
                    n_segments: int,
                    n_subjects: int,
                    comparison: Optional[Dict[str, Any]] = None,
                    subject_independent: Optional[Dict[str, Any]] = None,
                    fold_accuracies: Optional[List[float]] = None,
                    classifier_comparison: Optional[Dict[str, Any]] = None,
                    mrmr_curve: Optional[Dict[str, Any]] = None,
                    mode_stats: Optional[Dict[str, Any]] = None,
                    runtime_sec: Optional[float] = None,
                    extra: Optional[Dict[str, Any]] = None) -> str:
    """Write one experiment's complete result bundle to results/metrics/.

    Safe to call from separate manual runs: each experiment owns its own file.

    Returns:
        Path to the written JSON.
    """
    ensure_dirs()

    payload: Dict[str, Any] = {
        'experiment': exp_key,
        'title': EXPERIMENT_TITLES.get(exp_key, exp_key),
        'run_at': datetime.now().isoformat(timespec='seconds'),
        'runtime_sec': runtime_sec,
        'dataset': {
            'n_segments': int(n_segments),
            'n_subjects': int(n_subjects),
            'n_features_total': int(n_features_total),
            'n_features_used': int(n_features_used),
        },
        'knn_params': knn_params,
        'metrics_paper_protocol': {k: v for k, v in metrics.items()
                                   if k != 'class_names'},
        'class_names': metrics.get('class_names'),
        'paper_comparison': comparison,
        'fold_accuracies': fold_accuracies,
        'subject_independent': subject_independent,
        'classifier_comparison': classifier_comparison,
        'mrmr_curve': mrmr_curve,
        'mode_stats': mode_stats,
        'provenance': {
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'platform': platform.platform(),
        },
    }
    if extra:
        payload['extra'] = extra

    path = os.path.join(METRICS_DIR, f'{exp_key}.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_jsonable(payload), f, indent=2)
    logger.info(f"Results saved -> {path}")
    return path


def load_all_experiments() -> Dict[str, Dict[str, Any]]:
    """Load every saved experiment bundle, newest content as found on disk."""
    if not os.path.isdir(METRICS_DIR):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(os.listdir(METRICS_DIR)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(METRICS_DIR, name)
        try:
            with open(path, encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable result file {path}: {e}")
            continue
        key = data.get('experiment', os.path.splitext(name)[0])
        out[key] = data
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CSV tables (drop straight into the report)
# ─────────────────────────────────────────────────────────────────────────────

def save_table(exp_key: str, table_name: str,
               rows: List[Dict[str, Any]]) -> Optional[str]:
    """Write a list of uniform dicts as a CSV under results/tables/."""
    if not rows:
        return None
    ensure_dirs()
    import csv
    path = os.path.join(TABLES_DIR, f'{exp_key}_{table_name}.csv')
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: _jsonable(r.get(k)) for k in fieldnames})
    logger.info(f"Table saved -> {path}")
    return path


class Stopwatch:
    """Tiny context manager for recording stage runtimes into a dict."""

    def __init__(self, store: Dict[str, float], key: str):
        self.store = store
        self.key = key

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.store[self.key] = round(time.perf_counter() - self._t0, 2)
        logger.info(f"[timing] {self.key}: {self.store[self.key]:.2f}s")
        return False
