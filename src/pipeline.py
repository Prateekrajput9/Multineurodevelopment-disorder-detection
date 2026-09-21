"""
pipeline.py - Shared DS-1 experiment driver.

The three DS-1 experiments (rest, music, rest+music) differ only in which
recording conditions they load and which row of paper Table II supplies the
KNN hyperparameters. Everything after segmentation is identical, so it lives
here once. That keeps the three entry points from drifting apart (they had
already drifted: only one of them saved its results, and one looked up its
paper targets under a key that did not exist).

Each entry point still runs standalone and saves its own complete bundle:

    python -m experiments.ds1_rest
    python -m experiments.ds1_music
    python -m experiments.ds1_rest_music
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.data_loader import DS1Loader
from src.energy_integration import integrate_all_segments, verify_feature_matrix
from src.evaluation import (compare_with_paper_targets, compute_all_metrics,
                            plot_confusion_matrix_paper, print_results_table)
from src.features import extract_features_batch
from src.knn_model import (evaluate_knn_paper_style,
                           evaluate_knn_subject_independent)
from src.mrmr import compute_mrmr_ranking, select_features_by_accuracy
from src.preprocessing import preprocess_ds1
from src.results_io import (CM_DIR, PLOTS_DIR, Stopwatch, feature_fingerprint,
                            load_feature_cache, save_experiment,
                            save_feature_cache, save_table)
from src.segmentation import segment_eeg
from src.smvmd import decompose_segments
from src.utils import plot_accuracy_vs_features

DS1_N_CHANNELS = 14
DS1_CLASS_NAMES = ['TDC (0)', 'IDD (1)']


def build_ds1_features(config: Dict[str, Any],
                       conditions: List[str],
                       exp_key: str,
                       logger: logging.Logger,
                       timings: Dict[str, float],
                       use_cache: bool = True):
    """Load -> preprocess -> segment -> SMVMD -> features -> energy integration.

    Returns:
        (X, y, groups, n_modes) where X is (n_segments, 126).
    """
    ds1_cfg = config['ds1']
    fs = ds1_cfg['sampling_freq']
    mode = ds1_cfg['preprocessing']['mode']
    smvmd_cfg = ds1_cfg['smvmd']

    fingerprint = feature_fingerprint(config, 'ds1', '+'.join(conditions))

    if use_cache:
        cached = load_feature_cache(exp_key, fingerprint)
        if cached is not None:
            return cached

    loader = DS1Loader(config['paths']['ds1_data_root'], mode=mode)
    logger.info(loader.report())

    files = loader.discover_files()
    wanted = {
        subj: {c: paths[c] for c in conditions if c in paths}
        for subj, paths in files.items()
    }
    wanted = {s: c for s, c in wanted.items() if c}
    if not wanted:
        raise RuntimeError(
            f"No DS-1 recordings found for conditions={conditions}. "
            f"Check paths.ds1_data_root in configs/config.yaml."
        )

    all_segments, all_labels, all_groups = [], [], []
    with Stopwatch(timings, 'load_preprocess_segment'):
        for subject_id, cond_files in sorted(wanted.items()):
            for condition, path in sorted(cond_files.items()):
                eeg, info = loader.load_subject(subject_id, condition, path)
                if eeg is None:
                    logger.warning(f"Skipping {subject_id}/{condition}: load failed.")
                    continue

                eeg = preprocess_ds1(eeg, fs=fs, mode=mode,
                                     config=ds1_cfg['preprocessing'])
                segs, _meta = segment_eeg(
                    eeg=eeg, fs=fs,
                    window_sec=config['segmentation']['window_sec'],
                    overlap=config['segmentation']['overlap'],
                    subject_id=subject_id, label=info['label'],
                    dataset='DS1', condition=condition,
                )
                all_segments.append(segs)
                all_labels.extend([info['label']] * len(segs))
                all_groups.extend([subject_id] * len(segs))

    if not all_segments:
        raise RuntimeError("No DS-1 segments produced.")

    segments = np.concatenate(all_segments, axis=0)
    y = np.array(all_labels)
    groups = np.array(all_groups)

    expected = ds1_cfg['expected_segments']['total'] if len(conditions) > 1 \
        else ds1_cfg['expected_segments']['total'] // 2
    logger.info(f"Total segments: {len(segments)} (paper: {expected})")

    with Stopwatch(timings, 'smvmd'):
        smvmd_results = decompose_segments(
            segments=segments, alpha=smvmd_cfg['alpha'], tau=smvmd_cfg['tau'],
            tol=smvmd_cfg['tol'], max_iter=smvmd_cfg.get('max_iter', 500), fs=fs,
        )

    with Stopwatch(timings, 'features'):
        feature_lists = extract_features_batch(smvmd_results, fs=fs)

    with Stopwatch(timings, 'energy_integration'):
        X = integrate_all_segments(smvmd_results, feature_lists,
                                   n_channels=DS1_N_CHANNELS, dataset='DS1')
    verify_feature_matrix(X, exp_key)

    n_modes = np.array([r.n_modes for r in smvmd_results], dtype=np.int32)
    save_feature_cache(exp_key, X, y, groups, fingerprint, n_modes.tolist())
    return X, y, groups, n_modes


def run_ds1_experiment(config: Dict[str, Any],
                       conditions: List[str],
                       exp_key: str,
                       knn_key: str,
                       n_features_key: str,
                       title: str,
                       logger: logging.Logger,
                       use_cache: bool = True) -> Dict[str, Any]:
    """Run one complete DS-1 experiment and save every artefact it produces."""
    t_start = time.perf_counter()
    timings: Dict[str, float] = {}

    ds1_cfg = config['ds1']
    X, y, groups, n_modes = build_ds1_features(
        config, conditions, exp_key, logger, timings, use_cache=use_cache)

    n_features = ds1_cfg['mrmr'][n_features_key]
    knn_params = dict(ds1_cfg['knn'][knn_key])

    logger.info(
        f"\n{'='*60}\n{title}\n"
        f"  segments={len(X)}  features={X.shape[1]} -> mRMR keeps {n_features}\n"
        f"  KNN (paper Table II): {knn_params}\n{'='*60}"
    )

    # ── Accuracy vs. number of mRMR features (paper Fig. 5) ─────────────────
    mrmr_curve = None
    with Stopwatch(timings, 'mrmr_curve'):
        try:
            ranked_idx = compute_mrmr_ranking(X, y, n_features=X.shape[1])
            n_opt, feat_counts, accs = select_features_by_accuracy(
                X=X, y=y, ranked_indices=ranked_idx, knn_params=knn_params,
                cv_folds=config['evaluation']['paper_style']['n_splits'],
                random_seed=config.get('random_seed', 42),
            )
            plot_path = os.path.join(PLOTS_DIR, f'{exp_key}_mrmr_accuracy.png')
            plot_accuracy_vs_features(
                n_features_list=feat_counts.tolist(), accuracies=accs.tolist(),
                optimal_n=n_opt, dataset_name=title, save_path=plot_path,
            )
            mrmr_curve = {
                'feature_counts': feat_counts.tolist(),
                'accuracies': accs.tolist(),
                'optimal_n_empirical': int(n_opt),
                'n_features_paper': int(n_features),
                'ranked_feature_indices': ranked_idx.tolist(),
                'plot': plot_path,
            }
            save_table(exp_key, 'mrmr_curve', [
                {'n_features': int(n), 'cv_accuracy': float(a)}
                for n, a in zip(feat_counts, accs)
            ])
            logger.info(
                f"mRMR curve: empirical optimum={n_opt} features "
                f"(paper states {n_features} for this experiment)"
            )
        except Exception as e:
            logger.warning(f"mRMR curve step failed ({e}); continuing.")

    # ── Protocol A: paper-style 10-fold stratified CV ───────────────────────
    with Stopwatch(timings, 'cv_paper_protocol'):
        results = evaluate_knn_paper_style(
            X=X, y=y, knn_params=knn_params, n_features_target=n_features,
            cv_folds=config['evaluation']['paper_style']['n_splits'],
            random_seed=config.get('random_seed', 42), dataset_name=exp_key,
        )

    metrics = compute_all_metrics(results['y_true'], results['y_pred'],
                                  dataset='DS1', class_names=DS1_CLASS_NAMES)
    comparison = compare_with_paper_targets(metrics, exp_key)
    print_results_table(metrics, title, comparison)

    cm_path = os.path.join(CM_DIR, f'cm_{exp_key}.png')
    plot_confusion_matrix_paper(cm=metrics['confusion_matrix'],
                                class_names=DS1_CLASS_NAMES,
                                title=f'Confusion Matrix - {title}',
                                save_path=cm_path)

    # ── Protocol B: subject-independent audit (not a paper claim) ───────────
    si_summary = None
    with Stopwatch(timings, 'cv_subject_independent'):
        try:
            results_si = evaluate_knn_subject_independent(
                X=X, y=y, groups=groups, knn_params=knn_params,
                n_features_target=n_features, dataset_name=exp_key,
            )
            si_summary = {k: v for k, v in results_si.items()
                          if k not in ('y_true', 'y_pred')}
        except Exception as e:
            logger.warning(f"Subject-independent audit failed ({e}); continuing.")

    # ── Persist everything ──────────────────────────────────────────────────
    timings['total'] = round(time.perf_counter() - t_start, 2)
    # `max_modes` is a safety cap in SMVMD, not part of the paper. Segments that
    # reach it had their mode count decided by that cap rather than by the
    # residual-energy stopping criterion, so the share matters methodologically.
    mode_cap = 20
    mode_stats = {
        'mean_modes_per_segment': float(np.mean(n_modes)) if len(n_modes) else None,
        'min_modes': int(np.min(n_modes)) if len(n_modes) else None,
        'max_modes': int(np.max(n_modes)) if len(n_modes) else None,
        'mode_cap': mode_cap,
        'n_segments_at_mode_cap': int(np.sum(np.asarray(n_modes) >= mode_cap))
        if len(n_modes) else None,
        'pct_segments_at_mode_cap': float(
            np.mean(np.asarray(n_modes) >= mode_cap) * 100) if len(n_modes) else None,
    }

    save_experiment(
        exp_key=exp_key,
        metrics=metrics,
        knn_params=knn_params,
        n_features_used=n_features,
        n_features_total=int(X.shape[1]),
        n_segments=int(len(X)),
        n_subjects=int(len(np.unique(groups))),
        comparison=comparison,
        subject_independent=si_summary,
        fold_accuracies=results.get('fold_accuracies'),
        mrmr_curve=mrmr_curve,
        mode_stats=mode_stats,
        runtime_sec=timings['total'],
        extra={'timings_sec': timings, 'conditions': conditions,
               'confusion_matrix_plot': cm_path},
    )

    save_table(exp_key, 'fold_accuracies', [
        {'fold': i + 1, 'accuracy': float(a)}
        for i, a in enumerate(results.get('fold_accuracies', []))
    ])

    logger.info(f"{title} complete in {timings['total']:.1f}s")
    return metrics
