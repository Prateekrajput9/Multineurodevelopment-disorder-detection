"""
ds2_adhd.py - Complete DS-2 (ADHD) Experiment Pipeline

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

DS-2 Pipeline:
  Raw EEG (19 ch, 128 Hz) → DS-2 Preprocessing → Segmentation (640 samples, 50% overlap)
  → SMVMD (α=1000) → 9 features per MIMF per channel → Energy integration (171 features)
  → mRMR (171 selected) → KNN (n=2, cosine, squared inverse, no standardize)
  → 10-fold CV → Evaluation

Paper targets:
  Accuracy = 99.17%, Precision = 99.04%, Recall = 99.07%, F1 = 99.05%
  MCC = 0.983, Cohen's Kappa = 0.983, GDR ≈ 99.239, G-mean ≈ 0.9915

Staged workflow (configurable via pipeline config):
  Phase 1: Load one subject
  Phase 2: Process one segment
  Phase 3: Verify SMVMD
  Phase 4: Verify features
  Phase 5: Verify energy integration
  Phase 6: Process one subject
  Phase 7: Process 5 subjects
  Phase 8: Full dataset
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, setup_logger, set_random_seed
from src.data_loader import DS2Loader
from src.preprocessing import preprocess_ds2
from src.segmentation import (segment_eeg, segment_dataset_ds2,
                                verify_all_segments)
from src.smvmd import SMVMD, SMVMDResult, decompose_segments, validate_smvmd_result
from src.features import extract_features_from_segment, extract_features_batch
from src.energy_integration import (integrate_all_segments, energy_integrate_segment,
                                     verify_feature_matrix)
from src.mrmr import compute_mrmr_ranking, select_features_by_accuracy
from src.knn_model import (build_knn_with_scaler, evaluate_knn_paper_style,
                             evaluate_knn_subject_independent)
from src.evaluation import (compute_all_metrics, compare_with_paper_targets,
                              print_results_table, generate_full_report,
                              PAPER_TARGETS)
from src.utils import (plot_smvmd_decomposition, plot_accuracy_vs_features,
                        plot_confusion_matrix)
from src.results_io import (CM_DIR, PLOTS_DIR, Stopwatch, ensure_dirs,
                              feature_fingerprint, load_feature_cache,
                              save_experiment, save_feature_cache, save_table)

logger = setup_logger('DS2_Experiment', 'results/logs/ds2_adhd.log')


def run_phase1_load_one_subject(config: dict, loader: DS2Loader) -> dict:
    """Phase 1: Load one subject and verify."""
    logger.info("\n" + "="*60 + "\nPHASE 1: Load One Subject\n" + "="*60)

    files = loader.discover_files()
    if not files:
        logger.error("PHASE 1 FAILED: No files discovered")
        return {}

    first_id = sorted(files.keys())[0]
    eeg, info = loader.load_subject(first_id)

    if eeg is None:
        logger.error(f"PHASE 1 FAILED: Could not load {first_id}")
        return {}

    logger.info(f"PHASE 1 SUCCESS:")
    logger.info(f"  Subject: {first_id}")
    logger.info(f"  Shape: {eeg.shape}")
    logger.info(f"  Sampling rate: {info['fs']} Hz")
    logger.info(f"  Duration: {info['duration_sec']:.1f} s")
    logger.info(f"  Label: {info['label']} ({info['label_str']})")

    return {'eeg': eeg, 'info': info, 'subject_id': first_id}


def run_phase2_one_segment(phase1_result: dict, config: dict) -> dict:
    """Phase 2: Process one 5-second segment."""
    logger.info("\n" + "="*60 + "\nPHASE 2: One 5-Second Segment\n" + "="*60)

    eeg = phase1_result['eeg']
    info = phase1_result['info']

    # Preprocess first
    ds2_cfg = config['ds2']['preprocessing']
    eeg_preprocessed = preprocess_ds2(
        eeg=eeg,
        fs=config['ds2']['sampling_freq'],
        bandpass_low=ds2_cfg['bandpass_low'],
        bandpass_high=ds2_cfg['bandpass_high'],
        butterworth_order=ds2_cfg['butterworth_order'],
        notch_freq=ds2_cfg['notch_freq'],
        wavelet=ds2_cfg['wavelet'],
        wavelet_level=ds2_cfg['wavelet_level'],
    )

    # Extract first segment (0 to 640 samples)
    fs = config['ds2']['sampling_freq']
    window_samples = config['segmentation']['window_samples']

    segment = eeg_preprocessed[:, :window_samples]
    logger.info(f"PHASE 2: segment shape={segment.shape}, expected=(19, 640)")

    from src.utils import verify_segment_shape
    shape_ok = verify_segment_shape(segment, n_channels=19, n_samples=640)

    logger.info(f"PHASE 2 {'SUCCESS' if shape_ok else 'FAILED'}")
    return {'segment': segment, 'eeg_preprocessed': eeg_preprocessed}


def run_phase3_verify_smvmd(phase2_result: dict, config: dict) -> dict:
    """Phase 3: SMVMD decomposition of one segment."""
    logger.info("\n" + "="*60 + "\nPHASE 3: SMVMD Decomposition\n" + "="*60)

    segment = phase2_result['segment']
    smvmd_cfg = config['ds2']['smvmd']
    fs = config['ds2']['sampling_freq']

    smvmd = SMVMD(
        alpha=smvmd_cfg['alpha'],    # 1000 for DS-2 [paper explicit]
        tau=smvmd_cfg['tau'],        # 0 [paper explicit]
        tol=smvmd_cfg['tol'],        # 1e-10 [paper explicit]
        max_iter=smvmd_cfg.get('max_iter', 500),  # [implementation choice]
    )

    logger.info(f"SMVMD parameters: alpha={smvmd_cfg['alpha']}, tau={smvmd_cfg['tau']}, "
                f"tol={smvmd_cfg['tol']}")

    start_time = time.time()
    result = smvmd.decompose(segment, fs=fs)
    elapsed = time.time() - start_time

    logger.info(f"PHASE 3 Results:")
    logger.info(f"  Number of modes (K): {result.n_modes}")
    logger.info(f"  Center frequencies: {[f'{f:.2f} Hz' for f in result.center_freqs_hz]}")
    logger.info(f"  Reconstruction error: {result.reconstruction_error:.2e}")
    logger.info(f"  Computation time: {elapsed:.2f}s")

    # Validate
    validation = validate_smvmd_result(result, segment, fs=fs)
    logger.info(f"  Reconstruction OK: {validation['reconstruction_ok']}")
    logger.info(f"  Center freqs valid: {validation['center_freqs_valid']}")

    # Save diagnostic plot
    if result.n_modes > 0:
        plot_path = f"results/smvmd/ds2_phase3_segment_ch0.png"
        os.makedirs("results/smvmd", exist_ok=True)
        plot_smvmd_decomposition(
            original=segment,
            mimfs=result.mimfs,
            center_freqs=result.center_freqs,
            channel_idx=0,
            fs=fs,
            channel_name="Fp1",
            save_path=plot_path,
        )
        logger.info(f"  Diagnostic plot: {plot_path}")

    return {'smvmd_result': result}


def run_phase4_verify_features(phase2_result: dict, phase3_result: dict,
                                config: dict) -> dict:
    """Phase 4: Feature extraction verification."""
    logger.info("\n" + "="*60 + "\nPHASE 4: Feature Extraction\n" + "="*60)

    smvmd_result = phase3_result['smvmd_result']
    fs = config['ds2']['sampling_freq']

    if smvmd_result.n_modes == 0:
        logger.error("PHASE 4 FAILED: No SMVMD modes to extract features from")
        return {}

    features = extract_features_from_segment(smvmd_result.mimfs, fs=fs)
    expected_shape = (19, smvmd_result.n_modes, 9)

    logger.info(f"PHASE 4 Results:")
    logger.info(f"  Feature array shape: {features.shape}")
    logger.info(f"  Expected: (19, K={smvmd_result.n_modes}, 9)")
    logger.info(f"  Shape OK: {features.shape == expected_shape}")

    # Check for NaN/Inf
    from src.utils import check_array_health
    health_ok = check_array_health(features, name="DS2_features_phase4")

    # Show feature values for first mode, first channel
    logger.info(f"\n  Example (ch=0, mode=0):")
    from src.features import FEATURE_NAMES
    for fname, fval in zip(FEATURE_NAMES, features[0, 0, :]):
        logger.info(f"    {fname}: {fval:.6f}")

    return {'feature_array': features}


def run_phase5_verify_integration(phase2_result: dict, phase3_result: dict,
                                   phase4_result: dict, config: dict) -> dict:
    """Phase 5: Energy-based feature integration verification."""
    logger.info("\n" + "="*60 + "\nPHASE 5: Energy-Based Feature Integration\n" + "="*60)

    smvmd_result = phase3_result['smvmd_result']
    feature_array = phase4_result['feature_array']

    integrated = energy_integrate_segment(smvmd_result, feature_array)

    expected_len = 19 * 9  # = 171 for DS-2
    logger.info(f"PHASE 5 Results:")
    logger.info(f"  Integrated feature length: {len(integrated)}")
    logger.info(f"  Expected: {expected_len} (19 channels x 9 features)")
    logger.info(f"  Assertion: {'PASS' if len(integrated) == expected_len else 'FAIL'}")

    assert len(integrated) == expected_len, (
        f"Integrated feature length {len(integrated)} != {expected_len}"
    )

    from src.utils import check_array_health
    check_array_health(integrated, name="DS2_integrated_phase5")

    return {'integrated': integrated}


def run_full_ds2_pipeline(config: dict, use_cache: bool = True) -> dict:
    """Run complete DS-2 pipeline (Phase 8: full dataset) and save everything."""
    logger.info("\n" + "="*60 + "\nFULL DS-2 PIPELINE\n" + "="*60)

    ensure_dirs()
    t_start = time.perf_counter()
    timings: dict = {}

    fs = config['ds2']['sampling_freq']
    ds2_cfg = config['ds2']
    smvmd_cfg = ds2_cfg['smvmd']
    exp_key = 'DS2'

    fingerprint = feature_fingerprint(config, 'ds2', 'visual_task')
    cached = load_feature_cache(exp_key, fingerprint) if use_cache else None

    if cached is not None:
        X, y, groups, n_modes = cached
    else:
        # 1. Load all subjects
        loader = DS2Loader(config['paths']['ds2_adhd_dir'])
        eeg_data, metadata = loader.load_all_subjects()

        if not eeg_data:
            logger.error("No DS-2 data loaded. Aborting.")
            return {}

        # 2. Preprocess all subjects
        with Stopwatch(timings, 'preprocess'):
            logger.info("Preprocessing all DS-2 subjects...")
            eeg_preprocessed = {}
            for subj_id, eeg in eeg_data.items():
                eeg_preprocessed[subj_id] = preprocess_ds2(
                    eeg=eeg, fs=fs, **dict(ds2_cfg['preprocessing']),
                )

        # 3. Segment all subjects
        with Stopwatch(timings, 'segment'):
            logger.info("Segmenting all DS-2 subjects...")
            segments, segment_df = segment_dataset_ds2(
                eeg_data=eeg_preprocessed,
                metadata_list=metadata,
                fs=fs,
                window_sec=config['segmentation']['window_sec'],
                overlap=config['segmentation']['overlap'],
            )

        verify_all_segments(segments, segment_df, n_channels=19)

        meta_path = config['paths']['metadata'] + '/ds2_segments.csv'
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        segment_df.to_csv(meta_path, index=False)
        logger.info(f"Segment metadata saved: {meta_path}")

        y = segment_df['label'].values
        groups = segment_df['subject_id'].values

        # 4. SMVMD decomposition
        with Stopwatch(timings, 'smvmd'):
            logger.info(
                f"SMVMD decomposition: alpha={smvmd_cfg['alpha']}, "
                f"tau={smvmd_cfg['tau']}"
            )
            smvmd_results = decompose_segments(
                segments=segments,
                alpha=smvmd_cfg['alpha'],
                tau=smvmd_cfg['tau'],
                tol=smvmd_cfg['tol'],
                max_iter=smvmd_cfg.get('max_iter', 500),
                fs=fs,
            )

        # 5. Feature extraction
        with Stopwatch(timings, 'features'):
            logger.info("Extracting features from SMVMD modes...")
            feature_lists = extract_features_batch(smvmd_results, fs=fs)

        # 6. Energy integration
        with Stopwatch(timings, 'energy_integration'):
            logger.info("Applying energy-based feature integration...")
            X = integrate_all_segments(
                smvmd_results=smvmd_results,
                feature_lists=feature_lists,
                n_channels=19,
                dataset='DS2',
            )

        assert X.shape == (len(segments), 171), (
            f"Feature matrix shape {X.shape} != ({len(segments)}, 171)"
        )
        verify_feature_matrix(X, 'DS2')

        n_modes = np.array([r.n_modes for r in smvmd_results], dtype=np.int32)
        save_feature_cache(exp_key, X, y, groups, fingerprint, n_modes.tolist())

    logger.info(
        f"DS-2 feature matrix: {X.shape} "
        f"({int(np.sum(y == 1))} ADHD / {int(np.sum(y == 0))} NC segments; "
        f"paper: 3680 ADHD / 2908 NC)"
    )

    # 7. mRMR (DS-2: all 171 features retained, per the paper)
    n_features = ds2_cfg['mrmr']['n_features']  # 171
    logger.info(f"mRMR: using {n_features} features (paper-specified)")

    knn_params = {
        'n_neighbors': ds2_cfg['knn']['n_neighbors'],        # 2
        'distance': ds2_cfg['knn']['distance'],               # cosine
        'weights': ds2_cfg['knn']['weights'],                  # squared_inverse
        'standardize': ds2_cfg['knn']['standardize'],          # False
    }

    # Accuracy vs. number of ranked features (paper Fig. 5)
    mrmr_curve = None
    ranked_idx = np.arange(X.shape[1])
    with Stopwatch(timings, 'mrmr_curve'):
        try:
            ranked_idx = compute_mrmr_ranking(X, y, n_features=X.shape[1])
            n_opt, feat_counts, accs = select_features_by_accuracy(
                X=X, y=y, ranked_indices=ranked_idx, knn_params=knn_params,
                cv_folds=config['evaluation']['paper_style']['n_splits'],
                random_seed=config.get('random_seed', 42),
            )
            plot_path = os.path.join(PLOTS_DIR, 'DS2_mrmr_accuracy.png')
            plot_accuracy_vs_features(
                n_features_list=feat_counts.tolist(),
                accuracies=accs.tolist(),
                optimal_n=n_opt,
                dataset_name='DS-2 (ADHD)',
                save_path=plot_path,
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
        except Exception as e:
            logger.warning(f"mRMR curve step failed ({e}); continuing.")

    # 8. KNN evaluation, protocol A (paper-style segment-level CV)
    with Stopwatch(timings, 'cv_paper_protocol'):
        results_paper = evaluate_knn_paper_style(
            X=X, y=y,
            knn_params=knn_params,
            n_features_target=n_features,
            cv_folds=config['evaluation']['paper_style']['n_splits'],
            random_seed=config.get('random_seed', 42),
            dataset_name='DS2',
        )

    # Protocol B: subject-independent audit
    si_summary = None
    with Stopwatch(timings, 'cv_subject_independent'):
        try:
            results_si = evaluate_knn_subject_independent(
                X=X, y=y, groups=groups,
                knn_params=knn_params,
                n_features_target=n_features,
                dataset_name='DS2',
            )
            si_summary = {k: v for k, v in results_si.items()
                          if k not in ('y_true', 'y_pred')}
        except Exception as e:
            logger.warning(f"Subject-independent audit failed ({e}); continuing.")

    # 9. Full metrics
    metrics = compute_all_metrics(
        y_true=results_paper['y_true'],
        y_pred=results_paper['y_pred'],
        dataset='DS2',
        class_names=['NC (0)', 'ADHD (1)'],
    )

    comparison = compare_with_paper_targets(metrics, 'DS2')
    print_results_table(metrics, 'DS-2 ADHD Detection', comparison)

    # 10. Confusion matrix
    cm_path = os.path.join(CM_DIR, 'cm_DS2.png')
    plot_confusion_matrix(
        cm=metrics['confusion_matrix'],
        class_names=['NC', 'ADHD'],
        title='DS-2: ADHD vs NC',
        save_path=cm_path,
    )

    # 11. Classifier comparison (Table V)
    classifier_comparison = None
    with Stopwatch(timings, 'classifier_comparison'):
        try:
            classifier_comparison = _run_classifier_comparison(
                X, y, n_features, ranked_idx, config)
        except Exception as e:
            logger.warning(f"Classifier comparison failed ({e}); continuing.")

    # 12. Save the single consolidated bundle
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
        fold_accuracies=results_paper.get('fold_accuracies'),
        classifier_comparison=classifier_comparison,
        mrmr_curve=mrmr_curve,
        mode_stats=mode_stats,
        runtime_sec=timings['total'],
        extra={'timings_sec': timings,
               'segment_counts': {'adhd': int(np.sum(y == 1)),
                                  'nc': int(np.sum(y == 0)),
                                  'paper_adhd': 3680, 'paper_nc': 2908},
               'confusion_matrix_plot': cm_path},
    )

    save_table(exp_key, 'fold_accuracies', [
        {'fold': i + 1, 'accuracy': float(a)}
        for i, a in enumerate(results_paper.get('fold_accuracies', []))
    ])

    logger.info(f"DS-2 pipeline complete in {timings['total']:.1f}s")
    return metrics


def _run_classifier_comparison(X: np.ndarray, y: np.ndarray,
                                n_features: int, ranked_idx: np.ndarray,
                                config: dict) -> dict:
    """Run DT, KNN, SVM, Ensemble comparison (Paper Table V).

    Returns the per-classifier results so the caller can fold them into the
    experiment's single saved bundle.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import accuracy_score, f1_score

    logger.info("\nClassifier Comparison (Table V)...")
    X_sel = X[:, ranked_idx[:n_features]]
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    from src.knn_model import build_comparison_classifiers, build_knn_with_scaler
    classifiers = build_comparison_classifiers()

    # Add paper-tuned KNN
    classifiers['KNN'] = build_knn_with_scaler(config['ds2']['knn'])

    comparison_results = {}
    for name, clf in classifiers.items():
        if clf is None:
            continue
        try:
            import time
            t0 = time.time()
            y_pred = cross_val_predict(clf, X_sel, y, cv=skf, n_jobs=-1)
            elapsed_ms = (time.time() - t0) * 1000 / len(X_sel)

            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            comparison_results[name] = {
                'accuracy': float(acc),
                'f1': float(f1),
                'time_ms_per_sample': float(elapsed_ms),
            }
            logger.info(
                f"  {name}: Accuracy={acc*100:.2f}%, F1={f1*100:.2f}%"
            )
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    save_table('DS2', 'classifier_comparison', [
        {'classifier': name,
         'accuracy': r['accuracy'],
         'f1': r['f1'],
         'time_ms_per_sample': r['time_ms_per_sample']}
        for name, r in comparison_results.items()
    ])
    return comparison_results


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main(use_cache: bool = True, quick_checks: bool = None):
    """Main DS-2 experiment entry point.

    Runs staged pipeline as configured in config.yaml.

    Args:
        use_cache: Reuse a cached feature matrix when its fingerprint matches.
        quick_checks: Override the phase1-5 sanity-check flags. None keeps
            whatever configs/config.yaml says.
    """
    ensure_dirs()
    config = load_config('configs/config.yaml')
    set_random_seed(config.get('random_seed', 42))

    logger.info("=" * 70)
    logger.info("EEG-Based ADHD Detection (DS-2)")
    logger.info("Paper: Chandela, Faisal & Sharma, IEEE TCDS, 2025")
    logger.info("=" * 70)

    pipeline_cfg = dict(config.get('pipeline', {}))
    if quick_checks is not None:
        for phase in ('phase1_load_one_subject', 'phase2_one_segment',
                      'phase3_verify_smvmd', 'phase4_verify_features',
                      'phase5_verify_energy_integration'):
            pipeline_cfg[phase] = quick_checks

    loader = DS2Loader(config['paths']['ds2_adhd_dir'])
    logger.info(loader.report())

    results = {}

    # Stage 1
    if pipeline_cfg.get('phase1_load_one_subject', True):
        phase1 = run_phase1_load_one_subject(config, loader)
        if not phase1:
            logger.error("Phase 1 failed. Aborting.")
            return

    # Stage 2
    if pipeline_cfg.get('phase2_one_segment', True):
        phase2 = run_phase2_one_segment(phase1, config)

    # Stage 3
    if pipeline_cfg.get('phase3_verify_smvmd', True):
        phase3 = run_phase3_verify_smvmd(phase2, config)

    # Stage 4
    if pipeline_cfg.get('phase4_verify_features', True):
        phase4 = run_phase4_verify_features(phase2, phase3, config)

    # Stage 5
    if pipeline_cfg.get('phase5_verify_energy_integration', True):
        phase5 = run_phase5_verify_integration(phase2, phase3, phase4, config)

    # Stage 8 (Full Pipeline)
    if pipeline_cfg.get('phase8_full_dataset', False):
        results = run_full_ds2_pipeline(config, use_cache=use_cache)
    else:
        logger.warning(
            "pipeline.phase8_full_dataset is false in configs/config.yaml, so "
            "only the sanity checks ran and NO results were saved. Set it to "
            "true to produce the DS-2 results for the report."
        )

    logger.info("\n" + "="*70)
    logger.info("DS-2 Experiment Complete")
    logger.info("="*70)

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='DS-2 ADHD vs NC experiment')
    parser.add_argument('--no-cache', action='store_true',
                        help='Ignore any cached feature matrix and recompute '
                             'preprocessing, SMVMD and feature extraction.')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip the phase 1-5 single-segment sanity checks '
                             'and go straight to the full dataset.')
    args = parser.parse_args()
    main(use_cache=not args.no_cache,
         quick_checks=False if args.skip_checks else None)
