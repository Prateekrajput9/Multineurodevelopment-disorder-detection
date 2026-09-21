"""
subject_independent_audit.py - Subject-Independent Robustness Evaluation

IMPORTANT: This is NOT a claim of reproducing the paper.
This is an additional methodological audit using subject-independent cross-validation.

Protocol B (subject-independent):
  - GroupKFold or StratifiedGroupKFold
  - group = subject_id (all segments from one subject stay in one fold)
  - Prevents data leakage between train and test via subject identity

The paper uses Protocol A (segment-level 10-fold CV), which may overestimate
performance due to subject-level leakage. This experiment quantifies that gap.

Results are clearly labeled "Subject-Independent Audit — NOT Paper Reproduction"

NOTE: each of the four main experiments now runs this subject-independent
audit itself and saves it into results/metrics/<experiment>.json under
'subject_independent', so the report already contains Protocol B. This
script remains as a standalone way to run the audit on its own.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import logging

from src.utils import load_config, setup_logger, set_random_seed
from src.data_loader import DS1Loader, DS2Loader
from src.preprocessing import preprocess_ds1, preprocess_ds2
from src.segmentation import segment_eeg
from src.smvmd import decompose_segments
from src.features import extract_features_batch
from src.energy_integration import integrate_all_segments, verify_feature_matrix
from src.knn_model import evaluate_knn_subject_independent
from src.evaluation import compute_all_metrics, print_results_table

logger = setup_logger('SubjectIndependent', 'results/logs/subject_independent_audit.log')


# ─────────────────────────────────────────────────────────────────────────────
# DS-1 Subject-Independent Audit
# ─────────────────────────────────────────────────────────────────────────────

def run_ds1_audit(config: dict, condition: str = 'rest'):
    """Run subject-independent audit for DS-1.

    Args:
        config: Configuration dictionary.
        condition: 'rest', 'music', or 'both'.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"DS-1 SUBJECT-INDEPENDENT AUDIT: {condition.upper()}")
    logger.info("Protocol B: GroupKFold (subject_id as group)")
    logger.info("NOT a paper reproduction claim.")
    logger.info('='*60)

    ds1_cfg = config['ds1']
    fs = ds1_cfg['sampling_freq']
    smvmd_cfg = ds1_cfg['smvmd']
    mode = ds1_cfg['preprocessing']['mode']

    ds1_data_root = config['paths']['ds1_data_root']
    loader = DS1Loader(ds1_data_root, mode=mode)
    files = loader.discover_files()

    all_segments, all_labels, all_groups = [], [], []

    conditions = ['rest', 'music'] if condition == 'both' else [condition]

    for subject_id, cond_files in sorted(files.items()):
        for cond in conditions:
            if cond not in cond_files:
                continue
            eeg, info = loader.load_subject(subject_id, cond, cond_files[cond])
            if eeg is None:
                continue
            eeg = preprocess_ds1(eeg, fs=fs, mode=mode,
                                  config=ds1_cfg['preprocessing'])
            segs, _ = segment_eeg(eeg, fs=fs, subject_id=subject_id,
                                   label=info['label'], dataset='DS1',
                                   condition=cond)
            all_segments.append(segs)
            all_labels.extend([info['label']] * len(segs))
            all_groups.extend([subject_id] * len(segs))

    if not all_segments:
        logger.error("No segments produced for DS-1 audit.")
        return None

    segments = np.concatenate(all_segments, axis=0)
    y = np.array(all_labels)
    groups = np.array(all_groups)

    logger.info(f"Segments: {len(segments)}, Subjects: {len(set(groups))}")

    # SMVMD + features + integration
    smvmd_results = decompose_segments(segments=segments,
                                        alpha=smvmd_cfg['alpha'],
                                        tau=smvmd_cfg['tau'],
                                        tol=smvmd_cfg['tol'], fs=fs)
    feature_lists = extract_features_batch(smvmd_results, fs=fs)
    X = integrate_all_segments(smvmd_results, feature_lists,
                                n_channels=14, dataset='DS1')

    # Use paper's mRMR count for this condition
    cond_key = {
        'rest': 'n_features_rest', 'music': 'n_features_music',
        'both': 'n_features_rest_music'
    }[condition]
    n_features = ds1_cfg['mrmr'][cond_key]

    # KNN params from config
    knn_key = {'rest': 'rest', 'music': 'music', 'both': 'rest_music'}[condition]
    knn_params = ds1_cfg['knn'][knn_key]

    results_si = evaluate_knn_subject_independent(
        X=X, y=y, groups=groups, knn_params=knn_params,
        n_features_target=n_features,
        dataset_name=f'DS1_{condition}_SubjectIndependent'
    )

    metrics = compute_all_metrics(results_si['y_true'], results_si['y_pred'],
                                    dataset='DS1',
                                    class_names=['TDC (0)', 'IDD (1)'])
    print_results_table(
        metrics,
        f'DS-1 {condition.capitalize()} — SUBJECT-INDEPENDENT AUDIT (Protocol B)',
        comparison=None
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# DS-2 Subject-Independent Audit
# ─────────────────────────────────────────────────────────────────────────────

def run_ds2_audit(config: dict):
    """Run subject-independent audit for DS-2 (ADHD)."""
    logger.info(f"\n{'='*60}")
    logger.info("DS-2 SUBJECT-INDEPENDENT AUDIT: ADHD vs NC")
    logger.info("Protocol B: GroupKFold (subject_id as group)")
    logger.info("NOT a paper reproduction claim.")
    logger.info('='*60)

    ds2_cfg = config['ds2']
    fs = ds2_cfg['sampling_freq']
    smvmd_cfg = ds2_cfg['smvmd']
    pre_cfg = ds2_cfg['preprocessing']

    ds2_dir = config['paths']['ds2_adhd_dir']
    loader = DS2Loader(ds2_dir)
    files = loader.discover_files()

    all_segments, all_labels, all_groups = [], [], []

    for subject_id, file_info in sorted(files.items()):
        eeg, info = loader.load_subject(subject_id, file_info)
        if eeg is None:
            continue
        eeg = preprocess_ds2(eeg, fs=fs,
                              bandpass_low=pre_cfg['bandpass_low'],
                              bandpass_high=pre_cfg['bandpass_high'],
                              butterworth_order=pre_cfg['butterworth_order'],
                              notch_freq=pre_cfg['notch_freq'],
                              wavelet=pre_cfg['wavelet'],
                              wavelet_level=pre_cfg['wavelet_level'])
        segs, _ = segment_eeg(eeg, fs=fs, subject_id=subject_id,
                               label=info['label'], dataset='DS2',
                               condition='visual_task')
        all_segments.append(segs)
        all_labels.extend([info['label']] * len(segs))
        all_groups.extend([subject_id] * len(segs))

    if not all_segments:
        logger.error("No segments produced for DS-2 audit.")
        return None

    segments = np.concatenate(all_segments, axis=0)
    y = np.array(all_labels)
    groups = np.array(all_groups)

    logger.info(f"Segments: {len(segments)} (expected ~6588), "
                f"Subjects: {len(set(groups))} (expected 121)")

    smvmd_results = decompose_segments(segments=segments,
                                        alpha=smvmd_cfg['alpha'],
                                        tau=smvmd_cfg['tau'],
                                        tol=smvmd_cfg['tol'], fs=fs)
    feature_lists = extract_features_batch(smvmd_results, fs=fs)
    X = integrate_all_segments(smvmd_results, feature_lists,
                                n_channels=19, dataset='DS2')
    verify_feature_matrix(X, 'DS2_SubjectIndependent')

    n_features = ds2_cfg['mrmr']['n_features']  # 171

    knn_params = ds2_cfg['knn']
    results_si = evaluate_knn_subject_independent(
        X=X, y=y, groups=groups, knn_params=knn_params,
        n_features_target=n_features,
        dataset_name='DS2_ADHD_SubjectIndependent'
    )

    metrics = compute_all_metrics(results_si['y_true'], results_si['y_pred'],
                                    dataset='DS2',
                                    class_names=['NC (0)', 'ADHD (1)'])
    print_results_table(
        metrics,
        'DS-2 ADHD vs NC — SUBJECT-INDEPENDENT AUDIT (Protocol B)',
        comparison=None
    )
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    config = load_config('configs/config.yaml')
    set_random_seed(config.get('random_seed', 42))

    logger.info("SUBJECT-INDEPENDENT ROBUSTNESS AUDIT")
    logger.info("This is Protocol B — NOT Paper Reproduction")
    logger.info("Results cannot be directly compared to paper-reported numbers.")

    print("\n" + "="*60)
    print("DS-1 AUDITS")
    print("="*60)
    run_ds1_audit(config, condition='rest')
    run_ds1_audit(config, condition='music')
    run_ds1_audit(config, condition='both')

    print("\n" + "="*60)
    print("DS-2 AUDIT")
    print("="*60)
    run_ds2_audit(config)
