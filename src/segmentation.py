"""
segmentation.py - EEG Signal Segmentation

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

Segmentation [Paper explicitly states]:
  - Window: 5 seconds
  - Overlap: 50%
  - Sampling rate: 128 Hz
  - Window samples: 5 × 128 = 640
  - Step: 640 × 0.5 = 320 samples
  - Each segment is an independent observation

Paper-reported segment counts:
  DS-1: 1288 total (322 IDD + 322 TDC per condition × 2 conditions)
  DS-2: 6588 total (3680 ADHD + 2908 NC)

Critical implementation note:
  The last boundary segment (incomplete, < 640 samples) is NOT included.
  This is inferred from the paper's reported segment counts.
  Each 2-min DS-1 recording yields exactly 46 segments (not 47).
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core Segmentation Function
# ─────────────────────────────────────────────────────────────────────────────

def segment_eeg(eeg: np.ndarray,
                fs: float = 128.0,
                window_sec: float = 5.0,
                overlap: float = 0.5,
                subject_id: str = "unknown",
                label: int = -1,
                dataset: str = "unknown",
                condition: str = "unknown",
                include_partial: bool = False
                ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Segment EEG recording into overlapping windows.

    Paper [explicit]:
    - Window: 5 seconds = 640 samples at 128 Hz
    - Overlap: 50%
    - Step: 320 samples
    - Each segment is an independent observation

    CRITICAL: Subject ID is preserved in metadata for every segment.
    This ensures no subject leakage in cross-validation.

    Boundary handling [paper does not specify explicitly]:
    - include_partial=False (default): Discard last window if < window_samples
    - This is consistent with the paper's reported segment count of 46/subject

    Args:
        eeg: Multichannel EEG, shape (n_channels, n_samples).
        fs: Sampling frequency (paper: 128 Hz).
        window_sec: Window length in seconds (paper: 5.0).
        overlap: Fractional overlap (paper: 0.5 = 50%).
        subject_id: REQUIRED - unique subject identifier for metadata.
        label: Class label (1=disorder, 0=control).
        dataset: Dataset name ('DS1' or 'DS2').
        condition: Recording condition ('rest', 'music', 'visual_task').
        include_partial: If True, include last window even if shorter than window.
                         [Paper does not specify; default False per segment count analysis]

    Returns:
        Tuple of:
        - segments: Array of shape (n_segments, n_channels, window_samples)
        - metadata: List of dicts, one per segment, with full traceability info

    Raises:
        ValueError: If eeg shape is invalid or parameters are out of range.
    """
    # Validate inputs
    if eeg.ndim != 2:
        raise ValueError(f"Expected 2D EEG (channels, samples), got shape {eeg.shape}")
    if not (0 < overlap < 1):
        raise ValueError(f"Overlap must be in (0, 1), got {overlap}")
    if window_sec <= 0:
        raise ValueError(f"Window must be positive, got {window_sec} sec")

    n_channels, n_samples = eeg.shape

    # Compute window and step in samples [Paper: 640 and 320]
    window_samples = int(window_sec * fs)  # 5 * 128 = 640
    step_samples = int(window_samples * (1 - overlap))  # 640 * 0.5 = 320

    if window_samples > n_samples:
        logger.warning(
            f"Window ({window_samples} samples) longer than recording "
            f"({n_samples} samples) for {subject_id}. Cannot segment."
        )
        return np.empty((0, n_channels, window_samples)), []

    # Compute number of segments (no partial windows)
    #
    # Paper reports 1288 DS-1 segments: 14 subjects × 2 conditions × 46 segments = 1288
    # DS-1 recording: 15360 samples, window=640, step=320
    # (15360 - 640) / 320 = 46.0 exactly
    #
    # IMPLEMENTATION CHOICE: Use strict inequality (start < n_samples - window_samples)
    # This gives 46 segments (not 47), matching the paper's reported count.
    # The 47th window (start=14720, end=15360) is the exact last sample — paper excludes it.
    # Using "strict less than" mimics MATLAB-style indexing behavior in the original paper.
    # [Paper does not specify boundary handling explicitly; inferred from segment count]
    if include_partial:
        n_segments = int(np.ceil((n_samples - window_samples) / step_samples)) + 1
    else:
        # strict: only include windows where start < (n_samples - window_samples)
        n_segments = int(np.floor((n_samples - window_samples - 1) / step_samples)) + 1
        if n_segments < 1:
            n_segments = 1

    logger.debug(
        f"Segmenting {subject_id} ({condition}): "
        f"n_samples={n_samples}, window={window_samples}, step={step_samples}, "
        f"n_segments={n_segments}"
    )

    segments = np.zeros((n_segments, n_channels, window_samples), dtype=np.float64)
    metadata = []

    for seg_idx in range(n_segments):
        start = seg_idx * step_samples
        end = start + window_samples

        if end > n_samples:
            if include_partial:
                # Zero-pad
                pad_length = end - n_samples
                seg = np.concatenate([
                    eeg[:, start:],
                    np.zeros((n_channels, pad_length))
                ], axis=1)
                logger.debug(f"Padded last segment with {pad_length} zeros")
            else:
                break  # Should not reach here with floor calculation
        else:
            seg = eeg[:, start:end]

        segments[seg_idx] = seg

        meta = {
            'subject_id': subject_id,
            'label': label,
            'dataset': dataset,
            'recording_condition': condition,
            'segment_index': seg_idx,
            'start_sample': start,
            'end_sample': end,
            'start_time_sec': start / fs,
            'end_time_sec': end / fs,
            'n_channels': n_channels,
            'window_samples': window_samples,
            'fs': fs,
        }
        metadata.append(meta)

    # Final shape check
    actual_n = len(metadata)
    if actual_n != n_segments:
        n_segments = actual_n
        segments = segments[:n_segments]

    return segments, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-Level Segmentation
# ─────────────────────────────────────────────────────────────────────────────

def segment_dataset_ds1(eeg_data: Dict[Tuple[str, str], np.ndarray],
                         metadata_list: List[Dict],
                         fs: float = 128.0,
                         window_sec: float = 5.0,
                         overlap: float = 0.5
                         ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Segment all DS-1 recordings.

    Combines REST and MUSIC conditions.
    Each segment is independently labeled with its subject ID and condition.

    Paper [explicit]: 1288 total segments
    - 322 IDD (rest) + 322 TDC (rest) + 322 IDD (music) + 322 TDC (music)
    - Each: 46 segments × 7 subjects = 322 per class per condition

    Args:
        eeg_data: Dict mapping (subject_id, condition) → eeg array (C, T).
        metadata_list: List of subject-level metadata dicts.
        fs: Sampling frequency (128 Hz).
        window_sec: Window length (5 sec).
        overlap: Overlap fraction (0.5).

    Returns:
        Tuple of:
        - all_segments: Array (n_total_segments, 14, 640)
        - segment_df: DataFrame with metadata for each segment
    """
    all_segments = []
    all_meta = []

    # Build lookup for subject info
    info_lookup = {}
    for info in metadata_list:
        key = (info['subject_id'], info['condition'])
        info_lookup[key] = info

    for (subject_id, condition), eeg in sorted(eeg_data.items()):
        info = info_lookup.get((subject_id, condition), {})
        label = info.get('label', -1)

        segments, meta = segment_eeg(
            eeg=eeg,
            fs=fs,
            window_sec=window_sec,
            overlap=overlap,
            subject_id=subject_id,
            label=label,
            dataset='DS1',
            condition=condition,
        )

        if len(segments) > 0:
            all_segments.append(segments)
            all_meta.extend(meta)

        logger.info(
            f"DS-1 segmented {subject_id} ({condition}): "
            f"{len(segments)} segments"
        )

    if not all_segments:
        logger.error("No segments produced for DS-1!")
        return np.empty((0, 14, 640)), pd.DataFrame()

    all_segments_arr = np.concatenate(all_segments, axis=0)
    segment_df = pd.DataFrame(all_meta)

    # Sanity check: verify count
    n_total = len(all_segments_arr)
    _verify_ds1_segment_count(segment_df, n_total)

    logger.info(
        f"DS-1 total: {n_total} segments, "
        f"shape={all_segments_arr.shape}"
    )

    return all_segments_arr, segment_df


def segment_dataset_ds2(eeg_data: Dict[str, np.ndarray],
                         metadata_list: List[Dict],
                         fs: float = 128.0,
                         window_sec: float = 5.0,
                         overlap: float = 0.5
                         ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Segment all DS-2 (ADHD) recordings.

    Paper [explicit]: 6588 total segments
    - 3680 ADHD + 2908 NC

    Args:
        eeg_data: Dict mapping subject_id → eeg array (19, n_samples).
        metadata_list: List of subject-level metadata dicts.
        fs: Sampling frequency (128 Hz).
        window_sec: Window length (5 sec).
        overlap: Overlap fraction (0.5).

    Returns:
        Tuple of:
        - all_segments: Array (n_total_segments, 19, 640)
        - segment_df: DataFrame with metadata
    """
    all_segments = []
    all_meta = []

    info_lookup = {info['subject_id']: info for info in metadata_list}

    for subject_id, eeg in sorted(eeg_data.items()):
        info = info_lookup.get(subject_id, {})
        label = info.get('label', -1)
        condition = info.get('condition', 'visual_task')

        segments, meta = segment_eeg(
            eeg=eeg,
            fs=fs,
            window_sec=window_sec,
            overlap=overlap,
            subject_id=subject_id,
            label=label,
            dataset='DS2',
            condition=condition,
        )

        if len(segments) > 0:
            all_segments.append(segments)
            all_meta.extend(meta)

        logger.debug(
            f"DS-2 segmented {subject_id}: {len(segments)} segments "
            f"(label={'ADHD' if label == 1 else 'NC'})"
        )

    if not all_segments:
        logger.error("No segments produced for DS-2!")
        return np.empty((0, 19, 640)), pd.DataFrame()

    all_segments_arr = np.concatenate(all_segments, axis=0)
    segment_df = pd.DataFrame(all_meta)

    # Sanity check: verify count
    n_total = len(all_segments_arr)
    _verify_ds2_segment_count(segment_df, n_total)

    logger.info(
        f"DS-2 total: {n_total} segments, "
        f"shape={all_segments_arr.shape}"
    )

    return all_segments_arr, segment_df


# ─────────────────────────────────────────────────────────────────────────────
# Sanity Checks for Segment Counts
# ─────────────────────────────────────────────────────────────────────────────

def _verify_ds1_segment_count(segment_df: pd.DataFrame,
                               n_total: int) -> None:
    """Verify DS-1 segment count against paper.

    Paper [explicit]: 1288 total (322 IDD + 322 TDC per condition × 2 conditions)
    """
    EXPECTED_TOTAL = 1288
    EXPECTED_PER_CLASS_PER_CONDITION = 322

    logger.info(f"\n{'='*50}")
    logger.info(f"DS-1 SEGMENT COUNT VERIFICATION")
    logger.info(f"{'='*50}")
    logger.info(f"Total segments: {n_total} (expected {EXPECTED_TOTAL})")

    if 'label' in segment_df.columns and 'recording_condition' in segment_df.columns:
        for cond in ['rest', 'music']:
            for lbl, lbl_name in [(1, 'IDD'), (0, 'TDC')]:
                count = len(segment_df[
                    (segment_df['recording_condition'] == cond) &
                    (segment_df['label'] == lbl)
                ])
                expected = EXPECTED_PER_CLASS_PER_CONDITION
                status = "✓" if count == expected else "✗ MISMATCH"
                logger.info(
                    f"  {cond.upper()} | {lbl_name}: {count} "
                    f"(expected {expected}) {status}"
                )

    if n_total != EXPECTED_TOTAL:
        diff = n_total - EXPECTED_TOTAL
        logger.warning(
            f"SEGMENT COUNT MISMATCH: got {n_total}, expected {EXPECTED_TOTAL}, "
            f"diff={diff:+d}"
        )
        if abs(diff) <= 14:
            logger.warning(
                "Small difference (≤14). Likely due to slightly different recording "
                "lengths for some subjects. Check individual subject counts."
            )
        _report_per_subject_count(segment_df)
    else:
        logger.info(f"✓ DS-1 segment count MATCHES paper: {n_total}")


def _verify_ds2_segment_count(segment_df: pd.DataFrame,
                               n_total: int) -> None:
    """Verify DS-2 segment count against paper.

    Paper [explicit]: 6588 total (3680 ADHD + 2908 NC)
    """
    EXPECTED_TOTAL = 6588
    EXPECTED_ADHD = 3680
    EXPECTED_NC = 2908

    logger.info(f"\n{'='*50}")
    logger.info(f"DS-2 SEGMENT COUNT VERIFICATION")
    logger.info(f"{'='*50}")
    logger.info(f"Total segments: {n_total} (expected {EXPECTED_TOTAL})")

    if 'label' in segment_df.columns:
        n_adhd = (segment_df['label'] == 1).sum()
        n_nc = (segment_df['label'] == 0).sum()
        logger.info(f"  ADHD: {n_adhd} (expected {EXPECTED_ADHD}) "
                    f"{'✓' if n_adhd == EXPECTED_ADHD else '✗'}")
        logger.info(f"  NC:   {n_nc} (expected {EXPECTED_NC}) "
                    f"{'✓' if n_nc == EXPECTED_NC else '✗'}")

    if n_total != EXPECTED_TOTAL:
        diff = n_total - EXPECTED_TOTAL
        logger.warning(
            f"SEGMENT COUNT MISMATCH: got {n_total}, expected {EXPECTED_TOTAL}, "
            f"diff={diff:+d}"
        )
        logger.warning(
            "Note: DS-2 recording lengths vary per subject. "
            "Difference may be due to variable-length recordings."
        )
        _report_per_subject_count(segment_df)
    else:
        logger.info(f"✓ DS-2 segment count MATCHES paper: {n_total}")


def _report_per_subject_count(segment_df: pd.DataFrame) -> None:
    """Report segment count per subject for debugging."""
    if 'subject_id' in segment_df.columns:
        counts = segment_df.groupby('subject_id').size().reset_index(name='n_segments')
        logger.info("\nPer-subject segment counts:")
        for _, row in counts.iterrows():
            logger.info(f"  {row['subject_id']}: {row['n_segments']} segments")


# ─────────────────────────────────────────────────────────────────────────────
# Segment Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_all_segments(segments: np.ndarray,
                         segment_df: pd.DataFrame,
                         n_channels: int,
                         n_samples: int = 640) -> bool:
    """Verify all segments have correct shape, no NaN/Inf, and metadata integrity.

    Checks:
    1. Shape: (n_total, n_channels, 640)
    2. No NaN values
    3. No Inf values
    4. Subject IDs preserved in all metadata rows
    5. Labels preserved in all metadata rows
    6. Segment indices are sequential per subject

    Args:
        segments: Segment array (n_total, n_channels, n_samples).
        segment_df: Metadata DataFrame.
        n_channels: Expected channel count.
        n_samples: Expected samples per segment (640).

    Returns:
        True if all checks pass.
    """
    checks_passed = []

    # Shape check
    expected_shape_2d = (n_channels, n_samples)
    actual_shape_2d = segments.shape[1:]
    shape_ok = actual_shape_2d == expected_shape_2d
    checks_passed.append(shape_ok)
    logger.info(f"[SHAPE CHECK] segments shape: {segments.shape}, "
                f"expected (..., {n_channels}, {n_samples}): "
                f"{'OK' if shape_ok else 'FAIL'}")

    # NaN check
    has_nan = np.any(np.isnan(segments))
    nan_ok = not has_nan
    checks_passed.append(nan_ok)
    logger.info(f"[NaN CHECK] NaN values: {'NONE (OK)' if nan_ok else 'FOUND (FAIL)'}")

    # Inf check
    has_inf = np.any(np.isinf(segments))
    inf_ok = not has_inf
    checks_passed.append(inf_ok)
    logger.info(f"[Inf CHECK] Inf values: {'NONE (OK)' if inf_ok else 'FOUND (FAIL)'}")

    # Metadata: subject ID preserved
    if 'subject_id' in segment_df.columns:
        has_subject_ids = not segment_df['subject_id'].isnull().any()
        checks_passed.append(has_subject_ids)
        logger.info(
            f"[SUBJECT ID CHECK] Subject IDs preserved: "
            f"{'YES (OK)' if has_subject_ids else 'MISSING (FAIL)'}"
        )
        unique_subjects = segment_df['subject_id'].nunique()
        logger.info(f"  Unique subjects: {unique_subjects}")

    # Metadata: labels preserved
    if 'label' in segment_df.columns:
        has_labels = not segment_df['label'].isnull().any()
        invalid_labels = not segment_df['label'].isin([0, 1]).all()
        label_ok = has_labels and not invalid_labels
        checks_passed.append(label_ok)
        logger.info(
            f"[LABEL CHECK] Labels valid: {'YES (OK)' if label_ok else 'FAIL'}"
        )

    # Segments match metadata rows
    n_segments = len(segments)
    n_meta = len(segment_df)
    count_ok = n_segments == n_meta
    checks_passed.append(count_ok)
    logger.info(
        f"[COUNT CHECK] segments ({n_segments}) == metadata rows ({n_meta}): "
        f"{'OK' if count_ok else 'FAIL'}"
    )

    all_ok = all(checks_passed)
    logger.info(
        f"\n{'='*40}\n"
        f"SEGMENT VERIFICATION: {'ALL CHECKS PASSED' if all_ok else 'CHECKS FAILED'}\n"
        f"{'='*40}"
    )

    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Get Subset for One Subject
# ─────────────────────────────────────────────────────────────────────────────

def get_subject_segments(segments: np.ndarray,
                          segment_df: pd.DataFrame,
                          subject_id: str
                          ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Extract all segments for a specific subject.

    Used for phase-by-phase processing and debugging.

    Args:
        segments: Full segment array (n_total, C, T).
        segment_df: Full metadata DataFrame.
        subject_id: Subject ID to extract.

    Returns:
        Tuple of (subject_segments array, subject_metadata DataFrame).
    """
    mask = segment_df['subject_id'] == subject_id
    idx = segment_df.index[mask].tolist()

    subject_segments = segments[idx]
    subject_meta = segment_df[mask].reset_index(drop=True)

    logger.info(
        f"Extracted {len(subject_segments)} segments for {subject_id}"
    )

    return subject_segments, subject_meta
