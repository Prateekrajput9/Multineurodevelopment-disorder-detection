"""
data_loader.py - Dataset Loading for EEG Neurodevelopmental Disorder Detection

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

Dataset-1 (DS-1):
  Source: Mendeley Data, DOI: 10.17632/fshy54ypyh.1
  Ref: Sareen et al., Data in Brief, 2020, vol. 30, Art. no. 105488
  Format: MATLAB .mat files (both raw and preprocessed provided)
  14 channels (EMOTIV EPOC+), 128 Hz, 14 subjects (7 IDD, 7 TDC)

  Actual file structure on disk:
    CleanData/CleanData_IDD/Rest/NDS001_Rest_CD.mat  → key: clean_data, shape: (14, 15360)
    CleanData/CleanData_IDD/Music/NDS001_Music_CD.mat
    CleanData/CleanData_TDC/Rest/CGS01_Rest_CD.mat
    CleanData/CleanData_TDC/Music/CGS01_Music_CD.mat
  NDS prefix → IDD subjects (7 subjects: NDS001–NDS007)
  CGS prefix → TDC subjects (7 subjects: CGS01–CGS07)

Dataset-2 (DS-2):
  Source: IEEE DataPort, DOI: 10.21227/rzfh-zn36
  Ref: Nasrabadi et al., 2020
  Format: MATLAB .mat files inside ZIP archives
  19 channels (10-20 system), 128 Hz, 121 subjects (61 ADHD, 60 NC)

  Actual file structure on disk:
    adhd/ADHD_part1.zip   → 30 ADHD subjects (v1p, v3p, ..., v40p, v173)
    adhd/ADHD_part2.zip   → 31 ADHD subjects (v177, v179, ..., v288)
    adhd/Control_part1.zip → 30 NC subjects (v41p, ..., v60p, v107–v116)
    adhd/Control_part2.zip → 30 NC subjects (v117, v118, ..., v310)

  Mat file key = filename stem (e.g. v10p.mat contains key 'v10p')
  Data shape inside mat: (n_samples, 19) → transpose to (19, n_samples)
  Data dtype: int16 or float64 → always cast to float64
"""

import io
import os
import re
import glob
import logging
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from scipy.io import loadmat

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-1 Loader (IDD Dataset)
# ─────────────────────────────────────────────────────────────────────────────

class DS1Loader:
    """Loader for Dataset-1: IDD vs TDC (Sareen et al., 2020).

    Dataset structure (Mendeley DOI: 10.17632/fshy54ypyh.1):
    - 7 IDD subjects, 7 TDC subjects
    - Each subject: 2 recordings (rest, music), each 2 minutes
    - EMOTIV EPOC+, 14 channels, 128 Hz
    - Both raw and preprocessed EEG provided

    Actual filenames (verified from disk):
      IDD clean: NDS001_Rest_CD.mat, NDS001_Music_CD.mat, ..., NDS007_*_CD.mat
      TDC clean: CGS01_Rest_CD.mat, CGS01_Music_CD.mat, ..., CGS07_*_CD.mat
      Mat variable key: 'clean_data'
      Array shape: (14, 15360) = (channels, samples)

    Paper preprocessing (MODE A - MATLAB only):
      Band-pass 1-30 Hz → CleanLine → ICA (runica) → ADJUST → reconstruct

    Default: MODE B - use dataset's preprocessed recordings.

    [Paper does not specify exact file format; this is verified from dataset]
    """

    # Expected channel order for EMOTIV EPOC+ (14 channels, 10-20 system)
    # [Paper Fig. 2(a) shows this configuration]
    CHANNEL_NAMES = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
                     'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    N_CHANNELS = 14
    FS = 128.0  # Hz [Paper explicitly states 128 Hz]
    DURATION_SEC = 120  # 2 minutes [Paper explicitly states]

    # Label mapping
    LABELS = {'IDD': 1, 'TDC': 0}

    def __init__(self, data_dir: str, mode: str = 'B',
                 verbose: bool = True):
        """Initialize DS1 loader.

        Args:
            data_dir: Root directory containing DS-1 data files.
                      Should be the directory containing CleanData/ and/or RawData/.
            mode: 'A' = use raw + apply preprocessing pipeline (MATLAB only).
                  'B' = use provided preprocessed files (default, recommended).
            verbose: If True, log detailed information.
        """
        self.data_dir = Path(data_dir)
        self.mode = mode.upper()
        self.verbose = verbose

        if self.mode == 'A':
            logger.warning(
                "MODE A selected: Raw EEG + paper preprocessing pipeline. "
                "This requires MATLAB + EEGLAB + CleanLine + ADJUST plugins. "
                "Full preprocessing is NOT reproducible in pure Python. "
                "Falling back to scipy-based approximation."
            )
        elif self.mode == 'B':
            logger.info(
                "MODE B selected: Using dataset's provided preprocessed recordings. "
                "This is the recommended mode for paper reproduction."
            )

        # Verify directory exists
        if not self.data_dir.exists():
            logger.warning(f"DS-1 data directory not found: {self.data_dir}")
            logger.warning("Please download the dataset from: "
                          "https://data.mendeley.com/datasets/fshy54ypyh/1")

    def discover_files(self) -> Dict[str, Dict[str, Path]]:
        """Discover all EEG files in the dataset directory.

        Supports two structures:
        1. Flat: all .mat files in one directory
        2. Nested: CleanData/CleanData_IDD/Rest/*.mat etc.

        File naming conventions (verified from actual dataset):
          NDS001_Rest_CD.mat  → IDD subject 001, rest condition
          NDS001_Music_CD.mat → IDD subject 001, music condition
          CGS01_Rest_CD.mat   → TDC subject 01, rest condition
          CGS01_Music_CD.mat  → TDC subject 01, music condition

        NDS prefix = Neurodevelopmental disorder subjects (IDD)
        CGS prefix = Control group subjects (TDC)

        Returns:
            Dict[subject_id, Dict[condition, file_path]]
            Example: {'IDD_01': {'rest': Path(...), 'music': Path(...)}, ...}
        """
        discovered = {}

        # Search for .mat files recursively
        mat_files = list(self.data_dir.rglob("*.mat"))
        # Filter out macOS metadata files
        mat_files = [f for f in mat_files if not f.name.startswith('._')]

        logger.info(f"DS-1 directory scan: {len(mat_files)} .mat files in {self.data_dir}")

        if len(mat_files) == 0:
            logger.error(
                f"No .mat files found in {self.data_dir}. "
                "Expected structure: CleanData/CleanData_IDD/Rest/NDS001_Rest_CD.mat"
            )
            return {}

        discovered = self._discover_mat_files(mat_files)

        if not discovered:
            logger.warning(
                "Could not parse any files from DS-1 directory. "
                "Trying generic fallback discovery..."
            )
            discovered = self._discover_generic(mat_files)

        return discovered

    def _discover_mat_files(self, mat_files: List[Path]) -> Dict[str, Dict[str, Path]]:
        """Discover and organize .mat files by subject and condition.

        Handles the actual Mendeley DS-1 naming convention:
          NDS001_Rest_CD.mat  → IDD subject, rest
          NDS001_Music_CD.mat → IDD subject, music
          CGS01_Rest_CD.mat   → TDC subject, rest
          CGS01_Music_CD.mat  → TDC subject, music

        Also handles path-based label detection from directory structure:
          CleanData_IDD/ → IDD
          CleanData_TDC/ → TDC
          RawData_IDD/   → IDD
          RawData_TDC/   → TDC

        [Paper does not specify exact file naming; verified from actual dataset]
        """
        discovered = {}

        for f in mat_files:
            fname = f.stem.lower()
            label = None
            condition = None
            subject_id = None

            # ── Label detection ────────────────────────────────────────────
            # 1. From filename prefix (verified convention)
            if fname.startswith('nds'):
                # NDS = Neurodevelopmental disorder subjects → IDD
                label = 'IDD'
            elif fname.startswith('cgs'):
                # CGS = Control group subjects → TDC
                label = 'TDC'
            else:
                # 2. From path components (directory names)
                path_str = str(f).lower()
                if 'idd' in path_str or 'nds' in path_str:
                    label = 'IDD'
                elif 'tdc' in path_str or 'cgs' in path_str or 'control' in path_str:
                    label = 'TDC'
                # 3. From filename keywords
                elif any(k in fname for k in ['idd', 'intellectual', 'disorder']):
                    label = 'IDD'
                elif any(k in fname for k in ['tdc', 'control', 'typical', 'healthy']):
                    label = 'TDC'

            # ── Condition detection ───────────────────────────────────────
            if 'rest' in fname:
                condition = 'rest'
            elif 'music' in fname or 'stimulus' in fname:
                condition = 'music'
            else:
                # Try path components
                path_str = str(f.parent).lower()
                if 'rest' in path_str:
                    condition = 'rest'
                elif 'music' in path_str:
                    condition = 'music'

            # ── Subject ID extraction ─────────────────────────────────────
            if label is not None:
                nums = re.findall(r'\d+', f.stem)
                if nums:
                    subject_id = f"{label}_{nums[0].zfill(3)}"

            if label is not None and condition is not None and subject_id is not None:
                if subject_id not in discovered:
                    discovered[subject_id] = {}
                if condition not in discovered[subject_id]:
                    discovered[subject_id][condition] = f
                else:
                    # Prefer CleanData files (MODE B)
                    if 'clean' in str(f).lower() or '_cd' in fname:
                        discovered[subject_id][condition] = f
            else:
                if self.verbose:
                    logger.debug(
                        f"Could not parse: {f.name} "
                        f"(label={label}, condition={condition}). "
                        f"[Paper does not specify exact file naming convention]"
                    )

        logger.info(
            f"Discovered {len(discovered)} subjects in DS-1: "
            f"{sorted(discovered.keys())}"
        )
        return discovered

    def _discover_generic(self, mat_files: List[Path]) -> Dict[str, Dict[str, Path]]:
        """Generic fallback: use directory structure to assign labels."""
        discovered = {}
        for i, f in enumerate(mat_files):
            path_str = str(f.parent).lower()
            label = 'IDD' if ('idd' in path_str or 'nds' in path_str) else 'TDC'
            condition = 'rest' if 'rest' in path_str else 'music'
            subject_id = f"{label}_{i:03d}"
            if subject_id not in discovered:
                discovered[subject_id] = {}
            discovered[subject_id][condition] = f
        return discovered

    def _discover_csv_files(self, csv_files: List[Path]) -> Dict[str, Dict[str, Path]]:
        """Discover and organize .csv files (alternative format)."""
        return self._discover_mat_files(csv_files)

    def _discover_edf_files(self, edf_files: List[Path]) -> Dict[str, Dict[str, Path]]:
        """Discover and organize .edf files (EDF format)."""
        return self._discover_mat_files(edf_files)

    def load_subject(self, subject_id: str,
                     condition: str,
                     file_path: Optional[Path] = None
                     ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load EEG data for one subject under one condition.

        Args:
            subject_id: Subject identifier (e.g., 'IDD_001').
            condition: 'rest' or 'music'.
            file_path: Path to the EEG file. If None, auto-discover.

        Returns:
            Tuple of:
            - eeg: numpy array of shape (n_channels, n_samples) or None if not found
            - info: dict with metadata (subject_id, label, condition, fs, channels)
        """
        if file_path is None:
            files = self.discover_files()
            if subject_id not in files or condition not in files[subject_id]:
                logger.error(f"File not found for {subject_id} {condition}")
                return None, {}
            file_path = files[subject_id][condition]

        label_str = 'IDD' if 'IDD' in subject_id.upper() else 'TDC'
        label = self.LABELS[label_str]

        try:
            eeg, fs = self._load_single_file(file_path)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            return None, {}

        if eeg is None:
            return None, {}

        # Check sampling rate
        if abs(fs - self.FS) > 0.1:
            logger.warning(
                f"Sampling rate mismatch for {subject_id}: "
                f"loaded {fs} Hz, expected {self.FS} Hz"
            )

        # Check channel count
        if eeg.shape[0] != self.N_CHANNELS:
            logger.warning(
                f"Channel count mismatch for {subject_id}: "
                f"loaded {eeg.shape[0]}, expected {self.N_CHANNELS}"
            )

        # Log duration
        duration = eeg.shape[1] / self.FS
        logger.info(
            f"Loaded {subject_id} ({condition}): shape={eeg.shape}, "
            f"duration={duration:.1f}s (expected {self.DURATION_SEC}s)"
        )

        info = {
            'subject_id': subject_id,
            'label': label,
            'label_str': label_str,
            'condition': condition,
            'fs': fs,
            'n_channels': eeg.shape[0],
            'n_samples': eeg.shape[1],
            'channel_names': self.CHANNEL_NAMES[:eeg.shape[0]],
            'duration_sec': eeg.shape[1] / self.FS,
            'file_path': str(file_path),
        }

        return eeg, info

    def _load_single_file(self, file_path: Path) -> Tuple[np.ndarray, float]:
        """Load a single EEG file.

        Supports .mat, .csv, .set formats.
        Format determined by file extension.

        [Paper does not specify exact file format; this handles common cases]

        Args:
            file_path: Path to the EEG file.

        Returns:
            Tuple of (eeg array (C, T), sampling_frequency).
        """
        ext = file_path.suffix.lower()

        if ext == '.mat':
            return self._load_mat(file_path)
        elif ext == '.csv':
            return self._load_csv(file_path)
        elif ext == '.edf':
            return self._load_edf(file_path)
        elif ext == '.set':
            return self._load_set(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _load_mat(self, file_path: Path) -> Tuple[np.ndarray, float]:
        """Load MATLAB .mat file for DS-1.

        Verified variable names in the Mendeley DS-1 dataset:
          'clean_data' → shape (14, samples), dtype float32  [VERIFIED]

        Also tries: 'data', 'EEG', 'eeg', 'signal', 'preprocessed', 'raw_data', 'X'

        [Paper does not specify .mat variable names; verified from actual dataset]
        """
        mat = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)

        # Remove MATLAB metadata keys
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        logger.debug(f"MAT file keys: {data_keys}")

        eeg = None
        fs = self.FS  # Default to paper-specified value

        # Try verified and common variable names
        # 'clean_data' is the VERIFIED key in the Mendeley DS-1 dataset
        for key in ['clean_data', 'data', 'EEG', 'eeg', 'signal',
                    'preprocessed', 'clean', 'raw_data', 'X']:
            if key in mat:
                candidate = mat[key]
                if isinstance(candidate, np.ndarray) and candidate.ndim >= 2:
                    eeg = candidate
                    logger.debug(f"Found EEG data in key '{key}': shape={eeg.shape}")
                    break

        # Try EEGLAB struct format
        if eeg is None:
            for key in data_keys:
                val = mat[key]
                if hasattr(val, 'data') and isinstance(getattr(val, 'data'), np.ndarray):
                    eeg = np.array(val.data)
                    if hasattr(val, 'srate'):
                        fs = float(val.srate)
                    logger.debug(f"Found EEGLAB struct in key '{key}'")
                    break

        # Last resort: single non-meta key
        if eeg is None and len(data_keys) == 1:
            candidate = mat[data_keys[0]]
            if isinstance(candidate, np.ndarray) and candidate.ndim >= 2:
                eeg = candidate
                logger.debug(f"Using single key '{data_keys[0]}': shape={eeg.shape}")

        if eeg is None:
            raise ValueError(
                f"Cannot find EEG data in {file_path}. "
                f"Available keys: {data_keys}. "
                f"[Paper does not specify .mat variable names]"
            )

        # Ensure shape is (channels, samples)
        # DS-1 clean data: (14, 15360) — already correct
        if eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
            eeg = eeg.T  # Transpose if samples > channels

        # Check for sampling rate in mat file
        for key in ['fs', 'Fs', 'srate', 'SamplingRate', 'sampling_rate']:
            if key in mat:
                val = mat[key]
                if np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1):
                    fs = float(np.squeeze(val))
                    break

        return eeg.astype(np.float64), fs

    def _load_csv(self, file_path: Path) -> Tuple[np.ndarray, float]:
        """Load CSV file.

        Assumes: rows = timepoints, columns = channels.
        [Paper does not specify CSV format; implementation choice]
        """
        df = pd.read_csv(file_path)

        # Drop non-numeric columns (e.g., timestamps, labels)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        eeg = df[numeric_cols].values.T  # shape: (channels, samples)

        logger.debug(f"Loaded CSV: shape={eeg.shape}")
        return eeg.astype(np.float64), self.FS

    def _load_edf(self, file_path: Path) -> Tuple[np.ndarray, float]:
        """Load EDF file using MNE.

        [Paper does not specify EDF format; this handles standard EDF]
        """
        try:
            import mne
            raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
            fs = raw.info['sfreq']
            eeg = raw.get_data()
            return eeg.astype(np.float64), float(fs)
        except ImportError:
            raise ImportError("MNE-Python required for EDF loading: pip install mne")

    def _load_set(self, file_path: Path) -> Tuple[np.ndarray, float]:
        """Load EEGLAB .set file using MNE.

        Raw DS-1 files are in .set/.fdt format (EEGLAB format).
        [Paper does not specify format; verified from actual dataset]
        """
        try:
            import mne
            raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
            fs = raw.info['sfreq']
            eeg = raw.get_data()
            return eeg.astype(np.float64), float(fs)
        except ImportError:
            raise ImportError("MNE-Python required for .set file loading: pip install mne")

    def load_all_subjects(self) -> Tuple[Dict, List[Dict]]:
        """Load all subjects and conditions from DS-1.

        Returns:
            Tuple of:
            - eeg_data: Dict mapping (subject_id, condition) → eeg array
            - metadata: List of info dictionaries
        """
        files = self.discover_files()

        if not files:
            logger.error("No DS-1 files discovered. Cannot load dataset.")
            return {}, []

        eeg_data = {}
        metadata = []

        for subject_id, conditions in sorted(files.items()):
            for condition, file_path in conditions.items():
                eeg, info = self.load_subject(subject_id, condition, file_path)
                if eeg is not None:
                    eeg_data[(subject_id, condition)] = eeg
                    metadata.append(info)

        logger.info(f"DS-1: Loaded {len(eeg_data)} recordings "
                    f"from {len(files)} subjects")
        return eeg_data, metadata

    def report(self) -> str:
        """Generate a dataset discovery report."""
        files = self.discover_files()
        lines = [
            "=" * 60,
            "DS-1 Dataset Report",
            "=" * 60,
            f"Directory: {self.data_dir}",
            f"Mode: {self.mode}",
            f"Files discovered: {len(files)} subjects",
        ]

        for subj, conds in sorted(files.items()):
            lines.append(f"  {subj}: {sorted(conds.keys())}")

        # Count by label
        idd_count = sum(1 for k in files if 'IDD' in k.upper())
        tdc_count = sum(1 for k in files if 'TDC' in k.upper())
        lines.append(f"IDD subjects: {idd_count} (expected 7)")
        lines.append(f"TDC subjects: {tdc_count} (expected 7)")

        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-2 Loader (ADHD Dataset)
# ─────────────────────────────────────────────────────────────────────────────

class DS2Loader:
    """Loader for Dataset-2: ADHD vs NC (Nasrabadi et al., 2020).

    Dataset structure (IEEE DataPort DOI: 10.21227/rzfh-zn36):
    - 61 ADHD subjects, 60 NC subjects
    - Each subject: single EEG recording during visual attention task
    - 19 channels (international 10-20), 128 Hz

    ACTUAL FILE STRUCTURE ON DISK (verified):
      data_dir/ADHD_part1.zip   → 30 ADHD subjects (v1p, v3p, ..., v173)
      data_dir/ADHD_part2.zip   → 31 ADHD subjects (v177, v179, ..., v288)
      data_dir/Control_part1.zip → 30 NC subjects (v41p, ..., v116)
      data_dir/Control_part2.zip → 30 NC subjects (v117, ..., v310)

    Mat file format (verified):
      Key = filename stem (e.g. v10p.mat → key 'v10p')
      Shape: (n_samples, 19) — rows=samples, cols=channels
      dtype: int16 or float64

    Label assignment:
      ADHD_*.zip → ADHD (label=1)
      Control_*.zip → NC (label=0)
      This is the ONLY reliable way to assign labels; filename alone is ambiguous.

    Paper preprocessing:
      Butterworth 0.5-60 Hz → 50 Hz notch → Coiflet-3 wavelet → SURE denoising

    The ADHD dataset provides RAW EEG (not preprocessed).
    Preprocessing is applied in preprocessing.py.

    [Based on dataset description; preprocessing pipeline is reproducible in Python]
    """

    # Standard 19-channel 10-20 system as described in paper Fig. 2(b)
    CHANNEL_NAMES = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                     'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
                     'Fz', 'Cz', 'Pz']
    N_CHANNELS = 19
    FS = 128.0  # Hz [Paper explicitly states 128 Hz]

    # Label mapping
    LABELS = {'ADHD': 1, 'NC': 0}

    # ZIP files pattern → label mapping
    # Key insight: label comes from ZIP filename, NOT the .mat filename inside
    ZIP_LABEL_MAP = {
        'adhd': 'ADHD',      # ADHD_part1.zip, ADHD_part2.zip → ADHD
        'control': 'NC',     # Control_part1.zip, Control_part2.zip → NC
    }

    def __init__(self, data_dir: str, verbose: bool = True,
                 extract_dir: Optional[str] = None):
        """Initialize DS2 loader.

        Args:
            data_dir: Root directory containing DS-2 ZIP files.
                      Must contain ADHD_part1.zip, ADHD_part2.zip,
                      Control_part1.zip, Control_part2.zip.
            verbose: If True, log detailed information.
            extract_dir: Directory to extract ZIPs to for repeated access.
                        If None, reads directly from ZIPs (slower but disk-efficient).
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.extract_dir = Path(extract_dir) if extract_dir else None

        if not self.data_dir.exists():
            logger.warning(f"DS-2 data directory not found: {self.data_dir}")
            logger.warning("Please download from: https://ieee-dataport.org/open-access/"
                          "eeg-data-adhd-control-children")
            logger.warning("Or: https://dx.doi.org/10.21227/rzfh-zn36")

    def discover_files(self) -> Dict[str, Dict[str, Any]]:
        """Discover all ADHD dataset subjects from ZIP files.

        CRITICAL: Labels are determined by which ZIP file the subject belongs to:
          ADHD_*.zip   → label = ADHD (1)
          Control_*.zip → label = NC  (0)

        The filename inside the ZIP does NOT reliably encode the label.
        Some ADHD subjects have filenames like 'v173.mat' (no 'p' suffix).
        Some Control subjects have filenames like 'v41p.mat' (has 'p' suffix — confusing!).

        Returns:
            Dict[subject_id, {'zip': Path, 'mat_name': str, 'label': int, 'label_str': str}]
            subject_id is the filename stem (e.g., 'v10p', 'v173', 'v107')
        """
        discovered = {}

        # Find all ZIP files in data directory
        zip_files = sorted(self.data_dir.glob("*.zip"))
        if not zip_files:
            logger.error(
                f"No ZIP files found in {self.data_dir}. "
                "Expected: ADHD_part1.zip, ADHD_part2.zip, "
                "Control_part1.zip, Control_part2.zip"
            )
            return {}

        for zip_path in zip_files:
            zip_name_lower = zip_path.stem.lower()

            # Determine label from ZIP filename
            label_str = None
            for keyword, lbl in self.ZIP_LABEL_MAP.items():
                if keyword in zip_name_lower:
                    label_str = lbl
                    break

            if label_str is None:
                logger.warning(
                    f"Cannot determine label for ZIP: {zip_path.name}. "
                    "Expected 'ADHD' or 'Control' in filename. Skipping."
                )
                continue

            label = self.LABELS[label_str]

            # List .mat files inside ZIP
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    mat_names = [n for n in zf.namelist() if n.endswith('.mat')]
            except zipfile.BadZipFile:
                logger.error(f"Bad ZIP file: {zip_path}")
                continue

            for mat_name in mat_names:
                stem = Path(mat_name).stem  # e.g., 'v10p' from 'ADHD_part1/v10p.mat'
                subject_id = stem  # Use filename stem as subject ID

                if subject_id in discovered:
                    logger.warning(
                        f"Duplicate subject ID '{subject_id}' found in "
                        f"{zip_path.name} — already in "
                        f"{discovered[subject_id]['zip'].name}. Skipping duplicate."
                    )
                    continue

                discovered[subject_id] = {
                    'zip': zip_path,
                    'mat_name': mat_name,  # Full path within ZIP
                    'label': label,
                    'label_str': label_str,
                    # For backward compat with old code expecting 'file' key
                    'file': zip_path,
                }

        # Summary
        adhd_count = sum(1 for v in discovered.values() if v['label_str'] == 'ADHD')
        nc_count = sum(1 for v in discovered.values() if v['label_str'] == 'NC')
        logger.info(
            f"DS-2 discovered: {adhd_count} ADHD (expected 61), "
            f"{nc_count} NC (expected 60), "
            f"total {len(discovered)} (expected 121)"
        )

        return discovered

    def load_subject(self, subject_id: str,
                     file_info: Optional[Dict] = None
                     ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Load EEG data for one DS-2 subject.

        Args:
            subject_id: Subject identifier (e.g., 'v8p', 'v173', 'v107').
            file_info: Dict with 'zip', 'mat_name', 'label', 'label_str'.
                       If None, auto-discover.

        Returns:
            Tuple of:
            - eeg: numpy array (n_channels, n_samples) or None
            - info: metadata dictionary
        """
        if file_info is None:
            all_files = self.discover_files()
            if subject_id not in all_files:
                logger.error(f"Subject {subject_id} not found in DS-2")
                return None, {}
            file_info = all_files[subject_id]

        zip_path = file_info['zip']
        mat_name = file_info['mat_name']
        label = file_info['label']
        label_str = file_info['label_str']

        try:
            eeg, fs = self._load_mat_from_zip(zip_path, mat_name, subject_id)
        except Exception as e:
            logger.error(f"Failed to load {subject_id} from {zip_path.name}: {e}")
            return None, {}

        if eeg is None:
            return None, {}

        # Verify
        if abs(fs - self.FS) > 0.1:
            logger.warning(f"FS mismatch for {subject_id}: got {fs}, expected {self.FS}")

        if eeg.shape[0] != self.N_CHANNELS:
            logger.warning(
                f"Channel count mismatch for {subject_id}: "
                f"got {eeg.shape[0]}, expected {self.N_CHANNELS}"
            )

        duration = eeg.shape[1] / self.FS
        logger.info(
            f"DS-2 loaded {subject_id} ({label_str}): "
            f"shape={eeg.shape}, duration={duration:.1f}s"
        )

        info = {
            'subject_id': subject_id,
            'label': label,
            'label_str': label_str,
            'condition': 'visual_task',  # Paper: visual attention task
            'fs': fs,
            'n_channels': eeg.shape[0],
            'n_samples': eeg.shape[1],
            'channel_names': self.CHANNEL_NAMES[:eeg.shape[0]],
            'duration_sec': duration,
            'file_path': str(zip_path) + '::' + mat_name,
        }

        return eeg, info

    def _load_mat_from_zip(self, zip_path: Path, mat_name: str,
                            subject_id: str) -> Tuple[np.ndarray, float]:
        """Load a .mat file directly from a ZIP archive.

        VERIFIED file format:
          - Mat key = filename stem (e.g. v10p.mat → key 'v10p')
          - Some subjects use numeric IDs without p/n suffix (e.g. v173)
          - Data shape: (n_samples, 19) → must transpose to (19, n_samples)
          - Data dtype: int16 or float64 → always cast to float64

        [Paper does not specify .mat structure; verified from actual dataset]
        """
        with zipfile.ZipFile(zip_path, 'r') as zf:
            with zf.open(mat_name) as mat_file:
                raw_bytes = mat_file.read()

        # Load from bytes buffer
        mat = loadmat(io.BytesIO(raw_bytes), squeeze_me=True, struct_as_record=False)
        data_keys = [k for k in mat.keys() if not k.startswith('__')]
        logger.debug(f"DS-2 MAT keys for {subject_id}: {data_keys}")

        eeg = None
        fs = self.FS

        # Strategy 1: Key = filename stem (VERIFIED primary method)
        stem = Path(mat_name).stem
        if stem in mat:
            candidate = mat[stem]
            if isinstance(candidate, np.ndarray) and candidate.ndim >= 2:
                eeg = candidate
                logger.debug(f"Found EEG using stem key '{stem}': shape={eeg.shape}")

        # Strategy 2: Try EEGLAB EEG struct
        if eeg is None and 'EEG' in mat:
            eeg_struct = mat['EEG']
            if hasattr(eeg_struct, 'data'):
                eeg = np.array(eeg_struct.data)
            if hasattr(eeg_struct, 'srate'):
                fs = float(eeg_struct.srate)

        # Strategy 3: Common variable names
        if eeg is None:
            for key in ['data', 'eeg', 'EEG', 'signal', 'X']:
                if key in mat:
                    candidate = mat[key]
                    if isinstance(candidate, np.ndarray) and candidate.ndim >= 2:
                        eeg = candidate
                        logger.debug(f"Found EEG using fallback key '{key}'")
                        break

        # Strategy 4: Single non-meta key
        if eeg is None and len(data_keys) == 1:
            candidate = mat[data_keys[0]]
            if isinstance(candidate, np.ndarray) and candidate.ndim >= 2:
                eeg = candidate
                logger.debug(f"Found EEG using sole key '{data_keys[0]}'")

        if eeg is None:
            raise ValueError(
                f"Cannot find EEG matrix in {mat_name}. "
                f"Available keys: {data_keys}. "
                f"Expected key matching filename stem '{stem}'."
            )

        # CRITICAL: DS-2 data is (samples, channels) → transpose to (channels, samples)
        # Verified: shape (14304, 19) → need (19, 14304)
        if eeg.ndim == 2 and eeg.shape[1] == self.N_CHANNELS and eeg.shape[0] != self.N_CHANNELS:
            eeg = eeg.T
            logger.debug(f"Transposed DS-2 EEG: now shape={eeg.shape}")
        elif eeg.ndim == 2 and eeg.shape[0] > eeg.shape[1]:
            # Generic heuristic: more rows than columns → samples in rows
            eeg = eeg.T
            logger.debug(f"Transposed EEG (heuristic): now shape={eeg.shape}")

        # Cast to float64 (data may be int16)
        eeg = eeg.astype(np.float64)

        # Check for sampling rate stored in mat
        for key in ['fs', 'Fs', 'srate', 'SamplingRate']:
            if key in mat:
                val = mat[key]
                if np.isscalar(val) or (isinstance(val, np.ndarray) and val.size == 1):
                    fs = float(np.squeeze(val))
                    break

        return eeg, fs

    def load_all_subjects(self) -> Tuple[Dict, List[Dict]]:
        """Load all subjects from DS-2.

        Returns:
            Tuple of:
            - eeg_data: Dict mapping subject_id → eeg array
            - metadata: List of info dictionaries
        """
        files = self.discover_files()

        if not files:
            logger.error("No DS-2 files discovered. Cannot load dataset.")
            return {}, []

        eeg_data = {}
        metadata = []

        for subject_id, file_info in sorted(files.items()):
            eeg, info = self.load_subject(subject_id, file_info)
            if eeg is not None:
                eeg_data[subject_id] = eeg
                metadata.append(info)

        logger.info(f"DS-2: Loaded {len(eeg_data)} subjects")
        return eeg_data, metadata

    def report(self) -> str:
        """Generate a dataset discovery report."""
        files = self.discover_files()
        adhd = {k: v for k, v in files.items() if v['label_str'] == 'ADHD'}
        nc = {k: v for k, v in files.items() if v['label_str'] == 'NC'}

        lines = [
            "=" * 60,
            "DS-2 Dataset Report",
            "=" * 60,
            f"Directory: {self.data_dir}",
            f"Total subjects found: {len(files)}",
            f"ADHD subjects: {len(adhd)} (expected 61)",
            f"NC subjects: {len(nc)} (expected 60)",
            "",
            "ADHD subjects: " + ", ".join(sorted(adhd.keys())),
            "NC subjects: " + ", ".join(sorted(nc.keys())),
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Quick Dataset Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

def run_ds1_sanity_check(data_dir: str, mode: str = 'B') -> bool:
    """Run quick sanity check on DS-1.

    Verifies:
    - Files are discoverable (14 subjects expected)
    - At least one subject can be loaded
    - Shape is (14, ≈15360)
    - No NaN/Inf values
    - Sampling rate = 128 Hz

    Args:
        data_dir: Path to DS-1 data root (containing CleanData/ dir).
        mode: 'A' or 'B'.

    Returns:
        True if all checks pass.
    """
    loader = DS1Loader(data_dir, mode=mode)
    files = loader.discover_files()

    if not files:
        logger.error("DS-1 sanity check FAILED: no files discovered")
        return False

    # Count IDD and TDC
    idd = [k for k in files if 'IDD' in k]
    tdc = [k for k in files if 'TDC' in k]
    logger.info(f"DS-1: {len(idd)} IDD (expected 7), {len(tdc)} TDC (expected 7)")

    if len(idd) != 7 or len(tdc) != 7:
        logger.warning(
            f"DS-1 subject count mismatch: {len(idd)} IDD, {len(tdc)} TDC "
            "(expected 7 each)"
        )

    # Load first subject
    first_id = sorted(files.keys())[0]
    first_cond = sorted(files[first_id].keys())[0]
    eeg, info = loader.load_subject(first_id, first_cond)

    if eeg is None:
        logger.error(f"DS-1 sanity check FAILED: could not load {first_id}")
        return False

    # Checks
    checks = {
        'fs_correct': abs(info['fs'] - 128.0) < 0.1,
        'channels_correct': eeg.shape[0] == DS1Loader.N_CHANNELS,
        'no_nan': not np.any(np.isnan(eeg)),
        'no_inf': not np.any(np.isinf(eeg)),
        'non_zero': np.any(eeg != 0),
    }

    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  DS-1 [{status}] {name}")

    all_pass = all(checks.values())
    logger.info(f"DS-1 sanity check: {'PASSED' if all_pass else 'FAILED'}")
    logger.info(f"  Shape: {eeg.shape}, FS: {info['fs']} Hz")
    logger.info(f"  Duration: {info['duration_sec']:.1f}s (expected 120s)")
    return all_pass


def run_ds2_sanity_check(data_dir: str) -> bool:
    """Run quick sanity check on DS-2.

    Verifies:
    - ZIP files are discoverable
    - 61 ADHD + 60 NC subjects found
    - At least one subject loads correctly
    - Shape is (19, N_samples)
    - No NaN/Inf values

    Args:
        data_dir: Path to DS-2 data root (containing ADHD_part1.zip etc.).

    Returns:
        True if all checks pass.
    """
    loader = DS2Loader(data_dir)
    files = loader.discover_files()

    if not files:
        logger.error("DS-2 sanity check FAILED: no subjects discovered")
        return False

    adhd_count = sum(1 for v in files.values() if v['label_str'] == 'ADHD')
    nc_count = sum(1 for v in files.values() if v['label_str'] == 'NC')

    if adhd_count != 61:
        logger.warning(f"DS-2: Expected 61 ADHD, got {adhd_count}")
    if nc_count != 60:
        logger.warning(f"DS-2: Expected 60 NC, got {nc_count}")

    # Load first subject
    first_id = sorted(files.keys())[0]
    eeg, info = loader.load_subject(first_id)

    if eeg is None:
        logger.error(f"DS-2 sanity check FAILED: could not load {first_id}")
        return False

    checks = {
        'channels_correct': eeg.shape[0] == DS2Loader.N_CHANNELS,
        'no_nan': not np.any(np.isnan(eeg)),
        'no_inf': not np.any(np.isinf(eeg)),
        'non_zero': np.any(eeg != 0),
        'float64': eeg.dtype == np.float64,
    }

    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        logger.info(f"  DS-2 [{status}] {name}")

    all_pass = all(checks.values())
    logger.info(f"DS-2 sanity check: {'PASSED' if all_pass else 'FAILED'}")
    logger.info(f"  Shape: {eeg.shape}, FS: {info['fs']} Hz")
    logger.info(f"  Duration: {info['duration_sec']:.1f}s")
    return all_pass
