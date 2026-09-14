# SMVMD: Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental Disorders Detection in Children

Implementation and reproduction suite of the research article:
> **"Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental Disorders Detection in Children Using Successive Multivariate Variational Mode Decomposition"**  
> *Ujjawal Chandela, Kazi Newaj Faisal, and Rishi Raj Sharma*  
> **IEEE Transactions on Cognitive and Developmental Systems (2025)**  
> DOI: [10.1109/TCDS.2025.3556888](https://doi.org/10.1109/TCDS.2025.3556888)

---

## 🌟 Overview & Key Contributions

Early detection of pediatric neurodevelopmental disorders such as **Attention Deficit Hyperactivity Disorder (ADHD)** and **Intellectual Developmental Disorder (IDD)** is critical for timely clinical intervention. This codebase provides a complete Python framework implementing:

1. **Successive Multivariate Variational Mode Decomposition (SMVMD)**:
   - Successively isolates joint Multivariate Variational Mode Functions (MVMFs) without requiring a predetermined number of modes $K$.
   - Joint center frequency ($\omega_k$) tracking aligned across all EEG channels.
   - Robust ADMM optimization with spectral overlap suppression.

2. **Energy-Based Feature Integration (EBFI)**:
   - Solves the challenge of inconsistent feature counts across subjects and epochs.
   - Mode energy normalization and weighting ($w_k = E_k / \sum E_j$).
   - Canonical EEG frequency band mapping (Delta, Theta, Alpha, Beta, Gamma).
   - Clinical electrophysiological ratio calculation (Theta/Beta ratio - TBR, Slow-to-Fast ratio).

3. **Unified Multi-Disorder Classification**:
   - Primary distance-weighted **K-Nearest Neighbors (KNN)** alongside SVM, Random Forest, and MLP.
   - Rigorous **Stratified 5-Fold Cross-Validation** and **Subject-Independent Leave-One-Subject-Out (LOSO)** validation.
   - 99%+ accuracy benchmarks on ADHD and perfect scores on IDD scenarios.

4. **Interactive Web Diagnostic Suite**:
   - Live browser-based multi-channel EEG visualizer and SMVMD mode decomposition explorer.
   - Real-time diagnostic confidence scores, neuromarkers, and energy distribution charts.

---

## 📁 Repository Structure

```
├── smvmd/                       # Core package
│   ├── __init__.py              # Exports & version info
│   ├── smvmd.py                 # SMVMD, MVMD, SVMD, VMD algorithms
│   ├── preprocessing.py         # Bandpass, Notch, Epoching, Normalization
│   ├── features.py              # Multidomain feature extraction (Energy, Spectral, Stats, Entropy)
│   ├── feature_integration.py   # Energy-Based Feature Integration (EBFI) engine
│   ├── models.py                # Classifier wrappers, CV engines & metrics
│   ├── dataset.py               # Pediatric EEG synthesizer & loaders
│   └── visualization.py         # Publication-grade plotting utilities
├── experiments/                 # Benchmark reproduction scripts
│   ├── run_adhd_detection.py    # Experiment 1: ADHD vs Typical Control
│   ├── run_idd_detection.py     # Experiment 2: IDD 3-Scenario Detection
│   ├── run_unified_detection.py # Experiment 3: Unified 3-Class Diagnosis
│   └── ablation_study.py        # Architecture ablation benchmark
├── tests/                       # Automated test suite
│   ├── test_smvmd.py            # Decomposition & reconstruction tests
│   ├── test_features.py         # Feature extractor & EBFI tests
│   └── test_pipeline.py         # End-to-end classification tests
├── static/                      # Web dashboard frontend
│   ├── index.html               # Modern UI layout
│   ├── styles.css               # Glassmorphic dark styling
│   └── app.js                   # Canvas signal rendering & API client
├── app.py                       # FastAPI web server
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Run Automated Unit Tests
```bash
python -m unittest discover tests
```

### 2. Run Paper Experiments
- **ADHD Detection Benchmark (99.17% reported)**:
  ```bash
  python experiments/run_adhd_detection.py
  ```
- **IDD 3-Scenario Detection Benchmark**:
  ```bash
  python experiments/run_idd_detection.py
  ```
- **Unified 3-Class Diagnosis (Control vs ADHD vs IDD)**:
  ```bash
  python experiments/run_unified_detection.py
  ```
- **Systematic Architecture Ablation Study**:
  ```bash
  python experiments/ablation_study.py
  ```

### 3. Launch Interactive Web Diagnostic Dashboard
```bash
python app.py
```
Open your browser at `http://127.0.0.1:8000` to interact with live multi-channel EEG signals, run SMVMD decomposition, and view clinical diagnosis in real-time.
