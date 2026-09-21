# Paper vs Implementation Comparison

## Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental Disorders Detection in Children Using SMVMD"

| Component | Paper (Explicit) | Our Implementation | Exact Match? | Notes |
|-----------|-----------------|-------------------|--------------|-------|
| **Dataset DS-1** | 14 subjects, 7 IDD, 7 TDC | Same | ✅ Yes | |
| **DS-1 device** | EMOTIV EPOC+, 14 channels, 128 Hz | Same | ✅ Yes | |
| **DS-1 conditions** | Rest (2 min) + Music (2 min) | Same | ✅ Yes | |
| **DS-1 preprocessing** | EEGLAB: 1-30 Hz BP, CleanLine, runica ICA, ADJUST | MODE B: dataset preprocessed files | ⚠️ Partial | MATLAB-only tools; using provided preprocessed data |
| **Dataset DS-2** | 121 subjects (61 ADHD, 60 NC), 19 ch, 128 Hz | Same | ✅ Yes | |
| **DS-2 preprocessing** | Butterworth 0.5-60 Hz, 50 Hz notch, Coiflet-3, SURE level 6 | scipy + PyWavelets | ⚠️ Partial | Butterworth order not specified; SURE implementation differs |
| **Segmentation** | 5s windows, 50% overlap | 640 samples, 320 step | ✅ Yes | |
| **DS-1 segments** | 1288 total | Target: 1288 | ✅ Yes | 46 segments/subject confirmed |
| **DS-2 segments** | 6588 total | Target: 6588 | ⚠️ To verify | Depends on actual file lengths |
| **SMVMD method** | Successive MVMD, ADMM-based | Custom Python implementation | ⚠️ Partial | Stopping criterion not explicitly stated |
| **SMVMD α (DS-1)** | 2000 | 2000 | ✅ Yes | |
| **SMVMD α (DS-2)** | 1000 | 1000 | ✅ Yes | |
| **SMVMD τ** | 0 (both datasets) | 0 | ✅ Yes | |
| **SMVMD convergence** | 1e-10 (both datasets) | 1e-10 | ✅ Yes | |
| **Variable K modes** | Yes, different per segment | Yes | ✅ Yes | |
| **Feature: STD** | √(1/N · Σ(xᵢ-μ)²) | numpy.std(ddof=0) | ✅ Yes | |
| **Feature: VAR** | 1/N · Σ(xᵢ-μ)² | numpy.var(ddof=0) | ✅ Yes | |
| **Feature: RMS** | √(1/N · Σxᵢ²) | numpy implementation | ✅ Yes | |
| **Feature: IQR** | Q3 - Q1 | scipy.stats.iqr | ✅ Yes | |
| **Feature: ASSR** | \|Σ√\|xᵢ\|\| | Custom (handles negatives) | ⚠️ Partial | Interpretation of √ for negative values |
| **Feature: AP** | Modified periodogram, Hamming window | scipy.signal.periodogram | ⚠️ Partial | Normalization not specified |
| **Feature: MFL** | log₁₀(Σ(xᵢ₊₁-xᵢ)²)^(1/2) | Custom implementation | ✅ Yes | |
| **Feature: IP** | Gaussian kernel density estimate | Custom with Silverman's σ | ⚠️ Partial | σ not specified in paper |
| **Feature: PCC** | Centered correntropy, l=1 | Custom implementation | ⚠️ Partial | σ not specified; l=1 specified |
| **Energy integration** | Eq. 7: energy-weighted average | Exact implementation | ✅ Yes | Energy = Σuᵢ² per mode per channel |
| **DS-2 feature dim** | 19ch × 9feat = 171 | 171 | ✅ Yes | |
| **DS-1 feature dim** | 14ch × 9feat = 126 | 126 | ✅ Yes | |
| **mRMR ranking** | mRMR algorithm | pymrmr / mrmr-selection | ⚠️ Partial | MI vs F-stat variant not specified |
| **DS-1 features selected (Rest)** | 80 | 80 (configurable) | ✅ Target | May differ due to mRMR implementation |
| **DS-1 features selected (Music)** | 47 | 47 (configurable) | ✅ Target | |
| **DS-1 features selected (R+M)** | 103 | 103 (configurable) | ✅ Target | |
| **DS-2 features selected** | 171 (all) | 171 | ✅ Yes | No reduction for DS-2 |
| **Classifier** | KNN | KNeighborsClassifier (sklearn) | ✅ Yes | |
| **KNN DS-2 neighbors** | 2 | 2 | ✅ Yes | |
| **KNN DS-2 distance** | Cosine | 'cosine' in sklearn | ✅ Yes | |
| **KNN DS-2 weight** | Squared Inverse | Custom callable | ⚠️ Partial | sklearn has no built-in squared inverse |
| **KNN DS-2 standardize** | False | False | ✅ Yes | |
| **Bayesian optimization** | MATLAB R2023a, 200 iter | scikit-optimize, 200 iter | ⚠️ Partial | Different optimizer internals |
| **Cross-validation** | 10-fold CV | Stratified 10-fold (sklearn) | ⚠️ Partial | MATLAB default may not be stratified |
| **DS-1 accuracy (Rest)** | 100% | TBD | 🔲 Pending | |
| **DS-1 accuracy (Music)** | 100% | TBD | 🔲 Pending | |
| **DS-1 accuracy (Rest+Music)** | 100% | TBD | 🔲 Pending | |
| **DS-2 accuracy** | 99.17% | TBD | 🔲 Pending | |
| **DS-2 MCC** | 0.983 | TBD | 🔲 Pending | |
| **DS-2 Cohen's Kappa** | 0.983 | TBD | 🔲 Pending | |
| **DS-2 GDR** | 99.239 | TBD | 🔲 Pending | |
| **DS-2 G-mean** | 0.9915 | TBD | 🔲 Pending | |
| **Classifier comparison** | DT, KNN, SVM, Ensemble | Same four | ✅ Yes | |
| **Evaluation language** | MATLAB R2023a | Python 3.x + sklearn | ⚠️ Partial | |

## Legend
- ✅ Yes: Exact or functionally equivalent match
- ⚠️ Partial: Implemented with documented differences
- 🔲 Pending: Not yet evaluated
- ❌ No: Known significant deviation

## MATLAB-Only Components (Cannot Exactly Reproduce)
1. **EEGLAB `runica` ICA** - Python MNE provides similar FastICA but not identical
2. **ADJUST plugin** - No exact Python equivalent
3. **CleanLine plugin** - Zapline-Plus (Python) is similar but not identical
4. **MATLAB Classification Learner Bayesian optimizer** - scikit-optimize is equivalent but not identical
5. **MATLAB `mrmr` function** - MI-based mRMR in Python should match

## Key Methodological Decisions

### Why MODE B for DS-1?
The Mendeley dataset (DOI: 10.17632/fshy54ypyh.1) explicitly provides preprocessed EEG data. Using this data is the most reliable way to match the paper's experimental input. The paper authors (at DIAT Pune and MIST Bangladesh) would have used EEGLAB with these specific preprocessing steps, and the preprocessed files in the dataset represent this output.

### Why custom SMVMD?
No reliable Python SMVMD library exists. The implementation is based on:
- Liu & Yu (2022): "Successive Multivariate Variational Mode Decomposition"
- The paper's Equations 1-6
- The MVMD implementation by Rehman & Aftab (2019) as reference

### Why Silverman's rule for kernel bandwidth?
Information-theoretic features (IP, PCC) require a kernel bandwidth. The paper references the ITL toolbox (MATLAB) which uses Silverman's rule by default. This is the most defensible choice without access to the original MATLAB code.
