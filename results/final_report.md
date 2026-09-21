# EEG-Based Multiple Neurodevelopmental Disorders Detection
## Reproduction Results

Reproduction of: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma, 
*"Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental Disorders Detection in Children Using Successive Multivariate Variational Mode Decomposition"*, 
IEEE Transactions on Cognitive and Developmental Systems, 2025.

> **Research reproduction for academic study. Not a clinical diagnostic tool.** All figures below are EEG-based classification research results and cannot be used for medical diagnosis.

Experiments included: **4 of 4**.

---

## 1. Summary — reproduction vs. paper

| Experiment | Segments | Features used | Accuracy (ours) | Accuracy (paper) | Δ | F1 (ours) | F1 (paper) |
|---|---|---|---|---|---|---|---|
| DS-1: IDD vs TDC (rest state) | 644 | 80 / 126 | 99.69% | 100.00% | -0.31 pp | 99.69% | 100.00% |
| DS-1: IDD vs TDC (music stimulus) | 644 | 47 / 126 | 96.43% | 100.00% | -3.57 pp | 96.43% | 100.00% |
| DS-1: IDD vs TDC (rest + music combined) | 1288 | 103 / 126 | 99.77% | 100.00% | -0.23 pp | 99.77% | 100.00% |
| DS-2: ADHD vs NC (visual attention task) | 6588 | 171 / 171 | 85.81% | 99.17% | -13.36 pp | 85.79% | 99.05% |

Δ is our accuracy minus the paper's, in percentage points.

## 2. Classifier configuration (paper Table II)

| Experiment | k | Distance | Weighting | Standardize |
|---|---|---|---|---|
| DS-1: IDD vs TDC (rest state) | 2 | cityblock | uniform | True |
| DS-1: IDD vs TDC (music stimulus) | 2 | cityblock | squared_inverse | False |
| DS-1: IDD vs TDC (rest + music combined) | 1 | euclidean | squared_inverse | True |
| DS-2: ADHD vs NC (visual attention task) | 2 | cosine | squared_inverse | False |

---

## 3. Detailed results

### 3.1 DS-1: IDD vs TDC (rest state)

- Segments: **644** from **14** subjects
- Feature vector: 126-dimensional, mRMR keeps **80**
- SMVMD modes per segment: mean **10.79** (range 4–20) — variable K, resolved by energy-based integration
- **6.4%** of segments (41) reached the `max_modes=20` safety cap, so for those the mode count was set by that cap rather than by the residual-energy stopping rule. The paper does not state how it decides K, so treat this as an implementation caveat worth naming in the write-up.
- Run time: 45.6 min
- Run at: 2026-09-20T23:29:13

**Protocol A — paper-style 10-fold stratified cross-validation**

| Metric | Ours | Paper | Δ |
|---|---|---|---|
| ACCURACY | 99.69% | 100.00% | -0.31 pp |
| PRECISION | 99.69% | 100.00% | -0.31 pp |
| RECALL | 99.69% | 100.00% | -0.31 pp |
| F1 | 99.69% | 100.00% | -0.31 pp |


**Confusion matrix** (rows = true, columns = predicted)

| | pred TDC (0) | pred IDD (1) |
|---|---|---|
| **true TDC (0)** | 322 | 0 |
| **true IDD (1)** | 2 | 320 |

Per-fold accuracy: mean **99.69%**, std 0.62 pp, min 98.46%, max 100.00%.

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 98.46% | 98.46% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

**Protocol B — subject-independent audit (GroupKFold).** This is *not* a reproduction claim: no subject appears in both the training and test folds, which is the stricter test of whether the model generalises to unseen children.

| Metric | Value |
|---|---|
| ACCURACY | 61.96% |
| PRECISION | 64.00% |
| RECALL | 61.96% |
| F1 | 60.52% |

Accuracy drops from **99.69%** (Protocol A) to **61.96%** (Protocol B), a gap of **37.73 pp**. Discuss this gap in the report: it quantifies how much of the segment-level score comes from segments of the same child appearing in both folds.

**mRMR feature-count sweep (paper Fig. 5).** Empirical optimum at **70** features; the paper specifies **80** for this experiment. Plot: `results\plots\DS1_Rest_mrmr_accuracy.png`

Confusion matrix figure: `results\confusion_matrices\cm_DS1_Rest.png`

---

### 3.2 DS-1: IDD vs TDC (music stimulus)

- Segments: **644** from **14** subjects
- Feature vector: 126-dimensional, mRMR keeps **47**
- SMVMD modes per segment: mean **10.77** (range 5–20) — variable K, resolved by energy-based integration
- **5.1%** of segments (33) reached the `max_modes=20` safety cap, so for those the mode count was set by that cap rather than by the residual-energy stopping rule. The paper does not state how it decides K, so treat this as an implementation caveat worth naming in the write-up.
- Run time: 34.9 min
- Run at: 2026-09-21T00:10:54

**Protocol A — paper-style 10-fold stratified cross-validation**

| Metric | Ours | Paper | Δ |
|---|---|---|---|
| ACCURACY | 96.43% | 100.00% | -3.57 pp |
| PRECISION | 96.43% | 100.00% | -3.57 pp |
| RECALL | 96.43% | 100.00% | -3.57 pp |
| F1 | 96.43% | 100.00% | -3.57 pp |


**Confusion matrix** (rows = true, columns = predicted)

| | pred TDC (0) | pred IDD (1) |
|---|---|---|
| **true TDC (0)** | 311 | 11 |
| **true IDD (1)** | 12 | 310 |

Per-fold accuracy: mean **96.44%**, std 2.19 pp, min 92.31%, max 100.00%.

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 95.38% | 98.46% | 92.31% | 95.38% | 98.44% | 96.88% | 93.75% | 100.00% | 96.88% | 96.88% |

**Protocol B — subject-independent audit (GroupKFold).** This is *not* a reproduction claim: no subject appears in both the training and test folds, which is the stricter test of whether the model generalises to unseen children.

| Metric | Value |
|---|---|
| ACCURACY | 50.62% |
| PRECISION | 50.63% |
| RECALL | 50.62% |
| F1 | 50.45% |

Accuracy drops from **96.43%** (Protocol A) to **50.62%** (Protocol B), a gap of **45.81 pp**. Discuss this gap in the report: it quantifies how much of the segment-level score comes from segments of the same child appearing in both folds.

**mRMR feature-count sweep (paper Fig. 5).** Empirical optimum at **100** features; the paper specifies **47** for this experiment. Plot: `results\plots\DS1_Music_mrmr_accuracy.png`

Confusion matrix figure: `results\confusion_matrices\cm_DS1_Music.png`

---

### 3.3 DS-1: IDD vs TDC (rest + music combined)

- Segments: **1288** from **14** subjects
- Feature vector: 126-dimensional, mRMR keeps **103**
- SMVMD modes per segment: mean **10.78** (range 4–20) — variable K, resolved by energy-based integration
- **5.7%** of segments (74) reached the `max_modes=20` safety cap, so for those the mode count was set by that cap rather than by the residual-energy stopping rule. The paper does not state how it decides K, so treat this as an implementation caveat worth naming in the write-up.
- Run time: 96.4 min
- Run at: 2026-09-21T02:01:21

**Protocol A — paper-style 10-fold stratified cross-validation**

| Metric | Ours | Paper | Δ |
|---|---|---|---|
| ACCURACY | 99.77% | 100.00% | -0.23 pp |
| PRECISION | 99.77% | 100.00% | -0.23 pp |
| RECALL | 99.77% | 100.00% | -0.23 pp |
| F1 | 99.77% | 100.00% | -0.23 pp |


**Confusion matrix** (rows = true, columns = predicted)

| | pred TDC (0) | pred IDD (1) |
|---|---|---|
| **true TDC (0)** | 644 | 0 |
| **true IDD (1)** | 3 | 641 |

Per-fold accuracy: mean **99.77%**, std 0.50 pp, min 98.44%, max 100.00%.

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 100.00% | 100.00% | 100.00% | 100.00% | 99.22% | 100.00% | 100.00% | 100.00% | 100.00% | 98.44% |

**Protocol B — subject-independent audit (GroupKFold).** This is *not* a reproduction claim: no subject appears in both the training and test folds, which is the stricter test of whether the model generalises to unseen children.

| Metric | Value |
|---|---|
| ACCURACY | 62.19% |
| PRECISION | 63.09% |
| RECALL | 62.19% |
| F1 | 61.53% |

Accuracy drops from **99.77%** (Protocol A) to **62.19%** (Protocol B), a gap of **37.58 pp**. Discuss this gap in the report: it quantifies how much of the segment-level score comes from segments of the same child appearing in both folds.

**mRMR feature-count sweep (paper Fig. 5).** Empirical optimum at **95** features; the paper specifies **103** for this experiment. Plot: `results\plots\DS1_RestMusic_mrmr_accuracy.png`

Confusion matrix figure: `results\confusion_matrices\cm_DS1_RestMusic.png`

---

### 3.4 DS-2: ADHD vs NC (visual attention task)

- Segments: **6588** from **121** subjects
- Feature vector: 171-dimensional, mRMR keeps **171**
- SMVMD modes per segment: mean **4.55** (range 1–19) — variable K, resolved by energy-based integration
- Run time: 85.3 min
- Run at: 2026-09-21T03:31:19

**Protocol A — paper-style 10-fold stratified cross-validation**

| Metric | Ours | Paper | Δ |
|---|---|---|---|
| ACCURACY | 85.81% | 99.17% | -13.36 pp |
| PRECISION | 85.79% | 99.04% | -13.25 pp |
| RECALL | 85.81% | 99.07% | -13.26 pp |
| F1 | 85.79% | 99.05% | -13.26 pp |
| MCC | 0.7117 | not reported | — |
| KAPPA | 0.7115 | not reported | — |
| GDR | 85.450 | not reported | — |
| GMEAN | 0.8545 | not reported | — |


**Confusion matrix** (rows = true, columns = predicted)

| | pred NC (0) | pred ADHD (1) |
|---|---|---|
| **true NC (0)** | 2408 | 500 |
| **true ADHD (1)** | 435 | 3245 |

Per-fold accuracy: mean **85.81%**, std 0.96 pp, min 84.80%, max 87.86%.

| Fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Accuracy | 85.74% | 85.74% | 87.41% | 85.43% | 85.43% | 87.86% | 85.13% | 85.58% | 84.80% | 84.95% |

**Protocol B — subject-independent audit (GroupKFold).** This is *not* a reproduction claim: no subject appears in both the training and test folds, which is the stricter test of whether the model generalises to unseen children.

| Metric | Value |
|---|---|
| ACCURACY | 59.21% |
| PRECISION | 59.29% |
| RECALL | 59.21% |
| F1 | 59.25% |

Accuracy drops from **85.81%** (Protocol A) to **59.21%** (Protocol B), a gap of **26.59 pp**. Discuss this gap in the report: it quantifies how much of the segment-level score comes from segments of the same child appearing in both folds.

**mRMR feature-count sweep (paper Fig. 5).** Empirical optimum at **165** features; the paper specifies **171** for this experiment. Plot: `results\plots\DS2_mrmr_accuracy.png`

**Classifier comparison (paper Table V).**

| Classifier | Accuracy | F1 | Time (ms/sample) |
|---|---|---|---|
| DT | 70.04% | 70.07% | 0.516 |
| KNN | 85.81% | 85.79% | 0.162 |
| SVM | 65.29% | 61.48% | 23.630 |
| Ensemble | 80.69% | 80.42% | 34.250 |

Confusion matrix figure: `results\confusion_matrices\cm_DS2.png`

---

## 4. Pipeline as implemented

```
Raw EEG
  -> Preprocessing   DS-1: dataset-provided clean recordings (MODE B)
                     DS-2: Butterworth 0.5-60 Hz -> 50 Hz notch
                           -> Coiflet-3 wavelet -> SURE denoising (L6)
  -> Segmentation    5 s windows, 50% overlap (640 / 320 samples @128 Hz)
  -> SMVMD           DS-1: alpha=2000, DS-2: alpha=1000; tau=0; tol=1e-10
                     variable number of MIMFs (K) per segment
  -> Features        9 per mode per channel:
                     STD, VAR, RMS, IQR, ASSR, AP, MFL, IP, PCC
  -> Energy weighting (paper Eq. 7) collapses the variable K into a
                     fixed vector: DS-1 14x9=126, DS-2 19x9=171
  -> mRMR ranking -> KNN (paper Table II) -> 10-fold CV
```

## 5. Known deviations from the paper

These are the choices the paper leaves unspecified, or where this implementation knowingly differs. The full discussion is in `docs/reproducibility_notes.md`.

| # | Item | Choice made here |
|---|---|---|
| 1 | SMVMD stopping criterion (how K is decided) | Stop when residual energy falls below 1% of the segment energy |
| 2 | DS-1 preprocessing | MODE B: the dataset's own preprocessed recordings (the paper's ICA/ADJUST stage is EEGLAB/MATLAB-only) |
| 3 | SMVMD centre-frequency initialisation | Uniform over [0, 0.5] |
| 4 | mRMR scope | Fitted per fold on training data only — stricter than the paper, which does not say |
| 5 | IP / PCC kernel bandwidth | Silverman's rule |
| 6 | Last segment at the recording boundary | Excluded, which reproduces the paper's 46 segments per DS-1 recording |
| 7 | Bayesian optimisation | Not re-run; the Table II hyperparameters the paper reports are used directly |

## 6. Provenance

| Experiment | Run at | Runtime | Python | NumPy |
|---|---|---|---|---|
| DS1_Rest | 2026-09-20T23:29:13 | 45.6 min | 3.13.3 | 2.4.5 |
| DS1_Music | 2026-09-21T00:10:54 | 34.9 min | 3.13.3 | 2.4.5 |
| DS1_RestMusic | 2026-09-21T02:01:21 | 96.4 min | 3.13.3 | 2.4.5 |
| DS2 | 2026-09-21T03:31:19 | 85.3 min | 3.13.3 | 2.4.5 |

Per-experiment raw numbers: `results/metrics/*.json`. CSV tables: `results/tables/`. Figures: `results/confusion_matrices/`, `results/plots/`.
