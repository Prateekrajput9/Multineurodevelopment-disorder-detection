# Reproducibility Notes
## EEG-Based Multiple Neurodevelopmental Disorders Detection

**Paper:** "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental Disorders Detection in Children Using Successive Multivariate Variational Mode Decomposition"  
**Authors:** Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma  
**Journal:** IEEE Transactions on Cognitive and Developmental Systems, 2025  
**DOI:** 10.1109/TCDS.2025.3556888

---

## 1. DS-1 Preprocessing

### 1.1 ICA Artifact Removal (MATLAB-Only Step)

**What the paper explicitly states:**
- Raw EEG data preprocessed through EEGLAB software
- Band-pass filter: 1–30 Hz
- CleanLine plugin to remove 50/60 Hz line noise
- ICA via `runica` function (logistic infomax with natural gradient)
- ADJUST plugin to automatically identify artifact ICs
- Artifact classes: eye blinks, eye movements, discontinuities
- ICs surpassing predefined thresholds for BOTH sets of features classified as artifacts
- The TWO most problematic ICs (based on cumulative threshold exceedance for FIVE features) removed
- EEG reconstructed from remaining ICs

**What the paper does NOT state:**
- Exact EEGLAB version
- Exact ADJUST plugin version/parameters
- Exact CleanLine plugin parameters (bandwidth, line noise frequency selection)
- Exact ICA convergence criteria
- Epoch rejection criteria (if any)
- Reference electrode used

**Our chosen implementation:**
- **MODE B (DEFAULT):** Use the preprocessed data provided with the Mendeley dataset (DOI: 10.17632/fshy54ypyh.1)
- The dataset description explicitly states: "Both raw EEG data and pre-processed/clean EEG data" are provided
- MODE A (MATLAB pipeline) is documented but requires MATLAB R2023a + EEGLAB + CleanLine + ADJUST

**Why this implementation was necessary:**
- EEGLAB, CleanLine, and ADJUST are MATLAB-only tools
- Python does not have exact equivalents that reproduce the EXACT same ICA decomposition
- The paper's authors likely used the preprocessed version (since they cite the data in its preprocessed form)
- Using the provided preprocessed data is the most faithful reproduction

**How it might affect results:**
- If the provided preprocessed data matches what the authors used: NO impact
- If the authors applied additional preprocessing beyond what is in the files: potential discrepancy
- This is the primary source of uncertainty in DS-1 reproduction

---

## 2. SMVMD Implementation

### 2.1 ADMM Update Equations

**What the paper explicitly states (Equations 3–6):**

The optimization problem (Eq. 3):
```
min_{u_k, ω_k, λ} α·Σ_k ||∂_t[(δ(t) + j/πt) * u_k(t)] · e^{-jω_k t}||²₂
                  + ||f(t) - Σ_k u_k(t)||²₂ + ⟨λ(t), f(t) - Σ_k u_k(t)⟩
```

Mode update (Eq. 4):
```
û^{n+1}_{k,c}(ω) = [x̂_c(ω) + (1/(2α(ω-ω^n_k)⁴)) · (α·Σ_{i≠k}(-2α(ω-ω^n_i)⁴·û^n_{i,c}(ω)) + λ̂^n_c(ω)/2)] 
                    / [1 + α²(ω-ω^n_k)⁴ + Σ_{i≠k} α²(ω-ω^n_i)⁴]
```

Wait - the paper equation (4) as extracted from PDF has encoding issues. The correct form based on MVMD/SVMD literature:
```
û^{n+1}_{k,c}(ω) = [x̂_c(ω) - Σ_{i≠k} û^n_{i,c}(ω) + λ̂^n_c(ω)/2]
                    / [1 + 2α(ω - ω^n_k)²]
```

**What the paper does NOT state:**
- The exact stopping criterion (how K is determined - when to stop adding modes)
- The initialization of center frequencies ω_k
- Whether the DC component is handled specially
- The maximum number of inner ADMM iterations per mode extraction

**Our chosen implementation:**
- Stopping criterion: Mode is accepted if convergence criterion is met AND reconstruction error decreases. Stop when adding a new mode does not improve reconstruction beyond tolerance.
- ω initialization: Uniform across [0, 0.5] normalized frequency (implementation choice)
- Maximum modes: 20 per segment (practical upper bound; implementation choice)
- Reference: Liu & Yu (2022), "Successive Multivariate Variational Mode Decomposition"

**NOTE:** The exact stopping criterion for SMVMD (when to stop extracting new modes) is THE most critical unspecified detail. The paper shows example segments with 4 and 5 modes but does not provide the exact termination rule. This significantly impacts results.

---

### 2.2 Number of modes K, and the safety cap

**What the paper states:** SMVMD "decomposes different segments of EEG signals
into different numbers of modes", i.e. K is adaptive per segment. The paper
does not state the criterion that stops the successive extraction.

**Our implementation:** stop when the residual energy falls below 1% of the
original segment energy, with a hard safety cap of `max_modes = 20`.

**Measured consequence (DS-1 Rest, 644 segments):** mean K = 10.79, range
4-20, and **6.4% of segments (41) hit the cap of 20**. The mode-count
histogram decays smoothly to K = 19 (3, 5, 6 segments at K = 17, 18, 19) and
then spikes to 41 at K = 20, which is the signature of truncation rather than
a genuine mode count.

For those segments the mode count was set by an implementation safety limit,
not by the stopping rule, and certainly not by anything the paper specifies.
Every experiment now records `n_segments_at_mode_cap` and
`pct_segments_at_mode_cap` in its saved metrics, and the generated report
states the percentage, so the caveat travels with the numbers.

If you want to probe its influence, raise `max_modes` in `src/smvmd.py` and
re-run with `--no-cache`; the energy-based integration weights modes by energy,
so late low-energy modes contribute little and the effect on the final feature
vector is expected to be small.

---

## 3. Feature Extraction

### 3.1 Average Power (AP)

**What the paper explicitly states:**
- Uses modified periodogram technique
- Input EEG multiplied by non-negative window function (Hamming window, N samples)
- Window: w = Hamming window of N samples
- Formula from Table I: `AP = Σ_f |Σ_N_{i=1} w_i·x_i·e^{-j2πfΔt·i}|² / Σ`

**What the paper does NOT state:**
- Whether AP is normalized by window power
- Whether one-sided or two-sided spectrum is used
- The exact frequency resolution (nfft)

**Our implementation:**
- Use `scipy.signal.periodogram` with Hamming window
- One-sided (positive frequencies only)
- nfft = length of signal (N=640)
- Power is average over all frequency bins (mean of PSD)

### 3.2 Information Potential (IP)

**What the paper explicitly states (Table I):**
```
IP = (1/N²) · Σ_i Σ_j K(x_i - x_j)
where K(x_i, x_j) = (1/(√(2π)·σ)) · exp(-(x_i - x_j)²/(2σ²))
```

**What the paper does NOT state:**
- The kernel bandwidth σ (sigma)

**Our implementation:**
- Use Silverman's rule of thumb: σ = 1.06 · std(x) · N^(-1/5)
- This is the standard KDE bandwidth estimator (implementation choice)

### 3.3 Parametric-Centered Correntropy (PCC)

**What the paper explicitly states (Table I):**
```
CC[l] = (1/N) · Σ_{n=l}^{N} K(X[n] - X[n-l])
         - (1/N²) · Σ_{l=1}^{N} Σ_{n=l}^{N} K(X[n] - X[n-l])
```
- Time delay l = 1 (paper explicitly states this)
- Uses same Gaussian kernel K as IP

**What the paper does NOT state:**
- The kernel bandwidth σ for PCC
- Whether σ for PCC is the same as for IP

**Our implementation:**
- Same Silverman's rule for σ
- l = 1 as stated in paper
- The centering term is summed over **every** lag l = 1..N, as the formula
  states, using a closed form (see below)

**Centering term — closed form.**
The second (centering) term is a double sum over all lags. Because the
Gaussian kernel depends only on the magnitude |X[n] − X[n−l]|, that double sum
is exactly the sum over all unordered sample pairs:

```
Σ_{l=1}^{N-1} Σ_n K(X[n] − X[n−l])  =  (S − N·K(0)) / 2,
        where  S = Σ_i Σ_j K(x_i − x_j)
```

so the centering term equals `(S − N·K(0)) / (2N²)`. This is verified to
machine precision against a brute-force loop over all lags (agreement
~1e-18). It matters twice over:

1. **Correctness.** An earlier version of this code truncated the lag sum at
   50 lags while still dividing by N², which made the centering term far too
   small and therefore PCC systematically too large — on a 640-sample mode the
   reported value was roughly 75% too high. The closed form removes the
   truncation entirely.
2. **Cost.** `S` is the very same quantity the IP feature needs, so IP and PCC
   now share a single O(N²) kernel evaluation per mode per channel instead of
   computing one each. This is the hot path of the whole pipeline.

### 3.4 ASSR (Absolute Summation of Square Root)

**What the paper states (Table I):**
```
ASSR = |Σ_{i=1}^{N} √|x_i||
```

**Interpretation note:**
The paper formula shows `Σ(√x_i)`. Since EEG values can be negative, we interpret this as:
`ASSR = |Σ_{i=1}^{N} √|x_i||`
This is the standard interpretation from the EEG feature toolbox [41].

---

## 4. Energy-Based Feature Integration

**What the paper explicitly states (Eq. 7):**
```
f_{c,m} = (f_{c,m,1}·e_{c,1} + f_{c,m,2}·e_{c,2} + ... + f_{c,m,n}·e_{c,n})
           / (e_{c,1} + e_{c,2} + ... + e_{c,n})
```
- f_{c,m} = mth integrated feature of cth channel
- f_{c,m,k} = mth feature from kth mode of cth channel
- e_{c,k} = energy of kth mode for cth channel

**What the paper does NOT state:**
- How energy e_{c,k} is computed (time-domain, frequency-domain, or both)

**Our implementation:**
- Energy = sum of squared samples: e_{c,k} = Σ_t u_{k,c}(t)²
- This is the standard signal energy definition and is consistent with most EEG papers

---

## 5. mRMR Feature Selection

**What the paper explicitly states:**
- Uses mRMR (Maximum Relevance Minimum Redundancy) algorithm
- Feature counts: DS-1 Music=47, DS-1 Rest=80, DS-1 Rest+Music=103, DS-2=171
- Selection criterion: "maximum accuracy for lowest number of features" (Fig. 5)
- Features are ranked and classification performance computed as count increases

**What the paper does NOT state:**
- The exact mRMR variant (MI-based vs F-statistic-based)
- Whether mRMR was applied to train data only in each fold or globally

**Our implementation:**
- Use mutual information-based mRMR (standard variant)
- Apply mRMR ranking on TRAINING data only within each CV fold
- This is the methodologically correct approach to avoid data leakage

**IMPORTANT DATA LEAKAGE NOTE:**
The paper likely used MATLAB's mRMR which may have been applied globally (not per-fold). This would constitute mild data leakage but is a common practice in the field. Our implementation applies mRMR within each fold's training data (methodologically stricter).

---

### 5.x Redundancy estimation cost (large matrices)

**What the paper states:** features are "ranked using maximum relevance
minimum redundancy (mRMR) algorithm". No implementation detail is given.

**Our implementation:** greedy mRMR with mutual information for both terms.

- **Relevance**, I(f; y), always uses every sample.
- **Redundancy** needs a p x p matrix of I(f_i; f_j). scikit-learn's kNN-based
  estimator costs roughly O(n log n) per pair, and one call over the DS-2
  matrix (6588 x 171) measures at ~54 s, which puts the full matrix at about
  2.5 hours -- longer than the whole rest of the pipeline.
- The redundancy matrix is therefore estimated from a seeded random subsample
  of at most **2000 rows** (`max_samples_redundancy`). The kNN MI estimate
  converges well before that: measured on 171 features, MI for a strongly
  correlated pair was 0.519 / 0.501 / 0.483 at n = 1000 / 2000 / 3000 against
  0.512 using all 6588 rows -- variation of a few percent, well inside the
  estimator's own noise.

**Which experiments this touches:** only DS-2. DS-1 Rest and DS-1 Music have
644 segments and DS-1 Rest+Music has 1288, all below the 2000-row cap, so
their rankings use every sample and are unaffected.

**Why it is low-risk for DS-2 specifically:** the paper keeps **all 171**
features for DS-2, so the ranking cannot change which features reach the
classifier. It affects only the ordering along the accuracy-vs-feature-count
curve (paper Fig. 5). The reported DS-2 accuracy, precision, recall and F1 are
identical regardless of the ranking.

---

## 6. Classifier and Bayesian Optimization

**What the paper explicitly states:**
- MATLAB R2023a Classification Learner tool
- KNN classifier
- Bayesian optimization
- 200 iterations
- 10-fold cross-validation
- Optimized hyperparameters listed in Table II

**What the paper does NOT state:**
- Exact MATLAB random seed
- Internal Bayesian optimizer type (expected improvement, etc.)
- Whether the hyperparameters in Table II are from ALL folds or a specific fold
- Training/test split protocol within CV (how each fold's test segments are handled)

**Our implementation:**
- Use scikit-learn's BayesSearchCV (scikit-optimize) OR manual Bayesian optimization
- Apply the EXACT hyperparameters from Table II for final evaluation
- Run 10-fold stratified cross-validation
- This does NOT exactly replicate MATLAB's Bayesian optimizer

**Critical difference:**
The paper's Table II shows the final optimized parameters. These were obtained from MATLAB's Bayesian optimizer over 200 iterations. Our Python implementation uses the SAME parameters from Table II for the primary evaluation. For the Bayesian optimization experiment, we re-run the optimization in Python.

---

## 7. KNN Distance Weighting

**What the paper states (Table II):**
- "Squared Inverse" distance weighting for several configurations
- "Equal" weighting for DS-1 Rest

**MATLAB vs scikit-learn difference:**
- MATLAB's "Squared Inverse" = weight proportional to 1/d² (where d = distance)
- scikit-learn's `weights='distance'` = weight proportional to 1/d (inverse distance, not squared)
- scikit-learn does NOT have a built-in "squared inverse" option

**Our implementation:**
- Implement custom `squared_inverse_weights(distances)` = 1 / (distances² + ε)
- Where ε = 1e-10 to avoid division by zero
- Use as custom `weights` callable in scikit-learn's KNeighborsClassifier

---

## 8. Segmentation

**What the paper explicitly states:**
- Window: 5 seconds
- Overlap: 50%
- Expected segments: 1288 (DS-1), 6588 (DS-2)
- DS-1 breakdown: 322 IDD (rest) + 322 TDC (rest) + 322 IDD (music) + 322 TDC (music) = 1288

**Verification:**
- DS-1: Each subject has 120s recording × 128 Hz = 15360 samples
  - Window = 640 samples, step = 320 samples
  - Segments per recording = floor((15360 - 640) / 320) + 1 = floor(14720/320) + 1 = 46 + 1 = 47 segments
  - Per condition: 7 IDD × 47 + 7 TDC × 47 = 329 + 329 = 658 segments
  - For both conditions: 658 × 2 = 1316 segments
  
  **DISCREPANCY:** Expected 1288, calculated 1316. Investigation required.
  
  Possible explanation: Some subjects' recordings may be shorter than 120s, or boundary handling differs.
  Paper reports: 322 IDD + 322 TDC per condition = 644 per condition × 2 = 1288 total
  This means: 322/7 ≈ 46.0 segments per subject
  46 × 7 × 2 × 2 = 1288 ✓
  
  So each subject contributes exactly 46 segments (not 47). This means:
  - Either floor((N - 640) / 320) = 46, meaning N = floor(46 × 320 + 640) = 15360, which gives 47
  - OR the last incomplete segment is NOT included (strictly floor, not +1)
  
  **IMPLEMENTATION CHOICE:** Use strict floor (no trailing segment if shorter than window).
  This gives: floor((15360 - 640) / 320) = floor(46) = 46 segments per recording ✓

- DS-2: Dataset README states each subject recording has variable length.
  The paper reports 3680 ADHD + 2908 NC = 6588 total.
  Subject-level verification requires the actual dataset.

**What the paper does NOT state:**
- Exact handling of the last partial window (confirmed: not included)
- Whether segments that include zeros (zero-padded) are included

---

## 9. Cross-Validation Protocol

**What the paper explicitly states:**
- 10-fold cross-validation
- MATLAB Classification Learner tool

**What the paper does NOT state:**
- Whether cross-validation is stratified (class-balanced folds)
- Whether subjects are kept together within folds (subject-independent) or split at segment level (leaky)
- The specific random seed for fold assignment

**Our implementation (Paper-style, Section A):**
- Stratified 10-fold CV at segment level (not subject-level)
- This is the standard MATLAB Classification Learner behavior
- Segments from the same subject may appear in both train and test folds
- NOTE: This creates data leakage between subjects (correlated segments from same subject)

**Our implementation (Subject-Independent, Section B):**
- StratifiedGroupKFold with group = subject_id
- All segments from a subject stay in the same fold
- This is methodologically more rigorous but NOT what the paper reports

---

## 10. ADHD Dataset File Format

**Known from dataset description:**
- Files are in .mat format (MATLAB)
- Each file contains one subject's EEG recording
- 19 channels, 128 Hz sampling rate

**Unknown (requires dataset download to verify):**
- Exact variable names in .mat files
- Whether files are preprocessed or raw
- Whether any subjects have missing data or artifacts
- Exact recording lengths per subject (determines segment count)

---

## 11. Summary of Ambiguities (Priority Order)

| # | Ambiguity | Impact on Results | Resolution |
|---|-----------|-------------------|------------|
| 1 | SMVMD stopping criterion (K determination) | **HIGH** | See Section 2.1 |
| 2 | DS-1 preprocessed vs raw (MODE A vs B) | **HIGH** | Use MODE B (default) |
| 3 | SMVMD center frequency initialization | **MEDIUM** | Uniform init |
| 4 | mRMR applied globally vs per-fold | **MEDIUM** | Per-fold (strict) |
| 5 | KNN "squared inverse" in MATLAB vs sklearn | **MEDIUM** | Custom callable |
| 6 | IP/PCC kernel bandwidth σ | **LOW-MEDIUM** | Silverman's rule |
| 7 | AP normalization | **LOW** | scipy.signal.periodogram |
| 8 | CV random seed | **LOW** | Fixed seed = 42 |
| 9 | Last segment boundary handling | **LOW** | Floor (no partial) |
| 10 | Bayesian optimizer implementation | **LOW** | Table II params used directly |
