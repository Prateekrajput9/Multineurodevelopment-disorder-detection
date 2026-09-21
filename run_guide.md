# Run Guide

Everything runs from the project root, one command at a time. Each step saves
its own outputs, so you can stop between any two steps without losing work.

```powershell
cd C:\Users\Prateek\OneDrive\Desktop\btpnew
```

Run times below were measured on a 12-core machine. Run one experiment at a
time: each one already uses all the cores.

---

## Step 0 — Setup (once)

```powershell
pip install -r requirements.txt
pip install mne seaborn h5py openpyxl
```

The data must be in place: `idd/Data/CleanData/...` for DS-1 and
`adhd/*.zip` for DS-2 (see `STRUCTURE.md`).

---

## Step 1 — Reproduce the four settings

| Command | What it does | First run |
|---|---|---|
| `python -m experiments.ds1_rest` | DS-1 IDD vs TDC, rest | 46 min |
| `python -m experiments.ds1_music` | DS-1 IDD vs TDC, music | 35 min |
| `python -m experiments.ds1_rest_music` | DS-1, rest + music pooled | 96 min |
| `python -m experiments.ds2_adhd` | DS-2 ADHD vs NC, plus classifier comparison | 85 min |

Each run:
- writes `results/metrics/<experiment>.json`, which holds every metric, the
  settings used and stage timings;
- writes a confusion matrix, the accuracy-vs-feature-count plot and CSV tables;
- caches its feature matrix in `results/features/<experiment>_features.npz`.

**Re-runs are fast.** When the cache exists, the run skips preprocessing,
SMVMD and feature extraction. The cache stores a fingerprint of every
parameter that affects the features (SMVMD α/τ/tol, window, overlap,
preprocessing). If you change any of them in `configs/config.yaml`, the cache
is refused automatically and the features are recomputed.

Useful flags:

```powershell
python -m experiments.ds1_rest --no-cache       # force full recomputation
python -m experiments.ds2_adhd --skip-checks    # skip the single-window sanity checks
```

Sanity numbers to look for in the log: 644 windows (rest or music),
1288 (rest + music), and 6588 for DS-2 (3680 ADHD / 2908 NC).

---

## Step 2 — Subject-independent evaluation and diagnostics

Needs the four feature caches from Step 1.

```powershell
python -m experiments.stage1_analysis                 # all parts, in order
python -m experiments.stage1_analysis --part eval     # or one part at a time
```

| Part | What it produces | Run time |
|---|---|---|
| `reproduction` | Copies the Step 1 results into the summary | seconds |
| `modes` | Histogram of SMVMD modes per window, by dataset and class | seconds |
| `embed` | t-SNE figure, silhouette scores, child-identification probe | 1–5 min |
| `eval` | Segment-wise vs subject-wise (LOSO for DS-1, grouped 10-fold for DS-2), 50% and 0% overlap, paper-order mRMR, majority vote per child, per-child accuracy | 64 min |
| `raw` | Raw-channel baseline: same 9 features without SMVMD, both protocols | ≈1 h 45 min the first time (features then cached) |
| `md` | Rewrites `results_summary.md` only | seconds |

All numbers go into **`results/stage1/results_summary.json`**, the single
source for the report and the project record. Figures go to
`results/stage1/figures/` and per-child tables to `results/stage1/tables/`.

The 0%-overlap results need no extra computation. The window (640 samples) is
exactly twice the step (320), so the non-overlapping windows are the
even-numbered 50%-overlap windows, taken from the cache.

---

## Step 3 — Mid-semester report (LaTeX)

```powershell
python report/build_tables.py     # summary -> report/generated/numbers.tex + tab_*.tex
python report/check_report.py     # macros, citations, environments, typed digits
```

Then upload **`report/mse_report_overleaf.zip`** to Overleaf (New Project →
Upload Project) and compile `main.tex` with pdfLaTeX. Every result in
`main.tex` is a macro from `generated/numbers.tex`, so do not type numbers into
it. After changing any result, re-run `build_tables.py` and re-upload the
`generated/` files.

Before submitting, fill in the title-page placeholders and check that the body
is at most 7 pages (details in `report/README.md`).

---

## Step 4 — Project record (PDF)

```powershell
python report/build_project_record.py    # -> report/Project_Record.pdf
```

A 10-page record of everything done, all results, caveats, likely viva
questions, the plan, and a plain-language summary. All numbers are read from the
results files at build time. To fill in your name and supervisor, edit the
cover table near the top of the script.

---

## Optional

```powershell
python -m experiments.make_report    # results/final_report.md, a Markdown summary of Step 1
```

---

## Timeline

| Period | Work |
|---|---|
| August – 25 Sep 2026 | **Done:** reproduction, subject-wise evaluation and diagnostics, mid-semester report. Mid-semester evaluation on 25 September. |
| 28 Sep – 11 Oct | Nested subject-wise cross-validation (tuning inside each outer fold), bootstrap confidence intervals over children |
| 12 – 25 Oct | Ablation ladder: raw → fixed bands → VMD → MVMD → SMVMD; sensitivity to the SMVMD stopping rule |
| 26 Oct – 8 Nov | Interpretability: permutation importance, SHAP, channel maps, theta/beta baseline |
| 9 – 15 Nov | Cross-dataset analysis on the 10 shared electrodes (F7, F3, F4, F8, T7, T8, P7, P8, O1, O2) |
| 16 – 22 Nov | Channel reduction under subject-wise evaluation |
| 23 – 30 Nov | Final report; final evaluation and submission |

---

## Troubleshooting

- **`Channel_Labels.zip` / `Standard-10-20-Cap19new.zip` warnings** when
  loading DS-2 are expected: these are documentation, not EEG data.
- **"Feature cache IGNORED"** means a parameter changed since the cache was
  written. The run recomputes the features; this is intended.
- **A run looks frozen:** feature extraction is the slow stage. Progress is
  logged to the console and to `results/logs/<experiment>.log`.
- **SMVMD reconstruction error is not zero:** intended. The paper sets τ = 0 to
  avoid fully reconstructing noisy input.
