# EEG-Based Multiple Neurodevelopmental Disorders Detection

B.Tech. project, IIT Indore (August – November 2026).

The project reproduces, in Python, the pipeline of

> U. Chandela, K. N. Faisal, R. R. Sharma, "Electroencephalogram-Based Unified
> Approach for Multiple Neurodevelopmental Disorders Detection in Children Using
> Successive Multivariate Variational Mode Decomposition," *IEEE Trans. Cogn.
> Develop. Syst.*, vol. 17, no. 6, pp. 1350–1359, Dec. 2025.

It then evaluates that pipeline under **subject-independent** protocols: every
window of a child is kept on one side of the train/test split. The goal is to
measure how well the method works on children it has never seen, and what it
actually learns.

> **Research code, not a clinical tool.** No result here may be used for
> diagnosis.

---

## Where things stand (mid-semester, 25 Sep 2026)

| Setting | Segment-wise 10-fold (paper protocol) | Subject-wise (new children) | Children correct |
|---|---|---|---|
| DS-1 IDD, rest | 99.69% | 62.73% | 10/14 |
| DS-1 IDD, music | 96.43% | 43.01% | 8/14 |
| DS-1 IDD, rest + music | 99.77% | 62.34% | 9/14 |
| DS-2 ADHD | 85.81% | 57.54% | 74/121 |

The features largely identify **which child** a window came from: a
nearest-neighbour probe identifies the child with 98.91% accuracy on DS-1 rest
(chance 7.14%). Segment-wise splits therefore reward recognising known
children. Every number above is in `results/stage1/results_summary.json`.

A 10-page account of everything done, all results, and the plan is in
[`report/Project_Record.pdf`](report/Project_Record.pdf). The mid-semester
report is in [`report/`](report/).

---

## Datasets

| | DS-1 (IDD) | DS-2 (ADHD) |
|---|---|---|
| Source | Sareen et al., *Data in Brief*, 2020 (Mendeley, doi 10.17632/fshy54ypyh.1) | Motie Nasrabadi et al., IEEE DataPort, 2020 (doi 10.21227/rzfh-zn36) |
| Children | 7 IDD + 7 typically developing (TDC) | 61 ADHD + 60 controls (NC) |
| Recording | 14-channel EMOTIV EPOC+, 128 Hz, 2 min rest + 2 min music | 19-channel 10–20 EEG, 128 Hz, visual attention task |
| In this repo | `idd/Data/` (cleaned `.mat` files are used) | `adhd/*.zip` (read directly from the ZIPs) |

DS-2 labels come from **which ZIP** a subject is in, not from the file name.

---

## Pipeline

```
EEG -> preprocessing -> 5 s windows, 50% overlap -> SMVMD (variable K modes)
    -> 9 features per mode per channel -> energy-weighted integration (Eq. 7)
    -> mRMR ranking -> KNN (paper Table II settings) -> cross-validation
```

| | DS-1 | DS-2 |
|---|---|---|
| Preprocessing | dataset's own cleaned recordings | Butterworth 0.5–60 Hz, 50 Hz notch, Coiflet-3 + SURE (level 6) |
| Windows | 644 per condition, 1288 rest+music | 6588 (3680 ADHD / 2908 NC) |
| SMVMD | α = 2000, τ = 0, tol = 1e-10 | α = 1000, τ = 0, tol = 1e-10 |
| Feature vector | 14 × 9 = 126 | 19 × 9 = 171 |
| mRMR keeps | 80 rest, 47 music, 103 rest+music | all 171 |

Features: STD, VAR, RMS, IQR, ASSR, average power, maximum fractal length,
information potential, parametric-centred correntropy.

Standardisation, mRMR and KNN are always fitted inside each training fold.

---

## Quick start

```powershell
cd C:\Users\Prateek\OneDrive\Desktop\btpnew
pip install -r requirements.txt

python -m experiments.ds1_rest            # reproduce each setting
python -m experiments.ds1_music
python -m experiments.ds1_rest_music
python -m experiments.ds2_adhd
python -m experiments.stage1_analysis     # subject-wise protocols + diagnostics

python report/build_tables.py             # numbers/tables for the LaTeX report
python report/build_project_record.py     # regenerate Project_Record.pdf
```

Results are cached, so re-running an experiment takes seconds. The full
workflow, run times and outputs are in [`run_guide.md`](run_guide.md); the
folder layout is in [`STRUCTURE.md`](STRUCTURE.md).

---

## Evaluation protocols

| Protocol | Split | Question it answers |
|---|---|---|
| Segment-wise (paper) | Stratified 10-fold over windows | How well are windows from *already seen* children classified? |
| Subject-wise (this project) | Leave-one-subject-out (DS-1), grouped 10-fold (DS-2) | How well does it work on a *new* child? |

Every subject-wise run also reports child-level accuracy, by majority vote
over each held-out child's windows.

---

## Known deviations and caveats

Full details: [`docs/reproducibility_notes.md`](docs/reproducibility_notes.md).

1. DS-1 uses the dataset's cleaned recordings; the paper's EEGLAB ICA/ADJUST
   step is MATLAB-only and was not re-run.
2. The paper does not say how SMVMD chooses K. Here it stops at < 1% residual
   energy or at a 20-mode safety cap; 5.7% of DS-1 windows hit the cap.
3. KNN settings are taken from the paper's Table II; its Bayesian optimisation
   was not re-run.
4. The PCC feature follows the paper's formula over **all** lags, in closed
   form. An earlier version truncated this at 50 lags.
5. For DS-2 only, the mRMR redundancy matrix is estimated from a seeded
   2000-window sample. DS-2 keeps all 171 features, so accuracy is unaffected.
6. In DS-2 the ADHD children were taking methylphenidate and the controls were
   not, so the dataset cannot separate ADHD from its treatment.

---

## Next steps (to end of November 2026)

Nested subject-wise cross-validation → ablation ladder (raw → fixed bands →
VMD → MVMD → SMVMD) → interpretability (SHAP, topographic maps, theta/beta
baseline) → cross-dataset analysis on the 10 shared electrodes → channel
reduction → final report. Dates are in `run_guide.md` and the project record.
