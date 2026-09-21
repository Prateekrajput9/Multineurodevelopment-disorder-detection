# Project Structure

```
btpnew/
├── README.md                 Project overview, headline results, quick start
├── STRUCTURE.md              This file
├── run_guide.md              Step-by-step workflow, run times, outputs
├── requirements.txt          Python dependencies
├── paper_extracted.txt       Plain-text extract of the replicated paper (reference only)
├── BTP_Latex_template.zip    IIT Indore BTP LaTeX template (source of the report title page)
│
├── configs/
│   └── config.yaml           Every pipeline parameter (paths, SMVMD, windows, mRMR, KNN)
│
├── src/                      Core library
│   ├── data_loader.py        DS-1 (.mat) and DS-2 (read from ZIPs) loaders
│   ├── preprocessing.py      DS-1 mode A/B; DS-2 Butterworth, notch, Coiflet-3 + SURE
│   ├── segmentation.py       5 s windows, 50% overlap, subject ID kept per window
│   ├── smvmd.py              SMVMD (successive ADMM), parallel over windows
│   ├── features.py           9 features per mode; IP and PCC share one kernel sum
│   ├── energy_integration.py Energy-weighted integration across modes (paper Eq. 7)
│   ├── mrmr.py               mRMR ranking and accuracy-vs-feature-count sweep
│   ├── knn_model.py          KNN (squared-inverse weights), segment/subject-wise CV
│   ├── evaluation.py         Metrics, confusion matrices, report helpers
│   ├── pipeline.py           Shared driver for the three DS-1 experiments
│   ├── results_io.py         Result bundles, feature cache (with fingerprint), CSVs
│   └── utils.py              Logging, config loading, plotting
│
├── experiments/              Runnable entry points (python -m experiments.<name>)
│   ├── ds1_rest.py           DS-1 IDD vs TDC, rest
│   ├── ds1_music.py          DS-1 IDD vs TDC, music
│   ├── ds1_rest_music.py     DS-1 IDD vs TDC, rest + music
│   ├── ds2_adhd.py           DS-2 ADHD vs NC (+ classifier comparison)
│   ├── stage1_analysis.py    Subject-wise protocols, 0% overlap, raw baseline,
│   │                         t-SNE / child-ID probe, mode counts, per-child accuracy
│   ├── make_report.py        Markdown summary of the four reproductions
│   └── subject_independent_audit.py   Standalone older audit (superseded by stage1)
│
├── docs/
│   ├── reproducibility_notes.md    Every ambiguity in the paper and the choice made
│   └── paper_vs_implementation.md  Side-by-side of paper and code
│
├── report/                   Mid-semester report and project record
│   ├── main.tex              LaTeX report (IITI title page, 3 references)
│   ├── generated/            numbers.tex (one macro per result) + tab_*.tex
│   ├── figures/              Figures used by main.tex, plus IITI.png
│   ├── mse_report_overleaf.zip   Upload this to Overleaf and compile main.tex
│   ├── build_tables.py       results_summary.json -> generated/*.tex
│   ├── check_report.py       Checks macros, citations, environments, typed digits
│   ├── build_project_record.py   -> Project_Record.pdf
│   ├── Project_Record.pdf    Full record of work, results and plan (10 pages)
│   └── README.md             How to compile, placeholders, what to cut if too long
│
├── results/                  All generated outputs
│   ├── metrics/              <experiment>.json: every metric + settings + timings
│   ├── features/             Cached feature matrices:
│   │                           <exp>_features.npz      SMVMD features
│   │                           RAW_<exp>_features.npz  raw-channel baseline
│   ├── stage1/
│   │   ├── results_summary.json   Single source of every reported number
│   │   ├── results_summary.md     Same, human-readable
│   │   ├── figures/          t-SNE, mode histogram, per-child accuracy (PDF + PNG)
│   │   └── tables/           per_subject_*.csv (27 files)
│   ├── confusion_matrices/   cm_<exp>.png
│   ├── plots/                <exp>_mrmr_accuracy.png (accuracy vs feature count)
│   ├── tables/               fold accuracies, mRMR curves, DS-2 classifier comparison
│   ├── smvmd/                DS-2 single-window SMVMD diagnostic plot
│   ├── logs/                 One log per experiment and stage-1 part
│   └── final_report.md       Output of experiments/make_report.py
│
├── data/
│   ├── metadata/ds2_segments.csv   Per-window DS-2 metadata (needed by the 0% overlap analysis)
│   └── raw/ds2/              Unzipped copy of adhd/*.zip (byte-identical; not read by the code)
│
├── idd/                      DS-1 as downloaded (Mendeley)
│   ├── Data/CleanData/       Cleaned .mat recordings, the input the pipeline uses
│   ├── Data/RawData/         Raw .set/.fdt recordings (only for re-running the paper's cleaning)
│   ├── Data/QualitativeData.xlsx
│   └── Pipeline/             Dataset authors' MATLAB preprocessing code
│
└── adhd/                     DS-2 as downloaded (IEEE DataPort)
    ├── ADHD_part1.zip, ADHD_part2.zip, Control_part1.zip, Control_part2.zip
    └── Channel_Labels.zip, Standard-10-20-Cap19new.zip   (documentation, not EEG)
```

## How the pieces connect

1. **Reproduction.** The four `experiments/ds*.py` scripts each write
   `results/metrics/<exp>.json` and cache their features in `results/features/`.
2. **Analysis.** `experiments/stage1_analysis.py` reads those caches and writes
   everything into `results/stage1/`. Only its raw-channel part reads EEG again.
3. **Report.** `report/build_tables.py` turns `results_summary.json` into
   LaTeX macros and tables for `report/main.tex`, and
   `report/build_project_record.py` builds the PDF record from the same file.
   No result number is typed by hand anywhere.

The three DS-1 scripts are thin wrappers over `src/pipeline.py`; they differ
only in the recording condition and in which row of the paper's Table II
supplies the KNN settings.

## Safe to delete, recreated automatically

- `results/features/*.npz`: the next run recomputes them (slow: up to about
  1.5 h per experiment).
- `results/logs/*`
- Any `__pycache__/` folder

## Do not delete

- `results/stage1/results_summary.json`: the report and the project record are
  built from it.
- `data/metadata/ds2_segments.csv`
- `idd/`
- `adhd/`
