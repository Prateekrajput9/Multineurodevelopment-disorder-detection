"""
make_report.py - Consolidate every saved experiment into one report.

Run the experiments one at a time, in any order, then build the write-up:

    python -m experiments.ds1_rest
    python -m experiments.ds1_music
    python -m experiments.ds1_rest_music
    python -m experiments.ds2_adhd
    python -m experiments.make_report

The report is assembled purely from results/metrics/*.json, so it never
re-runs anything and it is safe to rebuild after each individual experiment
finishes. Experiments that have not been run yet are listed as outstanding
rather than silently omitted.

Output:
    results/final_report.md
    results/tables/summary_all_experiments.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.results_io import (EXPERIMENT_ORDER, EXPERIMENT_TITLES, RESULTS_ROOT,
                            ensure_dirs, load_all_experiments, save_table)
from src.utils import setup_logger

logger = setup_logger('MakeReport', 'results/logs/make_report.log')

PAPER_HEADLINE = {
    'DS1_Rest': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
    'DS1_Music': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
    'DS1_RestMusic': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
    'DS2': {'accuracy': 0.9917, 'precision': 0.9904,
            'recall': 0.9907, 'f1': 0.9905},
}

METRIC_ORDER = ['accuracy', 'precision', 'recall', 'f1',
                'mcc', 'kappa', 'gdr', 'gmean']


def _pct(v):
    return '—' if v is None else f'{v * 100:.2f}%'


def _num(v, nd=4):
    return '—' if v is None else f'{v:.{nd}f}'


def _fmt_metric(name, value):
    """GDR is reported as a percentage-like figure; MCC/kappa as coefficients."""
    if value is None:
        return '—'
    if name in ('mcc', 'kappa', 'gmean'):
        return _num(value)
    if name == 'gdr':
        return f'{value:.3f}'
    return _pct(value)


def _confusion_block(cm, class_names):
    if cm is None:
        return []
    cm = np.array(cm)
    if cm.ndim != 2:
        return []
    names = class_names or [f'Class {i}' for i in range(cm.shape[0])]
    lines = ['', '**Confusion matrix** (rows = true, columns = predicted)', '']
    lines.append('| | ' + ' | '.join(f'pred {n}' for n in names) + ' |')
    lines.append('|---|' + '---|' * len(names))
    for i, row in enumerate(cm):
        lines.append(f'| **true {names[i]}** | ' +
                     ' | '.join(str(int(v)) for v in row) + ' |')
    return lines


def build_report(data):
    L = []
    A = L.append

    A('# EEG-Based Multiple Neurodevelopmental Disorders Detection')
    A('## Reproduction Results')
    A('')
    A('Reproduction of: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma, ')
    A('*"Electroencephalogram-Based Unified Approach for Multiple '
      'Neurodevelopmental Disorders Detection in Children Using Successive '
      'Multivariate Variational Mode Decomposition"*, ')
    A('IEEE Transactions on Cognitive and Developmental Systems, 2025.')
    A('')
    A('> **Research reproduction for academic study. Not a clinical '
      'diagnostic tool.** All figures below are EEG-based classification '
      'research results and cannot be used for medical diagnosis.')
    A('')

    done = [k for k in EXPERIMENT_ORDER if k in data]
    missing = [k for k in EXPERIMENT_ORDER if k not in data]

    A(f'Experiments included: **{len(done)} of {len(EXPERIMENT_ORDER)}**.')
    if missing:
        A('')
        A('> **Outstanding experiments** (not yet run, so not reported): ' +
          ', '.join(f'`{m}`' for m in missing) + '.')
    A('')
    A('---')
    A('')

    # ── Summary table ───────────────────────────────────────────────────────
    A('## 1. Summary — reproduction vs. paper')
    A('')
    A('| Experiment | Segments | Features used | Accuracy (ours) | '
      'Accuracy (paper) | Δ | F1 (ours) | F1 (paper) |')
    A('|---|---|---|---|---|---|---|---|')

    summary_rows = []
    for key in EXPERIMENT_ORDER:
        if key not in data:
            continue
        d = data[key]
        m = d.get('metrics_paper_protocol', {})
        ds = d.get('dataset', {})
        paper = PAPER_HEADLINE.get(key, {})
        acc, f1 = m.get('accuracy'), m.get('f1')
        p_acc, p_f1 = paper.get('accuracy'), paper.get('f1')
        delta = (acc - p_acc) if (acc is not None and p_acc is not None) else None
        A(f"| {EXPERIMENT_TITLES.get(key, key)} | {ds.get('n_segments', '—')} | "
          f"{ds.get('n_features_used', '—')} / {ds.get('n_features_total', '—')} | "
          f"{_pct(acc)} | {_pct(p_acc)} | "
          f"{('%+.2f pp' % (delta * 100)) if delta is not None else '—'} | "
          f"{_pct(f1)} | {_pct(p_f1)} |")
        summary_rows.append({
            'experiment': key,
            'title': EXPERIMENT_TITLES.get(key, key),
            'n_segments': ds.get('n_segments'),
            'n_features_used': ds.get('n_features_used'),
            'accuracy_ours': acc,
            'accuracy_paper': p_acc,
            'f1_ours': f1,
            'f1_paper': p_f1,
            'runtime_sec': d.get('runtime_sec'),
        })
    A('')
    A('Δ is our accuracy minus the paper\'s, in percentage points.')
    A('')

    # ── KNN hyperparameters (paper Table II) ────────────────────────────────
    A('## 2. Classifier configuration (paper Table II)')
    A('')
    A('| Experiment | k | Distance | Weighting | Standardize |')
    A('|---|---|---|---|---|')
    for key in EXPERIMENT_ORDER:
        if key not in data:
            continue
        kp = data[key].get('knn_params', {})
        A(f"| {EXPERIMENT_TITLES.get(key, key)} | {kp.get('n_neighbors', '—')} | "
          f"{kp.get('distance', '—')} | {kp.get('weights', '—')} | "
          f"{kp.get('standardize', '—')} |")
    A('')
    A('---')
    A('')

    # ── Per-experiment detail ───────────────────────────────────────────────
    A('## 3. Detailed results')
    A('')
    for idx, key in enumerate([k for k in EXPERIMENT_ORDER if k in data], 1):
        d = data[key]
        m = d.get('metrics_paper_protocol', {})
        ds = d.get('dataset', {})
        A(f'### 3.{idx} {EXPERIMENT_TITLES.get(key, key)}')
        A('')
        A(f"- Segments: **{ds.get('n_segments', '—')}** from "
          f"**{ds.get('n_subjects', '—')}** subjects")
        A(f"- Feature vector: {ds.get('n_features_total', '—')}-dimensional, "
          f"mRMR keeps **{ds.get('n_features_used', '—')}**")
        ms = d.get('mode_stats') or {}
        if ms.get('mean_modes_per_segment') is not None:
            A(f"- SMVMD modes per segment: mean "
              f"**{ms['mean_modes_per_segment']:.2f}** "
              f"(range {ms.get('min_modes')}–{ms.get('max_modes')}) — "
              f"variable K, resolved by energy-based integration")
            pct_cap = ms.get('pct_segments_at_mode_cap')
            if pct_cap:
                A(f"- **{pct_cap:.1f}%** of segments "
                  f"({ms.get('n_segments_at_mode_cap')}) reached the "
                  f"`max_modes={ms.get('mode_cap')}` safety cap, so for those "
                  f"the mode count was set by that cap rather than by the "
                  f"residual-energy stopping rule. The paper does not state "
                  f"how it decides K, so treat this as an implementation "
                  f"caveat worth naming in the write-up.")
        if d.get('runtime_sec'):
            A(f"- Run time: {d['runtime_sec'] / 60:.1f} min")
        A(f"- Run at: {d.get('run_at', '—')}")
        A('')

        # Metrics vs paper
        paper = PAPER_HEADLINE.get(key, {})
        A('**Protocol A — paper-style 10-fold stratified cross-validation**')
        A('')
        A('| Metric | Ours | Paper | Δ |')
        A('|---|---|---|---|')
        for name in METRIC_ORDER:
            if name not in m:
                continue
            ours = m.get(name)
            pval = paper.get(name)
            if pval is not None and ours is not None:
                delta = f'{(ours - pval) * 100:+.2f} pp'
            else:
                delta = '—'
            A(f'| {name.upper()} | {_fmt_metric(name, ours)} | '
              f'{_fmt_metric(name, pval) if pval is not None else "not reported"} | '
              f'{delta} |')
        A('')

        for line in _confusion_block(m.get('confusion_matrix'),
                                     d.get('class_names')):
            A(line)
        A('')

        folds = d.get('fold_accuracies')
        if folds:
            arr = np.array(folds, dtype=float)
            A(f'Per-fold accuracy: mean **{arr.mean() * 100:.2f}%**, '
              f'std {arr.std() * 100:.2f} pp, '
              f'min {arr.min() * 100:.2f}%, max {arr.max() * 100:.2f}%.')
            A('')
            A('| Fold | ' + ' | '.join(str(i + 1) for i in range(len(arr))) + ' |')
            A('|---|' + '---|' * len(arr))
            A('| Accuracy | ' +
              ' | '.join(f'{a * 100:.2f}%' for a in arr) + ' |')
            A('')

        si = d.get('subject_independent')
        if si:
            A('**Protocol B — subject-independent audit (GroupKFold).** '
              'This is *not* a reproduction claim: no subject appears in both '
              'the training and test folds, which is the stricter test of '
              'whether the model generalises to unseen children.')
            A('')
            A('| Metric | Value |')
            A('|---|---|')
            for name in METRIC_ORDER:
                if name in si:
                    A(f'| {name.upper()} | {_fmt_metric(name, si[name])} |')
            A('')
            acc_a = m.get('accuracy')
            acc_b = si.get('accuracy')
            if acc_a is not None and acc_b is not None:
                A(f'Accuracy drops from **{_pct(acc_a)}** (Protocol A) to '
                  f'**{_pct(acc_b)}** (Protocol B), a gap of '
                  f'**{(acc_a - acc_b) * 100:.2f} pp**. Discuss this gap in '
                  f'the report: it quantifies how much of the segment-level '
                  f'score comes from segments of the same child appearing in '
                  f'both folds.')
                A('')

        curve = d.get('mrmr_curve')
        if curve:
            A(f"**mRMR feature-count sweep (paper Fig. 5).** Empirical optimum "
              f"at **{curve.get('optimal_n_empirical')}** features; the paper "
              f"specifies **{curve.get('n_features_paper')}** for this "
              f"experiment. Plot: `{curve.get('plot')}`")
            A('')

        cc = d.get('classifier_comparison')
        if cc:
            A('**Classifier comparison (paper Table V).**')
            A('')
            A('| Classifier | Accuracy | F1 | Time (ms/sample) |')
            A('|---|---|---|---|')
            for name, r in cc.items():
                A(f"| {name} | {_pct(r.get('accuracy'))} | "
                  f"{_pct(r.get('f1'))} | "
                  f"{_num(r.get('time_ms_per_sample'), 3)} |")
            A('')

        cm_plot = (d.get('extra') or {}).get('confusion_matrix_plot')
        if cm_plot:
            A(f'Confusion matrix figure: `{cm_plot}`')
            A('')
        A('---')
        A('')

    # ── Method summary ──────────────────────────────────────────────────────
    A('## 4. Pipeline as implemented')
    A('')
    A('```')
    A('Raw EEG')
    A('  -> Preprocessing   DS-1: dataset-provided clean recordings (MODE B)')
    A('                     DS-2: Butterworth 0.5-60 Hz -> 50 Hz notch')
    A('                           -> Coiflet-3 wavelet -> SURE denoising (L6)')
    A('  -> Segmentation    5 s windows, 50% overlap (640 / 320 samples @128 Hz)')
    A('  -> SMVMD           DS-1: alpha=2000, DS-2: alpha=1000; tau=0; tol=1e-10')
    A('                     variable number of MIMFs (K) per segment')
    A('  -> Features        9 per mode per channel:')
    A('                     STD, VAR, RMS, IQR, ASSR, AP, MFL, IP, PCC')
    A('  -> Energy weighting (paper Eq. 7) collapses the variable K into a')
    A('                     fixed vector: DS-1 14x9=126, DS-2 19x9=171')
    A('  -> mRMR ranking -> KNN (paper Table II) -> 10-fold CV')
    A('```')
    A('')

    A('## 5. Known deviations from the paper')
    A('')
    A('These are the choices the paper leaves unspecified, or where this '
      'implementation knowingly differs. The full discussion is in '
      '`docs/reproducibility_notes.md`.')
    A('')
    A('| # | Item | Choice made here |')
    A('|---|---|---|')
    A('| 1 | SMVMD stopping criterion (how K is decided) | Stop when residual '
      'energy falls below 1% of the segment energy |')
    A('| 2 | DS-1 preprocessing | MODE B: the dataset\'s own preprocessed '
      'recordings (the paper\'s ICA/ADJUST stage is EEGLAB/MATLAB-only) |')
    A('| 3 | SMVMD centre-frequency initialisation | Uniform over [0, 0.5] |')
    A('| 4 | mRMR scope | Fitted per fold on training data only — stricter '
      'than the paper, which does not say |')
    A('| 5 | IP / PCC kernel bandwidth | Silverman\'s rule |')
    A('| 6 | Last segment at the recording boundary | Excluded, which '
      'reproduces the paper\'s 46 segments per DS-1 recording |')
    A('| 7 | Bayesian optimisation | Not re-run; the Table II hyperparameters '
      'the paper reports are used directly |')
    A('')

    A('## 6. Provenance')
    A('')
    A('| Experiment | Run at | Runtime | Python | NumPy |')
    A('|---|---|---|---|---|')
    for key in EXPERIMENT_ORDER:
        if key not in data:
            continue
        d = data[key]
        pv = d.get('provenance', {})
        rt = d.get('runtime_sec')
        A(f"| {key} | {d.get('run_at', '—')} | "
          f"{(f'{rt / 60:.1f} min' if rt else '—')} | "
          f"{pv.get('python', '—')} | {pv.get('numpy', '—')} |")
    A('')
    A('Per-experiment raw numbers: `results/metrics/*.json`. '
      'CSV tables: `results/tables/`. Figures: '
      '`results/confusion_matrices/`, `results/plots/`.')
    A('')

    return '\n'.join(L), summary_rows


def main():
    ensure_dirs()
    data = load_all_experiments()

    if not data:
        logger.error(
            "No saved experiments found in results/metrics/. "
            "Run at least one experiment first, e.g. "
            "`python -m experiments.ds1_rest`."
        )
        return 1

    report, summary_rows = build_report(data)

    out_path = Path(RESULTS_ROOT) / 'final_report.md'
    out_path.write_text(report, encoding='utf-8')
    save_table('summary', 'all_experiments', summary_rows)

    found = [k for k in EXPERIMENT_ORDER if k in data]
    missing = [k for k in EXPERIMENT_ORDER if k not in data]
    logger.info(f"Report written -> {out_path} ({len(report)} chars)")
    logger.info(f"Experiments included: {', '.join(found)}")
    if missing:
        logger.warning(f"Not yet run: {', '.join(missing)}")
    print(f"\nReport: {out_path}")
    print(f"Included: {', '.join(found)}")
    if missing:
        print(f"Still to run: {', '.join(missing)}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
