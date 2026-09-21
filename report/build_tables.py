"""
build_tables.py - Generate every number and table in the MSE report from
results/stage1/results_summary.json.

    python report/build_tables.py

1. Adds two derived sections to the summary (still computed from the saved
   caches and per-subject CSVs, never typed in):
     dataset_counts   segments / subjects per class, 0%-overlap segment counts
     per_subject      per-child accuracy stats under subject-wise CV
2. Writes report/generated/numbers.tex  -- one macro per number used in text
3. Writes report/generated/tab_*.tex    -- the tables

The report text uses only these macros, so no number in main.tex is typed by
hand, and re-running an experiment + this script updates the whole report.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

SUMMARY = ROOT / 'results/stage1/results_summary.json'
TABLES = ROOT / 'results/stage1/tables'
GEN = ROOT / 'report/generated'
FEAT = ROOT / 'results/features'

EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
# LaTeX macro names cannot contain digits or underscores.
TAG = {'DS1_Rest': 'Rest', 'DS1_Music': 'Music', 'DS1_RestMusic': 'RestMusic', 'DS2': 'Adhd'}
LABEL = {'DS1_Rest': 'DS-1 Rest', 'DS1_Music': 'DS-1 Music',
         'DS1_RestMusic': 'DS-1 Rest+Music', 'DS2': 'DS-2'}
CLASSES = {'DS1_Rest': ('TDC', 'IDD'), 'DS1_Music': ('TDC', 'IDD'),
           'DS1_RestMusic': ('TDC', 'IDD'), 'DS2': ('NC', 'ADHD')}


def pct(v):
    return f'{v * 100:.2f}\\%'


def add_derived(s):
    from experiments.stage1_analysis import zero_overlap_mask
    counts = {}
    for e in EXPS:
        d = np.load(FEAT / f'{e}_features.npz', allow_pickle=False)
        y, g = d['y'].astype(int), d['groups'].astype(str)
        m = zero_overlap_mask(e, g)
        subj_lab = {sid: int(y[g == sid][0]) for sid in np.unique(g)}
        counts[e] = {
            'segments_total': int(len(y)),
            'segments_control': int((y == 0).sum()),
            'segments_disorder': int((y == 1).sum()),
            'subjects_total': len(subj_lab),
            'subjects_control': sum(v == 0 for v in subj_lab.values()),
            'subjects_disorder': sum(v == 1 for v in subj_lab.values()),
            'segments_overlap0': int(m.sum()),
            'segments_per_subject_min': int(min(np.sum(g == s_) for s_ in subj_lab)),
            'segments_per_subject_max': int(max(np.sum(g == s_) for s_ in subj_lab)),
        }
    s['dataset_counts'] = counts

    per = {}
    for e in EXPS:
        per[e] = {}
        for tag in ('smvmd_overlap50', 'raw_overlap50'):
            rows = list(csv.DictReader(open(TABLES / f'per_subject_{e}_{tag}_SUBJ.csv', encoding='utf-8')))
            acc = np.array([float(r['segment_accuracy']) for r in rows])
            per[e][tag] = {
                'subjects': [r['subject'] for r in rows],
                'segment_accuracy': acc.tolist(),
                'n_below_50pct': int((acc < 0.5).sum()),
                'n_zero': int((acc == 0).sum()),
                'n_perfect': int((acc == 1).sum()),
                'min': float(acc.min()), 'max': float(acc.max()),
                'median': float(np.median(acc)),
            }
    s['per_subject'] = per
    SUMMARY.write_text(json.dumps(s, indent=2), encoding='utf-8')
    return s


def main():
    s = json.loads(SUMMARY.read_text(encoding='utf-8'))
    s = add_derived(s)
    GEN.mkdir(parents=True, exist_ok=True)
    M = []

    def mac(name, value):
        M.append(f'\\newcommand{{\\{name}}}{{{value}}}')

    for e in EXPS:
        t = TAG[e]
        rep = s['reproduction'][e]['ours']
        mac(f'accRep{t}', pct(rep['accuracy']))
        mac(f'fRep{t}', pct(rep['f1']))
        P = s['protocols'][e]
        for ov, ovt in (('smvmd_overlap50', 'Fifty'), ('smvmd_overlap0', 'Zero')):
            for pr, prt in (('SEG10', 'Seg'), ('SUBJ', 'Subj')):
                r = P[ov]['protocols'][pr]
                mac(f'acc{prt}{ovt}{t}', pct(r['segment_accuracy']))
                mac(f'bal{prt}{ovt}{t}', pct(r['segment_balanced_accuracy']))
                mac(f'vote{prt}{ovt}{t}', f"{r['subject_majority_vote_correct']}/{r['n_subjects']}")
        if 'SEG10_GLOBAL' in P['smvmd_overlap50']['protocols']:
            mac(f'accGlobal{t}', pct(P['smvmd_overlap50']['protocols']['SEG10_GLOBAL']['segment_accuracy']))
        R = s['raw_baseline'][e]['protocols']
        mac(f'accRawSeg{t}', pct(R['SEG10']['segment_accuracy']))
        mac(f'accRawSubj{t}', pct(R['SUBJ']['segment_accuracy']))
        mac(f'voteRawSubj{t}', f"{R['SUBJ']['subject_majority_vote_correct']}/{R['SUBJ']['n_subjects']}")
        mac(f'ties{t}', str(P['smvmd_overlap50']['protocols']['SUBJ']['subject_vote_ties']))
        E = s['embedding'][e]
        mac(f'silClass{t}', f"{E['silhouette_by_class']:.3f}")
        mac(f'silSubj{t}', f"{E['silhouette_by_subject']:.3f}")
        mac(f'subjID{t}', pct(E['subject_identification_accuracy_1nn']))
        mac(f'subjIDchance{t}', pct(E['subject_identification_chance']))
        C = s['dataset_counts'][e]
        for k, n in (('segments_total', 'nSeg'), ('segments_control', 'nSegCtl'),
                     ('segments_disorder', 'nSegDis'), ('subjects_total', 'nSubj'),
                     ('subjects_control', 'nSubjCtl'), ('subjects_disorder', 'nSubjDis'),
                     ('segments_overlap0', 'nSegZero'),
                     ('segments_per_subject_min', 'nSegPerSubjMin'),
                     ('segments_per_subject_max', 'nSegPerSubjMax')):
            mac(f'{n}{t}', str(C[k]))
        mac(f'nFeat{t}', str(s['protocols'][e]['smvmd_overlap50']['n_features_total']))
        mac(f'nFoldsSubj{t}', str(P['smvmd_overlap50']['protocols']['SUBJ']['n_folds']))
        ps = s['per_subject'][e]['smvmd_overlap50']
        mac(f'nBelowHalf{t}', str(ps['n_below_50pct']))
        mac(f'nZero{t}', str(ps['n_zero']))
        mac(f'nPerfect{t}', str(ps['n_perfect']))

    # DS-2 only
    rep2 = s['reproduction']['DS2']['ours']
    mac('precRepAdhd', pct(rep2['precision']))
    mac('recRepAdhd', pct(rep2['recall']))
    mac('mccRepAdhd', f"{rep2['mcc']:.3f}")
    mac('kappaRepAdhd', f"{rep2['kappa']:.3f}")

    # Mode counts
    mc = s['mode_counts']
    for e, t in (('DS1_RestMusic', 'DsOne'), ('DS2', 'DsTwo')):
        ctl, dis = CLASSES[e]
        mac(f'modeMean{t}', f"{mc[e]['overall']['mean']:.2f}")
        mac(f'modeSd{t}', f"{mc[e]['overall']['std']:.2f}")
        mac(f'modeMin{t}', str(mc[e]['overall']['min']))
        mac(f'modeMax{t}', str(mc[e]['overall']['max']))
        mac(f'modeCap{t}', f"{mc[e]['overall']['pct_at_cap_20']:.1f}\\%")
        mac(f'modeCtl{t}', f"{mc[e][ctl]['mean']:.2f}")
        mac(f'modeDis{t}', f"{mc[e][dis]['mean']:.2f}")

    (GEN / 'numbers.tex').write_text(
        '% AUTO-GENERATED by report/build_tables.py from results_summary.json. Do not edit.\n'
        + '\n'.join(M) + '\n', encoding='utf-8')

    # ── Table: reproduced results (segment-wise protocol) ──────────────────
    L = [r'\begin{tabular}{lcccccl}', r'\toprule',
         r'Setting & Segments & Features & Accuracy & Precision & Recall & KNN (Table II of \cite{chandela2025})\\',
         r'\midrule']
    for e in EXPS:
        r = s['reproduction'][e]
        o, kp = r['ours'], r['knn_params']
        dist = {'cityblock': 'city-block', 'euclidean': 'Euclidean', 'cosine': 'cosine'}[kp['distance']]
        w = {'uniform': 'equal', 'squared_inverse': 'sq.\\ inverse'}[kp['weights']]
        L.append(f"{LABEL[e]} & {r['n_segments']} & {r['n_features_used']} & {pct(o['accuracy'])} & "
                 f"{pct(o['precision'])} & {pct(o['recall'])} & "
                 f"$k{{=}}{kp['n_neighbors']}$, {dist}, {w}{', std.' if kp['standardize'] else ''} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_reproduction.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')

    # ── Table: protocol comparison ─────────────────────────────────────────
    L = [r'\begin{tabular}{l cccc cc cc}', r'\toprule',
         r' & \multicolumn{4}{c}{SMVMD features, 50\% overlap} & \multicolumn{2}{c}{SMVMD, 0\% overlap} & \multicolumn{2}{c}{Raw channels, 50\%}\\',
         r'\cmidrule(lr){2-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
         r'Setting & Seg-wise & Subj-wise & Subj bal.\ acc. & Children & Seg-wise & Subj-wise & Seg-wise & Subj-wise\\',
         r'\midrule']
    for e in EXPS:
        P, R = s['protocols'][e], s['raw_baseline'][e]['protocols']
        a, z = P['smvmd_overlap50']['protocols'], P['smvmd_overlap0']['protocols']
        L.append(f"{LABEL[e]} & {pct(a['SEG10']['segment_accuracy'])} & {pct(a['SUBJ']['segment_accuracy'])} & "
                 f"{pct(a['SUBJ']['segment_balanced_accuracy'])} & "
                 f"{a['SUBJ']['subject_majority_vote_correct']}/{a['SUBJ']['n_subjects']} & "
                 f"{pct(z['SEG10']['segment_accuracy'])} & {pct(z['SUBJ']['segment_accuracy'])} & "
                 f"{pct(R['SEG10']['segment_accuracy'])} & {pct(R['SUBJ']['segment_accuracy'])} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_protocols.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')

    # ── Table: class vs subject structure ──────────────────────────────────
    L = [r'\begin{tabular}{lcccc}', r'\toprule',
         r'Setting & Silhouette (class) & Silhouette (subject) & Subject ID, 1-NN & Chance\\', r'\midrule']
    for e in EXPS:
        E = s['embedding'][e]
        L.append(f"{LABEL[e]} & {E['silhouette_by_class']:.3f} & {E['silhouette_by_subject']:.3f} & "
                 f"{pct(E['subject_identification_accuracy_1nn'])} & {pct(E['subject_identification_chance'])} \\\\")
    L += [r'\bottomrule', r'\end{tabular}']
    (GEN / 'tab_embedding.tex').write_text('\n'.join(L) + '\n', encoding='utf-8')

    print(f'wrote {len(M)} macros and 3 tables to {GEN}')


if __name__ == '__main__':
    main()
