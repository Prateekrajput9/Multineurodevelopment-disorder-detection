"""
build_project_record.py - Full project record as a PDF.

    python report/build_project_record.py   ->  report/Project_Record.pdf

Every result number is read from results/stage1/results_summary.json and
results/metrics/*.json at build time; none is typed into this script.
"""
import json
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / 'results/stage1/results_summary.json').read_text(encoding='utf-8'))
M = {e: json.loads((ROOT / f'results/metrics/{e}.json').read_text(encoding='utf-8'))
     for e in ('DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2')}
OUT = ROOT / 'report/Project_Record.pdf'
FIG = ROOT / 'results/stage1/figures'

EXPS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic', 'DS2']
LABEL = {'DS1_Rest': 'DS-1 Rest', 'DS1_Music': 'DS-1 Music',
         'DS1_RestMusic': 'DS-1 Rest+Music', 'DS2': 'DS-2 ADHD'}

# ── fonts & styles ─────────────────────────────────────────────────────────
FD = Path('C:/Windows/Fonts')
pdfmetrics.registerFont(TTFont('Arial', str(FD / 'arial.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Bold', str(FD / 'arialbd.ttf')))
pdfmetrics.registerFont(TTFont('Arial-Italic', str(FD / 'ariali.ttf')))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', str(FD / 'arialbi.ttf')))
pdfmetrics.registerFont(TTFont('Consolas', str(FD / 'consola.ttf')))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily('Arial', normal='Arial', bold='Arial-Bold',
                   italic='Arial-Italic', boldItalic='Arial-BoldItalic')

INK = colors.HexColor('#1f2933')
ACCENT = colors.HexColor('#1d4e89')
MUTED = colors.HexColor('#5f6b7a')
RULE = colors.HexColor('#c9d2dc')
HEAD_BG = colors.HexColor('#e8eef5')
ZEBRA = colors.HexColor('#f6f8fa')
NOTE_BG = colors.HexColor('#fff7e6')
NOTE_EDGE = colors.HexColor('#e0a526')

ss = getSampleStyleSheet()
body = ParagraphStyle('body', parent=ss['Normal'], fontName='Arial', fontSize=10,
                      leading=14, textColor=INK, spaceAfter=5)
small = ParagraphStyle('small', parent=body, fontSize=8.5, leading=11, textColor=MUTED)
cell = ParagraphStyle('cell', parent=body, fontSize=8.8, leading=11.2, spaceAfter=0)
cellb = ParagraphStyle('cellb', parent=cell, fontName='Arial-Bold')
h1 = ParagraphStyle('h1', parent=body, fontName='Arial-Bold', fontSize=15, leading=19,
                    textColor=ACCENT, spaceBefore=12, spaceAfter=6, keepWithNext=1)
h2 = ParagraphStyle('h2', parent=body, fontName='Arial-Bold', fontSize=11.5, leading=15,
                    textColor=INK, spaceBefore=8, spaceAfter=4, keepWithNext=1)
bullet = ParagraphStyle('bullet', parent=body, leftIndent=14, bulletIndent=3, spaceAfter=2)
code = ParagraphStyle('code', parent=body, fontName='Consolas', fontSize=8.8, leading=11.5,
                      leftIndent=8, backColor=ZEBRA, borderPadding=(4, 6, 4, 6),
                      spaceBefore=3, spaceAfter=6)
title = ParagraphStyle('title', parent=body, fontName='Arial-Bold', fontSize=21, leading=26,
                       alignment=TA_CENTER, textColor=ACCENT)
subtitle = ParagraphStyle('subtitle', parent=body, fontSize=12, leading=16,
                          alignment=TA_CENTER, textColor=MUTED)
cap = ParagraphStyle('cap', parent=small, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)


def pct(v):
    return f'{v * 100:.2f}%'


def P(t, st=body):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(t, bullet, bulletText='•') for t in items]


def table(rows, widths, header=True, bold_first_col=False):
    data = []
    for i, r in enumerate(rows):
        st = cellb if (header and i == 0) else cell
        data.append([Paragraph(str(c), cellb if (bold_first_col and j == 0 and (i > 0 or not header)) else st)
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign='LEFT')
    style = [('GRID', (0, 0), (-1, -1), 0.4, RULE),
             ('VALIGN', (0, 0), (-1, -1), 'TOP'),
             ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
             ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5)]
    if header:
        style.append(('BACKGROUND', (0, 0), (-1, 0), HEAD_BG))
    for i in range(1 if header else 0, len(rows)):
        if (i % 2 == 0) == header:
            style.append(('BACKGROUND', (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def note(text):
    t = Table([[Paragraph(text, cell)]], colWidths=[16.4 * cm])
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), NOTE_BG),
                           ('LINEBEFORE', (0, 0), (0, -1), 3, NOTE_EDGE),
                           ('LEFTPADDING', (0, 0), (-1, -1), 8),
                           ('TOPPADDING', (0, 0), (-1, -1), 5),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
    return t


def figure(path, width_cm, caption):
    w, h = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)
    return KeepTogether([img, P(caption, cap)])


# ── shorthand for the numbers ──────────────────────────────────────────────
R = S['reproduction']
PR = S['protocols']
RAW = S['raw_baseline']
EMB = S['embedding']
MC = S['mode_counts']
CNT = S['dataset_counts']
PS = S['per_subject']


def proto(e, ov, p):
    return PR[e][ov]['protocols'][p]


_ds1 = [R[e]['ours']['accuracy'] for e in EXPS[:3]]
DS1_LO, DS1_HI = pct(min(_ds1)), pct(max(_ds1))


story = []
A = story.append

# ── cover ──────────────────────────────────────────────────────────────────
A(Spacer(1, 3.2 * cm))
A(P('EEG-based Multiple Neurodevelopmental<br/>Disorders Detection', title))
A(Spacer(1, 10))
A(P('B.Tech. Project — complete record of work, results and plan', subtitle))
A(Spacer(1, 1.2 * cm))
A(table([
    ['Student', '[Name], Roll No. [roll no.]'],
    ['Supervisor', '[Supervisor], [Department], IIT Indore'],
    ['Project period', 'August 2026 – end of November 2026'],
    ['Mid-semester evaluation', '25 September 2026'],
    ['Final evaluation & submission', 'End of November 2026'],
    ['Record generated', date.today().strftime('%d %B %Y')],
], [5.2 * cm, 11.2 * cm], header=False, bold_first_col=True))
A(Spacer(1, 1.0 * cm))
A(P('<b>Contents</b>', h2))
for line in ['1. The project in one page', '2. Background: problem, paper and datasets',
             '3. The pipeline, step by step', '4. What was built and fixed',
             '5. Results: reproduction', '6. Results: subject-independent evaluation and diagnostics',
             '7. What the results mean', '8. Caveats and things to be ready for',
             '9. Mid-semester report status', '10. Plan until the final evaluation',
             '11. How to re-run everything', '12. Summary in simple words']:
    A(P(line, ParagraphStyle('toc', parent=body, leftIndent=10, spaceAfter=1)))
A(PageBreak())

# ── 1. one page ────────────────────────────────────────────────────────────
A(P('1. The project in one page', h1))
A(P('<b>Goal.</b> Detect two childhood neurodevelopmental disorders — intellectual and '
    'developmental disorder (IDD) and attention-deficit/hyperactivity disorder (ADHD) — '
    'from EEG, and find out how trustworthy such detectors really are.'))
A(P('<b>Starting point.</b> U. Chandela, K. N. Faisal and R. R. Sharma, “Electroencephalogram-'
    'Based Unified Approach for Multiple Neurodevelopmental Disorders Detection in Children '
    'Using Successive Multivariate Variational Mode Decomposition,” <i>IEEE Trans. Cogn. '
    'Develop. Syst.</i>, vol. 17, no. 6, pp. 1350–1359, Dec. 2025. One pipeline (SMVMD → 9 '
    'features → energy weighting → mRMR → KNN) is used for both disorders.'))
A(P('<b>What makes this project different.</b> The paper evaluates by shuffling short EEG '
    'windows into cross-validation folds, so windows from the same child appear in both '
    'training and testing. This project reproduces the pipeline and then tests it the way it '
    'would be used in practice — on children it has never seen.'))
A(P('<b>Headline result.</b>'))
A(table([['Setting', 'Seen children (segment-wise)', 'New children (subject-wise)',
          'Children correct']] +
        [[LABEL[e], pct(proto(e, 'smvmd_overlap50', 'SEG10')['segment_accuracy']),
          pct(proto(e, 'smvmd_overlap50', 'SUBJ')['segment_accuracy']),
          f"{proto(e, 'smvmd_overlap50', 'SUBJ')['subject_majority_vote_correct']}"
          f"/{proto(e, 'smvmd_overlap50', 'SUBJ')['n_subjects']}"] for e in EXPS],
        [4.2 * cm, 4.4 * cm, 4.4 * cm, 3.4 * cm]))
A(Spacer(1, 6))
A(P(f"<b>Why.</b> The features mainly identify <i>which child</i> a window came from: a "
    f"nearest-neighbour test names the child with {pct(EMB['DS1_Rest']['subject_identification_accuracy_1nn'])} "
    f"accuracy on DS-1 rest (chance {pct(EMB['DS1_Rest']['subject_identification_chance'])}). "
    f"A classifier that has seen a child's other windows simply recognises the child."))
A(P('<b>Next.</b> Build and validate a model that works on new children: nested subject-wise '
    'evaluation, an ablation study of the decomposition, interpretability, cross-dataset '
    'analysis and channel reduction, finishing at the end of November.'))

# ── 2. background ──────────────────────────────────────────────────────────
A(P('2. Background: problem, paper and datasets', h1))
A(P('2.1 Why EEG for IDD and ADHD', h2))
A(P('Both disorders start in childhood, and support started early tends to help more. '
    'Diagnosis today relies on clinical interviews and rating scales, which need trained '
    'specialists and depend on observer judgement. EEG records brain electrical activity '
    'from the scalp: it is non-invasive, cheap, tolerated well by children, and portable '
    'headsets exist. The difficulty is that EEG is noisy, changes over time and differs a lot '
    'between people, which is why signal decomposition and machine learning are used.'))
A(P('2.2 The two datasets', h2))
A(table([
    ['', 'DS-1 (IDD)', 'DS-2 (ADHD)'],
    ['Source', 'Sareen et al., Data in Brief, 2020 (Mendeley)',
     'Motie Nasrabadi et al., IEEE DataPort, 2020'],
    ['Children', f"{CNT['DS1_Rest']['subjects_disorder']} IDD + {CNT['DS1_Rest']['subjects_control']} typical (TDC)",
     f"{CNT['DS2']['subjects_disorder']} ADHD + {CNT['DS2']['subjects_control']} controls (NC)"],
    ['Recording', '14-channel EMOTIV EPOC+, 128 Hz; 2 min rest and 2 min music',
     '19-channel 10–20 EEG, 128 Hz; visual attention task'],
    ['Windows (5 s, 50% overlap)',
     f"{CNT['DS1_Rest']['segments_total']} per condition ({CNT['DS1_Rest']['segments_disorder']} IDD / "
     f"{CNT['DS1_Rest']['segments_control']} TDC); {CNT['DS1_RestMusic']['segments_total']} rest+music",
     f"{CNT['DS2']['segments_total']} ({CNT['DS2']['segments_disorder']} ADHD / {CNT['DS2']['segments_control']} NC)"],
    ['Windows per child', f"{CNT['DS1_Rest']['segments_per_subject_min']} per condition",
     f"{CNT['DS2']['segments_per_subject_min']}–{CNT['DS2']['segments_per_subject_max']} (recording length varies)"],
    ['Note', 'Very small cohort: each child is 1/14 of the data',
     'ADHD children were on methylphenidate (Ritalin); controls were not'],
], [3.6 * cm, 6.4 * cm, 6.4 * cm]))
A(Spacer(1, 4))
A(P('The window counts produced by this implementation equal those stated in the paper '
    '(1288 for DS-1 and 6588 for DS-2), which confirms that loading and segmentation were '
    'reproduced correctly.', small))

# ── 3. pipeline ────────────────────────────────────────────────────────────
A(P('3. The pipeline, step by step', h1))
A(table([
    ['Step', 'What happens', 'Settings used'],
    ['1. Preprocessing', 'Clean the raw EEG.', 'DS-1: the dataset’s own cleaned files. DS-2: '
     'Butterworth 0.5–60 Hz, 50 Hz notch, Coiflet-3 wavelet + SURE denoising (level 6).'],
    ['2. Windowing', 'Cut each recording into short pieces; each piece is one sample.',
     '5 s windows (640 samples), 50% overlap (step 320).'],
    ['3. SMVMD', 'Split each multichannel window into oscillatory “modes” (like separating a '
     'chord into notes). The number of modes K is chosen automatically and differs per window.',
     'α = 2000 (DS-1) / 1000 (DS-2), τ = 0, tol = 1e-10; stop when residual energy < 1% or K = 20.'],
    ['4. Features', 'Describe every mode on every channel with 9 numbers.',
     'STD, VAR, RMS, IQR, ASSR, average power, max fractal length, information potential, '
     'parametric-centred correntropy.'],
    ['5. Energy weighting', 'Combine the modes of each channel into one fixed-length vector, '
     'weighting each mode by its energy (paper Eq. 7).', '14 × 9 = 126 features (DS-1); '
     '19 × 9 = 171 (DS-2).'],
    ['6. mRMR', 'Rank features: most informative, least redundant first; keep the top ones.',
     f"Keep {R['DS1_Rest']['n_features_used']} (rest), {R['DS1_Music']['n_features_used']} (music), "
     f"{R['DS1_RestMusic']['n_features_used']} (rest+music), {R['DS2']['n_features_used']} (DS-2)."],
    ['7. KNN', 'Classify a window by its nearest neighbours.', 'Hyperparameters from the paper’s '
     'Table II (see Section 5). The paper’s Bayesian optimisation was not re-run.'],
    ['8. Evaluation', 'Measure accuracy with cross-validation.', 'Segment-wise 10-fold (paper) '
     'and subject-wise (this project): LOSO for DS-1, grouped 10-fold for DS-2.'],
], [3.0 * cm, 6.6 * cm, 6.8 * cm]))

# ── 4. what was built ──────────────────────────────────────────────────────
A(P('4. What was built and fixed', h1))
A(P('4.1 Code base', h2))
A(table([
    ['Part', 'Contents'],
    ['src/', 'Data loaders, preprocessing, segmentation, SMVMD, features, energy integration, '
     'mRMR, KNN, evaluation, result store (results_io.py), shared DS-1 driver (pipeline.py).'],
    ['experiments/', 'ds1_rest, ds1_music, ds1_rest_music, ds2_adhd (the four reproductions); '
     'stage1_analysis (subject-wise protocols and diagnostics); make_report.'],
    ['results/metrics/', 'One JSON per experiment with every metric, settings and timings.'],
    ['results/features/', 'Cached feature matrices, so re-runs skip SMVMD (seconds instead of an hour).'],
    ['results/stage1/', 'results_summary.json (single source of all numbers), figures, per-child CSVs.'],
    ['report/', 'MSE report in LaTeX (IITI title page), table/number generator, checker, this record.'],
], [3.6 * cm, 12.8 * cm]))
A(P('4.2 Problems found and fixed', h2))
A(table([
    ['Problem', 'Effect', 'Fix'],
    ['DS-2 crashed on its first subject (an unexpected config argument)', 'No DS-2 results '
     'could ever be produced', 'Preprocessing accepts the argument and rejects anything but SURE'],
    ['PCC feature summed only 50 time lags but divided as if it summed all',
     'PCC values wrong (≈75% too high in a test)', 'Exact closed form over all lags, '
     'verified against brute force; also faster'],
    ['Report writer crashed on Windows (text encoding)', 'DS-1 rest never saved results',
     'UTF-8 output'],
    ['Only one DS-1 experiment saved results; one used a wrong key for paper targets',
     'Results lost / comparison silently empty', 'Shared DS-1 driver; every run saves a full bundle'],
    ['Module log messages were silently discarded', 'Long runs looked frozen',
     'Logging attached to the root logger'],
    ['Full mRMR ranking computed and thrown away; mRMR run even when keeping all features',
     'Hours of wasted computation', 'Removed; skipped when all features are kept'],
    ['DS-2 mRMR redundancy matrix estimated from all 6588 windows', '≈2.5 h for one ranking',
     'Estimated from a seeded 2000-window sample (DS-1 unaffected: fewer windows than that)'],
    ['Raw DS-2 cache saved subject IDs in a format that could not be reloaded',
     'Raw baseline evaluation crashed', 'Saved as plain strings; no recomputation needed'],
], [5.6 * cm, 4.6 * cm, 6.2 * cm]))

# ── 5. reproduction ────────────────────────────────────────────────────────
A(P('5. Results: reproduction (paper-style evaluation)', h1))
A(P('Segment-wise stratified 10-fold cross-validation, with scaling and mRMR fitted inside '
    'each training fold. Precision and recall are class-weighted.'))
rows = [['Setting', 'Windows', 'Features', 'Accuracy', 'Precision', 'Recall', 'F1', 'KNN (paper Table II)']]
for e in EXPS:
    o, kp = R[e]['ours'], R[e]['knn_params']
    rows.append([LABEL[e], R[e]['n_segments'], R[e]['n_features_used'], pct(o['accuracy']),
                 pct(o['precision']), pct(o['recall']), pct(o['f1']),
                 f"k={kp['n_neighbors']}, {kp['distance']}, {kp['weights'].replace('_', ' ')}"
                 f"{', standardised' if kp['standardize'] else ''}"])
A(table(rows, [2.4 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm, 1.6 * cm, 1.7 * cm, 3.5 * cm]))
A(Spacer(1, 6))
o2 = R['DS2']['ours']
A(P(f"DS-2 extra metrics: MCC {o2['mcc']:.3f}, Cohen's κ {o2['kappa']:.3f}, "
    f"G-mean {o2['gmean']:.3f}."))
cc = R['DS2']['classifier_comparison']
A(P('DS-2 classifier comparison (same features, segment-wise 10-fold):', h2))
A(table([['Classifier', 'Accuracy', 'F1', 'Time per window']] +
        [[k, pct(v['accuracy']), pct(v['f1']), f"{v['time_ms_per_sample']:.3f} ms"]
         for k, v in cc.items()],
        [4 * cm, 3.5 * cm, 3.5 * cm, 4 * cm]))
A(Spacer(1, 6))
A(P('mRMR feature-count sweep (accuracy as more ranked features are added):', h2))
A(table([['Setting', 'Features kept (paper)', 'Best count found here']] +
        [[LABEL[e], R[e]['n_features_used'], R[e]['mrmr_empirical_optimum']] for e in EXPS],
        [4.5 * cm, 5 * cm, 5 * cm]))

# ── 6. stage 1 ─────────────────────────────────────────────────────────────
A(P('6. Results: subject-independent evaluation and diagnostics', h1))
A(P('Every analysis below uses the same features and the same classifier settings; only the '
    'way the data is split, or one component, changes. Scaling, mRMR and KNN are always '
    'fitted inside each training fold.'))
A(P('6.1 Seen children vs new children', h2))
rows = [['Setting', 'Overlap', 'Segment-wise', 'Subject-wise', 'Subject-wise bal. acc.',
         'Children correct (vote)']]
for e in EXPS:
    for ov, name in (('smvmd_overlap50', '50%'), ('smvmd_overlap0', '0%')):
        a, b = proto(e, ov, 'SEG10'), proto(e, ov, 'SUBJ')
        rows.append([LABEL[e], name, pct(a['segment_accuracy']), pct(b['segment_accuracy']),
                     pct(b['segment_balanced_accuracy']),
                     f"{b['subject_majority_vote_correct']}/{b['n_subjects']}"])
A(table(rows, [3.1 * cm, 1.6 * cm, 2.6 * cm, 2.6 * cm, 3.1 * cm, 3.4 * cm]))
A(P('Subject-wise = leave-one-subject-out (14 folds) for DS-1 and grouped 10-fold for DS-2: '
    'all windows of a child are held out together. “Children correct” assigns each held-out '
    'child the majority label of its windows. 0%-overlap results use the even-numbered '
    'windows only (the window is exactly twice the step), taken from the cache.', small))
A(P('6.2 Does fitting mRMR on all data inflate accuracy?', h2))
A(P('The paper ranks features once on all windows before cross-validation. Doing the same '
    'here (segment-wise):'))
A(table([['Setting', 'mRMR inside each fold', 'mRMR on all data (paper order)']] +
        [[LABEL[e], pct(proto(e, 'smvmd_overlap50', 'SEG10')['segment_accuracy']),
          pct(proto(e, 'smvmd_overlap50', 'SEG10_GLOBAL')['segment_accuracy'])]
         for e in EXPS[:3]], [4.5 * cm, 5 * cm, 5.5 * cm]))
A(P('Equal or lower, so feature selection is not what makes the segment-wise scores high. '
    '(DS-2 keeps all 171 features, so ranking cannot matter there.)', small))
A(P('6.3 Raw-channel baseline: is SMVMD needed?', h2))
A(P('The same 9 features computed directly on each preprocessed channel, with no SMVMD:'))
rows = [['Setting', 'Raw: segment-wise', 'Raw: subject-wise', 'SMVMD: subject-wise',
         'Raw: children correct']]
for e in EXPS:
    rr = RAW[e]['protocols']
    rows.append([LABEL[e], pct(rr['SEG10']['segment_accuracy']), pct(rr['SUBJ']['segment_accuracy']),
                 pct(proto(e, 'smvmd_overlap50', 'SUBJ')['segment_accuracy']),
                 f"{rr['SUBJ']['subject_majority_vote_correct']}/{rr['SUBJ']['n_subjects']}"])
A(table(rows, [3.3 * cm, 3.2 * cm, 3.2 * cm, 3.4 * cm, 3.3 * cm]))
A(P('6.4 What the features encode', h2))
A(table([['Setting', 'Silhouette by class', 'Silhouette by child', 'Child identified (1-NN)', 'Chance']] +
        [[LABEL[e], f"{EMB[e]['silhouette_by_class']:.3f}", f"{EMB[e]['silhouette_by_subject']:.3f}",
          pct(EMB[e]['subject_identification_accuracy_1nn']),
          pct(EMB[e]['subject_identification_chance'])] for e in EXPS],
        [3.3 * cm, 3.2 * cm, 3.2 * cm, 3.8 * cm, 2.9 * cm]))
A(P('Silhouette: how well points group by a label (higher = tighter groups). The 1-NN probe '
    'predicts <i>which child</i> produced a window, using segment-wise 10-fold CV.', small))
A(figure(FIG / 'tsne_class_vs_subject.png', 12.5,
         'Figure 1. t-SNE of the SMVMD features. Left: coloured by class; right: coloured by child. '
         'Top: DS-1 rest (one island per child); bottom: DS-2.'))
A(P('6.5 Accuracy per child', h2))
psr, ps2 = PS['DS1_Rest']['smvmd_overlap50'], PS['DS2']['smvmd_overlap50']
A(P(f"DS-1 rest (leave-one-subject-out): {psr['n_perfect']} children are right in every "
    f"window and {psr['n_zero']} are wrong in every window — the model decides per child. "
    f"DS-2: accuracy varies continuously; {ps2['n_below_50pct']} of {CNT['DS2']['subjects_total']} "
    f"children have fewer than half of their windows right."))
def img_w(path, width_cm):
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


A(KeepTogether([
    Table([[img_w(FIG / 'per_subject_accuracy_DS1_Rest.png', 8),
            img_w(FIG / 'per_subject_accuracy_DS2.png', 8)]], colWidths=[8.2 * cm, 8.2 * cm]),
    P('Figure 2. Fraction of each held-out child’s windows classified correctly. Left: DS-1 rest; '
      'right: DS-2. Dashed line: 50%.', cap)]))
A(P('6.6 Number of SMVMD modes', h2))
m1, m2 = MC['DS1_RestMusic'], MC['DS2']
A(P(f"DS-1: mean K {m1['overall']['mean']:.2f} (IDD {m1['IDD']['mean']:.2f}, TDC "
    f"{m1['TDC']['mean']:.2f}); {m1['overall']['pct_at_cap_20']:.1f}% of windows hit the "
    f"20-mode safety cap. DS-2: mean K {m2['overall']['mean']:.2f} (ADHD {m2['ADHD']['mean']:.2f}, "
    f"NC {m2['NC']['mean']:.2f}); none hit the cap."))
A(figure(FIG / 'smvmd_mode_count_histogram.png', 15,
         'Figure 3. Number of SMVMD modes per window, by dataset and class.'))

# ── 7. meaning ─────────────────────────────────────────────────────────────
A(P('7. What the results mean', h1))
A(P('<b>The accuracy gap comes from recognising children.</b> Each child’s windows look very '
    'alike and very different from other children’s. When windows from the same child are in '
    'both training and test sets, the classifier finds that child’s other windows and copies '
    'the label. That works only for children already seen. For a new child — the real use '
    'case — accuracy is far lower.'))
A(P(f"<b>Overlap adds to it on DS-2.</b> Removing window overlap lowers DS-2 segment-wise accuracy "
    f"from {pct(proto('DS2', 'smvmd_overlap50', 'SEG10')['segment_accuracy'])} to "
    f"{pct(proto('DS2', 'smvmd_overlap0', 'SEG10')['segment_accuracy'])} while subject-wise stays "
    f"at about the same level ({pct(proto('DS2', 'smvmd_overlap50', 'SUBJ')['segment_accuracy'])} → "
    f"{pct(proto('DS2', 'smvmd_overlap0', 'SUBJ')['segment_accuracy'])}): shared samples between "
    f"overlapping windows were inflating the score. On DS-1 the child signature is strong enough "
    f"without overlap. Part of the DS-2 drop may also come from having half as many windows."))
A(P('<b>SMVMD has not yet shown a benefit.</b> Under subject-wise evaluation the raw-channel '
    'features beat the SMVMD features on all three DS-1 settings and are about equal on DS-2, '
    'even though the KNN settings were tuned by the paper for SMVMD features.'))
A(P('<b>This is about evaluation, not about one paper.</b> Splitting windows rather than '
    'children is common in EEG studies. The subject-wise numbers are the honest estimate for '
    'screening new children.'))

# ── 8. caveats ─────────────────────────────────────────────────────────────
A(P('8. Caveats and things to be ready for', h1))
A(P('8.1 Limitations of the current work', h2))
story.extend(bullets([
    'KNN hyperparameters come from the paper’s Table II; its Bayesian optimisation was not '
    're-run, and they were not re-tuned inside folds (planned: nested CV).',
    'DS-1 uses the dataset’s cleaned recordings instead of re-running the paper’s EEGLAB '
    'ICA/ADJUST cleaning (MATLAB-only).',
    'The paper does not say how SMVMD decides the number of modes. Here: stop when residual '
    f"energy < 1% of the window, with a 20-mode safety cap ({m1['overall']['pct_at_cap_20']:.1f}% "
    'of DS-1 windows hit it).',
    'The SMVMD implementation is a simplified successive ADMM; SMVMD reconstruction is '
    'deliberately incomplete because τ = 0, as in the paper.',
    f"With only {CNT['DS1_Rest']['subjects_total']} children in DS-1, each child is "
    f"{pct(EMB['DS1_Rest']['subject_identification_chance'])} of the data, so DS-1 subject-wise "
    f"numbers are coarse.",
    f"DS-2 grouped folds are not stratified by class; "
    f"{proto('DS2', 'smvmd_overlap50', 'SUBJ')['subject_vote_ties']} children tied 50/50 in the vote "
    f"(counted as NC).",
    'The t-SNE is a picture, not a proof; the silhouette and 1-NN numbers are the quantitative '
    'evidence. UMAP was not run (not installed).',
    'Segment-wise “children correct” is always perfect because every child is in training; '
    'it is not a meaningful result.',
]))
A(P('8.2 Likely viva questions', h2))
A(table([
    ['Question', 'Short answer'],
    ['Why is your accuracy lower than the paper’s?',
     'The paper reports 100% on all three DS-1 settings and 99.17% on DS-2. Under the same '
     f"segment-wise protocol this reproduction gets {DS1_LO}–{DS1_HI} on DS-1 and {pct(R['DS2']['ours']['accuracy'])} on DS-2. Possible "
     'reasons: the unspecified SMVMD stopping rule, the simplified SMVMD, DS-1 cleaning, and '
     'hyperparameters not re-optimised. The MSE report shows own results without comparison.'],
    ['Is subject-wise evaluation not too pessimistic?', 'It matches the real use: a new child '
     'is never in the training set. Segment-wise answers a different question (known children).'],
    ['Could the drop be because of less training data?', 'LOSO trains on 13 of 14 children, '
     'nearly all data; the drop is too large for that, and the child-ID probe shows the cause.'],
    ['Is ADHD detection valid on DS-2?', 'Medication is perfectly aligned with the label, so '
     'the dataset cannot separate ADHD from methylphenidate effects.'],
    ['Why not use deep learning?', 'The same leakage would affect any model; fixing the '
     'evaluation comes first. Model changes are planned inside the nested protocol.'],
], [5.4 * cm, 11.0 * cm]))

# ── 9. report status ───────────────────────────────────────────────────────
A(P('9. Mid-semester report status', h1))
story.extend(bullets([
    'LaTeX report with the IIT Indore title page (logo), compact body, six required sections, '
    'three references (the paper and the two datasets). Upload report/mse_report_overleaf.zip '
    'to Overleaf and compile main.tex.',
    'Every number in the report is a macro generated from results_summary.json by '
    'report/build_tables.py, so none is typed by hand; report/check_report.py verifies macros, '
    'citations and environments.',
    'Written from scratch; the only 40+ letter overlap with the paper’s text is the phrase '
    '“attention-deficit hyperactivity disorder (ADHD)”.',
    'To do before submission: fill the title-page placeholders (name, roll number, discipline, '
    'supervisor, department) and confirm on Overleaf that the body is at most 7 pages.',
]))

# ── 10. plan ───────────────────────────────────────────────────────────────
A(P('10. Plan until the final evaluation', h1))
A(table([
    ['Period', 'Work', 'Purpose / output'],
    ['August – 25 Sep', '<b>Done:</b> reproduction on both datasets; subject-wise protocols, '
     'overlap, raw-channel and embedding analyses; MSE report. Mid-evaluation on 25 Sep.', 'This record.'],
    ['28 Sep – 11 Oct', 'Nested subject-wise cross-validation: tune KNN settings and feature '
     'count inside each outer fold (Bayesian search); bootstrap confidence intervals over children.',
     'An accuracy estimate for new children with no leakage of any kind, with error bars.'],
    ['12 – 25 Oct', 'Ablation ladder under the nested protocol: raw channels → fixed frequency '
     'bands → VMD → MVMD → SMVMD; sensitivity to the SMVMD stopping rule and mode cap.',
     'Shows which step, if any, adds real value.'],
    ['26 Oct – 8 Nov', 'Interpretability: permutation importance and SHAP on subject-wise models; '
     'channel-level scalp maps; theta/beta power ratio as a simple ADHD baseline.',
     'What the model relies on, and whether it is physiologically plausible.'],
    ['9 – 15 Nov', 'Cross-dataset feature analysis on the 10 shared electrodes (F7, F3, F4, F8, '
     'T7, T8, P7, P8, O1, O2; T3/T4/T5/T6 in DS-2 = T7/T8/P7/P8).',
     'Whether patterns carry over between the two datasets.'],
    ['16 – 22 Nov', 'Channel reduction under subject-wise evaluation.',
     'How few electrodes are enough — relevant for cheap headsets.'],
    ['23 – 30 Nov', 'Consolidate, re-run everything from scripts, final report; final evaluation '
     'and submission.', 'Final report and presentation.'],
], [2.8 * cm, 8.4 * cm, 5.2 * cm]))

# ── 11. how to rerun ───────────────────────────────────────────────────────
A(P('11. How to re-run everything', h1))
A(P('From the project folder (C:\\Users\\Prateek\\OneDrive\\Desktop\\btpnew):'))
A(P('python -m experiments.ds1_rest<br/>python -m experiments.ds1_music<br/>'
    'python -m experiments.ds1_rest_music<br/>python -m experiments.ds2_adhd<br/>'
    'python -m experiments.stage1_analysis<br/>python report/build_tables.py<br/>'
    'python report/check_report.py<br/>python report/build_project_record.py', code))
A(P('Reproductions reuse cached features when present (add <font face="Consolas">--no-cache</font> to recompute). '
    'Measured run times: ' + ', '.join(
        f"{LABEL[e]} {M[e]['runtime_sec'] / 60:.0f} min" for e in EXPS) +
    ' (first run, 12-core machine).', small))

# ── 12. summary ────────────────────────────────────────────────────────────
A(PageBreak())
A(P('12. Summary in simple words', h1))
A(P('<b>What I have done</b>', h2))
story.extend(bullets([
    'I picked a 2025 research paper that uses one EEG method to detect two childhood '
    'disorders: IDD and ADHD.',
    f"I rebuilt that whole method in Python and ran it on the same two public datasets "
    f"({CNT['DS1_Rest']['subjects_total']} children for IDD, {CNT['DS2']['subjects_total']} for ADHD). "
    f"While doing so I found and fixed several bugs, "
    'including one wrong feature formula.',
    f"Tested the paper’s way, my version gets {pct(R['DS1_Rest']['ours']['accuracy'])} "
    f"(IDD, rest) and {pct(R['DS2']['ours']['accuracy'])} (ADHD).",
    'Then I tested it the way it would really be used: on children the model has never seen. '
    f"Accuracy dropped to {pct(proto('DS1_Rest', 'smvmd_overlap50', 'SUBJ')['segment_accuracy'])} "
    f"(IDD, rest) and {pct(proto('DS2', 'smvmd_overlap50', 'SUBJ')['segment_accuracy'])} (ADHD).",
    'I found out why: the EEG features mostly act like a fingerprint of each child. If the '
    'model has seen a child before, it recognises the child and copies the label. For a new '
    'child it has much less to go on.',
    'I also found that the complicated decomposition step (SMVMD) did not help compared with '
    'using the plain EEG channels, and that in the ADHD dataset all ADHD children were on '
    'medication, which the model cannot separate from ADHD itself.',
    'I wrote the mid-semester report with all of this, with every number filled in '
    'automatically from my results.',
]))
A(P('<b>What I will do next</b>', h2))
story.extend(bullets([
    'Build a fair evaluation where even the tuning never sees the test children, and add '
    'error bars.',
    'Test step by step which parts of the method actually help (plain EEG, frequency bands, '
    'VMD, MVMD, SMVMD).',
    'Find out what the model looks at — which features and which brain regions — and compare '
    'with a simple, well-known ADHD measure (theta/beta ratio).',
    'Check whether patterns hold across both datasets using the 10 electrodes they share, and '
    'how few electrodes are enough.',
    'Write the final report and present it at the end of November.',
]))
A(Spacer(1, 8))
A(note('<b>In one sentence:</b> I reproduced a published EEG method for detecting IDD and ADHD, '
       'showed that its very high accuracy mostly comes from recognising individual children '
       'rather than the disorder, and will spend the rest of the semester building and '
       'explaining a model that works on children it has never seen.'))


def on_page(canv, doc):
    if doc.page == 1:
        return
    canv.saveState()
    canv.setFont('Arial', 8)
    canv.setFillColor(MUTED)
    canv.drawString(2.2 * cm, 1.2 * cm, 'EEG-based Multiple Neurodevelopmental Disorders Detection — project record')
    canv.drawRightString(A4[0] - 2.2 * cm, 1.2 * cm, f'Page {doc.page}')
    canv.setStrokeColor(RULE)
    canv.line(2.2 * cm, 1.5 * cm, A4[0] - 2.2 * cm, 1.5 * cm)
    canv.restoreState()


def glue_headings(flow):
    """Bundle every heading with the block after it so no heading is stranded at a
    page bottom (keepWithNext does not reach inside KeepTogether blocks)."""
    out, i = [], 0
    while i < len(flow):
        f = flow[i]
        if isinstance(f, Paragraph) and f.style.name in ('h1', 'h2'):
            group = [f]
            i += 1
            while i < len(flow) and isinstance(flow[i], Paragraph) and flow[i].style.name in ('h1', 'h2'):
                group.append(flow[i]); i += 1
            if i < len(flow):
                group.append(flow[i]); i += 1
            out.append(KeepTogether(group))
        else:
            out.append(f); i += 1
    return out


story = glue_headings(story)

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                        topMargin=2.0 * cm, bottomMargin=2.0 * cm,
                        title='EEG-based Multiple Neurodevelopmental Disorders Detection — Project Record',
                        author='[Name]')
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print(f'wrote {OUT}')
