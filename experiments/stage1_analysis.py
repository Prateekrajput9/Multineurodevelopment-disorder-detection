"""
stage1_analysis.py - Evaluation-rigour analyses on top of the reproduced pipeline.

Everything here reuses the cached feature matrices written by the four main
experiments (results/features/*_features.npz). Only the raw-channel baseline
has to touch the EEG again, and it caches its own features too.

Parts (run all, or one at a time):

    python -m experiments.stage1_analysis                 # everything
    python -m experiments.stage1_analysis --part eval     # (b) (e) (g) protocols
    python -m experiments.stage1_analysis --part raw      # (f) raw-channel baseline
    python -m experiments.stage1_analysis --part embed    # (c) t-SNE + separability
    python -m experiments.stage1_analysis --part modes    # (d) SMVMD mode counts

Protocols
---------
SEG10          StratifiedKFold(10) over segments. Scaling, mRMR and KNN are all
               fitted inside each fold (Pipeline). This is the paper's split
               with selection moved inside the fold.
SEG10_GLOBAL   StratifiedKFold(10) over segments, but the mRMR ranking is fitted
               ONCE on all segments before CV, as the paper describes. Measures
               selection leakage on top of segment leakage (DS-1 only; DS-2 keeps
               all 171 features, so ranking cannot matter there).
SUBJ           Subject-wise. DS-1: leave-one-subject-out (14 folds).
               DS-2: GroupKFold(10). Subject IDs passed as groups; scaling,
               mRMR and KNN fitted inside each fold.

KNN hyperparameters are the paper's Table II values for every protocol. They are
NOT re-tuned per fold here; nested tuning is planned work.

For every protocol we report segment-level metrics and subject-level accuracy by
majority vote over each child's out-of-fold segment predictions.

Outputs
-------
results/stage1/results_summary.json   every number (single source of truth)
results/stage1/results_summary.md     the same, human-readable
results/stage1/figures/*.pdf|png
results/stage1/tables/*.csv
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, silhouette_score)
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     StratifiedKFold)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.knn_model import squared_inverse_weights
from src.mrmr import compute_mrmr_ranking
from src.utils import load_config, set_random_seed, setup_logger

logger = setup_logger('Stage1', 'results/logs/stage1_analysis.log')

OUT = Path('results/stage1')
FIG = OUT / 'figures'
TAB = OUT / 'tables'
SUMMARY = OUT / 'results_summary.json'
FEAT = Path('results/features')

SEED = 42

# experiment key -> (config knn key, config mRMR key, dataset)
EXPERIMENTS = {
    'DS1_Rest': ('rest', 'n_features_rest', 'ds1'),
    'DS1_Music': ('music', 'n_features_music', 'ds1'),
    'DS1_RestMusic': ('rest_music', 'n_features_rest_music', 'ds1'),
    'DS2': (None, 'n_features', 'ds2'),
}
DS1_KEYS = ['DS1_Rest', 'DS1_Music', 'DS1_RestMusic']


# ─────────────────────────────────────────────────────────────────────────────
# Summary file (merge-on-write so parts can run independently)
# ─────────────────────────────────────────────────────────────────────────────

def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def load_summary():
    if SUMMARY.exists():
        return json.loads(SUMMARY.read_text(encoding='utf-8'))
    return {}


def update_summary(section, payload):
    s = load_summary()
    s[section] = _jsonable(payload)
    s['_updated'] = datetime.now().isoformat(timespec='seconds')
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(s, indent=2), encoding='utf-8')
    logger.info(f"results_summary.json <- [{section}]")


def save_fig(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=200, bbox_inches='tight')
    logger.info(f"figure -> {FIG / name}.pdf|png")


def save_csv(name, rows):
    import csv
    if not rows:
        return
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / f'{name}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline components
# ─────────────────────────────────────────────────────────────────────────────

class MRMRSelector(BaseEstimator, TransformerMixin):
    """mRMR as a Pipeline step, so it is refitted on each training fold.

    If `fixed_indices` is given, the step does NOT learn anything and simply
    applies that precomputed ranking (used for the SEG10_GLOBAL protocol, where
    the ranking was deliberately fitted on all data, as in the paper).
    """

    def __init__(self, n_features=80, fixed_indices=None,
                 max_samples_redundancy=2000, random_state=SEED):
        self.n_features = n_features
        self.fixed_indices = fixed_indices
        self.max_samples_redundancy = max_samples_redundancy
        self.random_state = random_state

    def fit(self, X, y):
        p = X.shape[1]
        if self.fixed_indices is not None:
            self.idx_ = np.asarray(self.fixed_indices)[:self.n_features]
        elif self.n_features >= p:
            self.idx_ = np.arange(p)
        else:
            self.idx_ = compute_mrmr_ranking(
                X, y, n_features=self.n_features,
                max_samples_redundancy=self.max_samples_redundancy,
                random_seed=self.random_state)[:self.n_features]
        return self

    def transform(self, X):
        return X[:, self.idx_]


def knn_params_for(cfg, exp):
    knn_key, nf_key, ds = EXPERIMENTS[exp]
    kp = cfg['ds1']['knn'][knn_key] if ds == 'ds1' else cfg['ds2']['knn']
    nf = cfg['ds1']['mrmr'][nf_key] if ds == 'ds1' else cfg['ds2']['mrmr'][nf_key]
    return dict(kp), int(nf)


def build_pipeline(knn_params, n_features, fixed_indices=None):
    metric = {'cityblock': 'manhattan'}.get(knn_params['distance'],
                                           knn_params['distance'])
    w = knn_params['weights']
    weights = (squared_inverse_weights if w == 'squared_inverse'
               else 'uniform' if w in ('uniform', 'equal') else 'distance')
    return Pipeline([
        ('scale', StandardScaler() if knn_params.get('standardize') else 'passthrough'),
        ('mrmr', MRMRSelector(n_features=n_features, fixed_indices=fixed_indices)),
        # n_jobs=1: folds are already parallelised across cores below.
        ('knn', KNeighborsClassifier(n_neighbors=knn_params['n_neighbors'],
                                     metric=metric, weights=weights, n_jobs=1)),
    ])


def _fit_fold(pipe, X, y, tr, te):
    m = clone(pipe).fit(X[tr], y[tr])
    return te, m.predict(X[te])


def run_protocol(X, y, groups, pipe, splitter, n_jobs=-1):
    splits = list(splitter.split(X, y, groups))
    out = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_fit_fold)(pipe, X, y, tr, te) for tr, te in splits)
    y_pred = np.full(len(y), -1)
    fold_acc = []
    for te, p in out:
        y_pred[te] = p
        fold_acc.append(accuracy_score(y[te], p))
    assert (y_pred >= 0).all(), 'some segments were never predicted'
    return y_pred, fold_acc, len(splits)


def score(y, y_pred, groups, fold_acc, n_folds):
    """Segment metrics + subject-level majority vote + per-subject accuracy."""
    res = {
        'n_folds': n_folds,
        'segment_accuracy': accuracy_score(y, y_pred),
        'segment_balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'segment_precision_weighted': precision_score(y, y_pred, average='weighted', zero_division=0),
        'segment_recall_weighted': recall_score(y, y_pred, average='weighted', zero_division=0),
        'segment_f1_weighted': f1_score(y, y_pred, average='weighted', zero_division=0),
        'segment_confusion_matrix': confusion_matrix(y, y_pred),
        'fold_accuracy_mean': float(np.mean(fold_acc)),
        'fold_accuracy_std': float(np.std(fold_acc)),
    }
    subj_rows, subj_true, subj_pred, ties = [], [], [], 0
    for s in np.unique(groups):
        m = groups == s
        true = int(np.bincount(y[m]).argmax())
        frac1 = float(np.mean(y_pred[m] == 1))
        if frac1 == 0.5:
            ties += 1
        vote = int(frac1 > 0.5)            # tie -> class 0, counted in `ties`
        subj_true.append(true)
        subj_pred.append(vote)
        subj_rows.append({'subject': str(s), 'label': true,
                          'n_segments': int(m.sum()),
                          'segment_accuracy': float(np.mean(y_pred[m] == y[m])),
                          'fraction_predicted_disorder': frac1,
                          'majority_vote': vote,
                          'vote_correct': int(vote == true)})
    subj_true, subj_pred = np.array(subj_true), np.array(subj_pred)
    res.update({
        'n_subjects': len(subj_rows),
        'subject_majority_vote_accuracy': accuracy_score(subj_true, subj_pred),
        'subject_majority_vote_correct': int((subj_true == subj_pred).sum()),
        'subject_confusion_matrix': confusion_matrix(subj_true, subj_pred, labels=[0, 1]),
        'subject_vote_ties': ties,
    })
    return res, subj_rows


# ─────────────────────────────────────────────────────────────────────────────
# Data access
# ─────────────────────────────────────────────────────────────────────────────

def load_cached(exp, kind='smvmd'):
    path = FEAT / (f'{exp}_features.npz' if kind == 'smvmd' else f'RAW_{exp}_features.npz')
    d = np.load(path, allow_pickle=False)
    return d['X'], d['y'].astype(int), d['groups'].astype(str), d['n_modes']


def window_position(exp, groups):
    """Index of each segment within its own recording (0, 1, 2, ...)."""
    if exp == 'DS2':
        import pandas as pd
        df = pd.read_csv('data/metadata/ds2_segments.csv')
        assert (df['subject_id'].astype(str).values == groups).all()
        return df['start_sample'].values // 320
    pos = np.zeros(len(groups), dtype=int)
    counters = {}
    for i, g in enumerate(groups):
        pos[i] = counters.get(g, 0)
        counters[g] = pos[i] + 1
    # DS-1 recordings are 46 windows long; rest+music stacks two per subject.
    return pos % 46


def zero_overlap_mask(exp, groups):
    """0%-overlap windows = even-indexed 50%-overlap windows of each recording.

    Window 640, step 320: the non-overlapping windows start at 0, 640, 1280, ...
    which are exactly positions 0, 2, 4, ... The strict boundary rule is the same
    for both, and SMVMD + features are computed per segment, so the cached
    features for these windows are identical to recomputing them.
    """
    return window_position(exp, groups) % 2 == 0


def subject_splitter(exp):
    return LeaveOneGroupOut() if exp.startswith('DS1') else GroupKFold(n_splits=10)


def segment_splitter():
    return StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)


# ─────────────────────────────────────────────────────────────────────────────
# (b) (e) (g): protocol comparison
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_feature_set(cfg, exp, X, y, groups, tag, global_rank=None):
    kp, nf = knn_params_for(cfg, exp)
    results, per_subject = {}, {}
    protos = [('SEG10', segment_splitter(), None),
              ('SUBJ', subject_splitter(exp), None)]
    if global_rank is not None and nf < X.shape[1]:
        protos.insert(1, ('SEG10_GLOBAL', segment_splitter(), global_rank))
    for name, splitter, fixed in protos:
        t0 = time.perf_counter()
        pipe = build_pipeline(kp, nf, fixed_indices=fixed)
        y_pred, fold_acc, nfold = run_protocol(X, y, groups, pipe, splitter)
        res, rows = score(y, y_pred, groups, fold_acc, nfold)
        res['runtime_sec'] = round(time.perf_counter() - t0, 1)
        results[name] = res
        per_subject[name] = rows
        save_csv(f'per_subject_{exp}_{tag}_{name}', rows)
        logger.info(f"[{exp} | {tag} | {name}] seg acc={res['segment_accuracy']*100:.2f}%  "
                    f"subject vote acc={res['subject_majority_vote_accuracy']*100:.2f}% "
                    f"({res['subject_majority_vote_correct']}/{res['n_subjects']})  "
                    f"[{res['runtime_sec']}s]")
    return {'n_segments': int(len(y)), 'n_subjects': int(len(np.unique(groups))),
            'n_features_total': int(X.shape[1]), 'n_features_used': nf,
            'knn_params': kp, 'protocols': results}, per_subject


def part_eval(cfg):
    out = {}
    for exp in EXPERIMENTS:
        X, y, groups, _ = load_cached(exp)
        rank = None
        if exp.startswith('DS1'):
            meta = json.load(open(f'results/metrics/{exp}.json', encoding='utf-8'))
            rank = (meta.get('mrmr_curve') or {}).get('ranked_feature_indices')
        full, ps_full = evaluate_feature_set(cfg, exp, X, y, groups, 'smvmd_overlap50', rank)
        m = zero_overlap_mask(exp, groups)
        zero, _ = evaluate_feature_set(cfg, exp, X[m], y[m], groups[m], 'smvmd_overlap0')
        out[exp] = {'smvmd_overlap50': full, 'smvmd_overlap0': zero}
        plot_per_subject(exp, ps_full['SUBJ'])
        update_summary('protocols', {**load_summary().get('protocols', {}), **{exp: out[exp]}})
    return out


def plot_per_subject(exp, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: (r['label'], r['segment_accuracy']))
    acc = [r['segment_accuracy'] * 100 for r in rows]
    cols = ['#d62728' if r['label'] == 1 else '#1f77b4' for r in rows]
    # Fixed aspect: the figure is placed at half page width in the report, so
    # widening it with the number of children would shrink its text.
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(rows)), acc, color=cols)
    if len(rows) <= 20:
        # Label every bar: a 0% subject otherwise looks like missing data.
        for i, a in enumerate(acc):
            ax.text(i, a + 1.5, f'{a:.0f}', ha='center', va='bottom', fontsize=7)
    ax.axhline(50, color='k', lw=0.8, ls='--')
    ax.set_ylim(0, 112)          # headroom for the value labels above 100% bars
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Segment accuracy (%)')
    ax.set_xlabel('Held-out subject (sorted within class)')
    dis = 'ADHD' if exp == 'DS2' else 'IDD'
    ctl = 'NC' if exp == 'DS2' else 'TDC'
    ax.set_title(f'{exp}: per-subject accuracy, subject-wise CV')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#1f77b4', label=ctl), Patch(color='#d62728', label=dis)],
              loc='lower right', fontsize=8)
    if len(rows) <= 20:
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r['subject'] for r in rows], rotation=90, fontsize=7)
    else:
        ax.set_xticks([])
    save_fig(fig, f'per_subject_accuracy_{exp}')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# (f) raw-channel baseline: same 9 features, no SMVMD
# ─────────────────────────────────────────────────────────────────────────────

def _raw_features_one(seg, fs):
    from src.features import extract_features_from_mode
    return np.concatenate([extract_features_from_mode(seg[c], c, fs) for c in range(seg.shape[0])])


def _raw_matrix(segments, fs):
    rows = Parallel(n_jobs=-1, backend='loky', batch_size=16)(
        delayed(_raw_features_one)(s, fs) for s in segments)
    return np.nan_to_num(np.vstack(rows), nan=0.0, posinf=0.0, neginf=0.0)


def build_raw_ds1(cfg):
    from src.data_loader import DS1Loader
    from src.preprocessing import preprocess_ds1
    from src.segmentation import segment_eeg
    ds1 = cfg['ds1']
    fs = ds1['sampling_freq']
    loader = DS1Loader(cfg['paths']['ds1_data_root'], mode=ds1['preprocessing']['mode'])
    files = loader.discover_files()
    per = {}
    for cond in ('rest', 'music'):
        segs, ys, gs = [], [], []
        for sid in sorted(files):
            if cond not in files[sid]:
                continue
            eeg, info = loader.load_subject(sid, cond, files[sid][cond])
            eeg = preprocess_ds1(eeg, fs=fs, mode=ds1['preprocessing']['mode'],
                                 config=ds1['preprocessing'])
            s, _ = segment_eeg(eeg=eeg, fs=fs, subject_id=sid, label=info['label'],
                               dataset='DS1', condition=cond)
            segs.append(s); ys += [info['label']] * len(s); gs += [sid] * len(s)
        per[cond] = (np.concatenate(segs), np.array(ys), np.array(gs))
    out = {}
    feats = {c: _raw_matrix(per[c][0], fs) for c in per}
    out['DS1_Rest'] = (feats['rest'], per['rest'][1], per['rest'][2])
    out['DS1_Music'] = (feats['music'], per['music'][1], per['music'][2])
    # Rest+music: same per-subject order as the SMVMD pipeline (music, then rest).
    Xr, yr, gr = out['DS1_Rest']; Xm, ym, gm = out['DS1_Music']
    X, Y, G = [], [], []
    for sid in sorted(set(gr.tolist())):
        for (Xc, yc, gc) in ((Xm, ym, gm), (Xr, yr, gr)):
            k = gc == sid
            X.append(Xc[k]); Y.append(yc[k]); G.append(gc[k])
    out['DS1_RestMusic'] = (np.vstack(X), np.concatenate(Y), np.concatenate(G))
    return out


def build_raw_ds2(cfg):
    from src.data_loader import DS2Loader
    from src.preprocessing import preprocess_ds2
    from src.segmentation import segment_dataset_ds2
    ds2 = cfg['ds2']
    fs = ds2['sampling_freq']
    eeg, meta = DS2Loader(cfg['paths']['ds2_adhd_dir']).load_all_subjects()
    pre = {s: preprocess_ds2(eeg=e, fs=fs, **dict(ds2['preprocessing'])) for s, e in eeg.items()}
    segs, df = segment_dataset_ds2(eeg_data=pre, metadata_list=meta, fs=fs,
                                   window_sec=cfg['segmentation']['window_sec'],
                                   overlap=cfg['segmentation']['overlap'])
    return {'DS2': (_raw_matrix(segs, fs), df['label'].values.astype(int),
                    df['subject_id'].astype(str).values)}


def part_raw(cfg):
    FEAT.mkdir(parents=True, exist_ok=True)
    built = {}
    need_ds1 = any(not (FEAT / f'RAW_{e}_features.npz').exists() for e in DS1_KEYS)
    if need_ds1:
        t0 = time.perf_counter(); built.update(build_raw_ds1(cfg))
        logger.info(f"raw DS-1 features in {time.perf_counter()-t0:.0f}s")
    if not (FEAT / 'RAW_DS2_features.npz').exists():
        t0 = time.perf_counter(); built.update(build_raw_ds2(cfg))
        logger.info(f"raw DS-2 features in {time.perf_counter()-t0:.0f}s")
    for exp, (X, y, g) in built.items():
        # groups as fixed-width unicode, not object dtype: object arrays need
        # pickle to load, and the cache is read with allow_pickle=False.
        np.savez_compressed(FEAT / f'RAW_{exp}_features.npz', X=X, y=y,
                            groups=np.asarray(g).astype(str),
                            n_modes=np.zeros(0, dtype=np.int32))

    out = {}
    for exp in EXPERIMENTS:
        X, y, g, _ = load_cached(exp, 'raw')
        _, ys, gs, _ = load_cached(exp)          # alignment check vs SMVMD cache
        aligned = bool(len(y) == len(ys) and (y == ys).all() and (g == gs).all())
        if not aligned:
            logger.warning(f"{exp}: raw segments not aligned with SMVMD cache "
                           f"({len(y)} vs {len(ys)}); still evaluated on their own.")
        res, _ = evaluate_feature_set(cfg, exp, X, y, g, 'raw_overlap50')
        res['aligned_with_smvmd_segments'] = aligned
        out[exp] = res
        update_summary('raw_baseline', {**load_summary().get('raw_baseline', {}), exp: res})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# (c) embeddings and class-vs-subject separability
# ─────────────────────────────────────────────────────────────────────────────

def part_embed(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.model_selection import cross_val_predict

    stats = {}
    emb = {}
    for exp in EXPERIMENTS:
        X, y, g, _ = load_cached(exp)
        Z = StandardScaler().fit_transform(X)
        # How strongly do the features encode WHO the child is, vs. the class?
        sil_class = silhouette_score(Z, y, random_state=SEED,
                                     sample_size=min(len(y), 4000))
        sil_subj = silhouette_score(Z, g, random_state=SEED,
                                    sample_size=min(len(y), 4000))
        # Subject identification: 1-NN on standardised features, segment-wise 10-fold.
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
        pid = cross_val_predict(
            Pipeline([('s', StandardScaler()), ('k', KNeighborsClassifier(1))]),
            X, g, cv=skf, n_jobs=-1)
        subj_id_acc = accuracy_score(g, pid)
        n_subj = len(np.unique(g))
        stats[exp] = {
            'silhouette_by_class': float(sil_class),
            'silhouette_by_subject': float(sil_subj),
            'subject_identification_accuracy_1nn': float(subj_id_acc),
            'subject_identification_chance': 1.0 / n_subj,
            'n_subjects': n_subj,
            'n_segments': int(len(y)),
            'note': 'standardised SMVMD-integrated features; silhouette on '
                    'Euclidean distance (sampled to <=4000 points); subject ID '
                    'decoded with 1-NN under segment-wise 10-fold CV',
        }
        logger.info(f"[{exp}] silhouette class={sil_class:.3f} subject={sil_subj:.3f} "
                    f"subject-ID 1NN acc={subj_id_acc*100:.1f}% (chance {100/n_subj:.1f}%)")
        if exp in ('DS1_Rest', 'DS2'):
            P = PCA(n_components=min(50, Z.shape[1]), random_state=SEED).fit_transform(Z)
            emb[exp] = (TSNE(n_components=2, perplexity=30, init='pca',
                             random_state=SEED).fit_transform(P), y, g)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for row, exp in enumerate(('DS1_Rest', 'DS2')):
        E, y, g = emb[exp]
        dis, ctl = ('ADHD', 'NC') if exp == 'DS2' else ('IDD', 'TDC')
        ax = axes[row, 0]
        for lab, c, name in ((0, '#1f77b4', ctl), (1, '#d62728', dis)):
            k = y == lab
            ax.scatter(E[k, 0], E[k, 1], s=4, c=c, alpha=0.6, label=name, lw=0)
        ax.legend(markerscale=4, fontsize=8)
        ax.set_title(f'{exp}: coloured by class')
        ax = axes[row, 1]
        subs = np.unique(g)
        cmap = plt.get_cmap('tab20' if len(subs) <= 20 else 'gist_ncar')
        for i, s in enumerate(subs):
            k = g == s
            ax.scatter(E[k, 0], E[k, 1], s=4, color=cmap(i / max(1, len(subs) - 1)),
                       alpha=0.7, lw=0)
        ax.set_title(f'{exp}: coloured by subject ({len(subs)} subjects)')
        for a in axes[row]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle('t-SNE of energy-integrated SMVMD features', y=0.995)
    fig.tight_layout()
    save_fig(fig, 'tsne_class_vs_subject')
    plt.close(fig)
    update_summary('embedding', stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# (d) SMVMD mode-count distribution
# ─────────────────────────────────────────────────────────────────────────────

def part_modes(cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import mannwhitneyu

    stats = {}
    panels = {'DS-1 (rest + music)': 'DS1_RestMusic', 'DS-2': 'DS2'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, (title, exp) in zip(axes, panels.items()):
        _, y, _, k = load_cached(exp)
        dis, ctl = ('ADHD', 'NC') if exp == 'DS2' else ('IDD', 'TDC')
        bins = np.arange(0.5, max(21, k.max() + 1.5))
        for lab, c, name in ((0, '#1f77b4', ctl), (1, '#d62728', dis)):
            ax.hist(k[y == lab], bins=bins, density=True, alpha=0.55, color=c,
                    label=f'{name} (n={int((y == lab).sum())})')
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel('Number of SMVMD modes K per segment')
        ax.set_ylabel('Fraction of segments')
        ax.set_title(title)
        ax.legend(fontsize=8)
        u = mannwhitneyu(k[y == 1], k[y == 0], alternative='two-sided')
        stats[exp] = {
            'overall': {'mean': float(k.mean()), 'std': float(k.std()),
                        'median': float(np.median(k)), 'min': int(k.min()), 'max': int(k.max()),
                        'pct_at_cap_20': float(np.mean(k >= 20) * 100)},
            ctl: {'mean': float(k[y == 0].mean()), 'std': float(k[y == 0].std()),
                  'median': float(np.median(k[y == 0]))},
            dis: {'mean': float(k[y == 1].mean()), 'std': float(k[y == 1].std()),
                  'median': float(np.median(k[y == 1]))},
            'mannwhitney_U': float(u.statistic), 'mannwhitney_p': float(u.pvalue),
            'note': 'segments are not independent (overlapping windows, many per '
                    'child), so the p-value is descriptive only',
        }
    fig.tight_layout()
    save_fig(fig, 'smvmd_mode_count_histogram')
    plt.close(fig)
    update_summary('mode_counts', stats)
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Reproduction numbers (copied verbatim from the four main runs)
# ─────────────────────────────────────────────────────────────────────────────

def part_reproduction():
    paper = {'DS1_Rest': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
             'DS1_Music': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
             'DS1_RestMusic': {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0},
             'DS2': {'accuracy': 0.9917, 'precision': 0.9904, 'recall': 0.9907, 'f1': 0.9905}}
    out = {}
    for exp in EXPERIMENTS:
        d = json.load(open(f'results/metrics/{exp}.json', encoding='utf-8'))
        m = d['metrics_paper_protocol']
        out[exp] = {
            'source': f'results/metrics/{exp}.json (run {d["run_at"]})',
            'ours': {k: m[k] for k in ('accuracy', 'precision', 'recall', 'f1',
                                       'mcc', 'kappa', 'gdr', 'gmean') if k in m},
            'paper': paper[exp],
            'confusion_matrix': m['confusion_matrix'],
            'n_segments': d['dataset']['n_segments'],
            'n_features_used': d['dataset']['n_features_used'],
            'knn_params': d['knn_params'],
            'mrmr_empirical_optimum': (d.get('mrmr_curve') or {}).get('optimal_n_empirical'),
            'mode_stats': d.get('mode_stats'),
            'classifier_comparison': d.get('classifier_comparison'),
            'protocol': 'segment-wise stratified 10-fold, mRMR inside folds, '
                        'Table II KNN params (no re-tuning)',
        }
    update_summary('reproduction', out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Human-readable summary
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown():
    s = load_summary()
    L = ['# Stage 1 results summary', '',
         f'Generated {s.get("_updated", "?")} from `results/stage1/results_summary.json`.',
         'Every number below is copied from that file.', '']
    pct = lambda v: '—' if v is None else f'{v*100:.2f}%'

    if 'reproduction' in s:
        L += ['## Reproduction (segment-wise 10-fold) vs paper', '',
              '| Exp | Acc ours | Acc paper | F1 ours | F1 paper |', '|---|---|---|---|---|']
        for e, r in s['reproduction'].items():
            L.append(f"| {e} | {pct(r['ours']['accuracy'])} | {pct(r['paper']['accuracy'])} | "
                     f"{pct(r['ours']['f1'])} | {pct(r['paper']['f1'])} |")
        L.append('')

    def proto_table(section, title, feature_tag=None):
        if section not in s:
            return
        L.extend([f'## {title}', '',
                  '| Exp | Features | Protocol | Folds | Seg acc | Bal acc | Seg F1 | Subject vote acc |',
                  '|---|---|---|---|---|---|---|---|'])
        for e, block in s[section].items():
            sets = {feature_tag: block} if feature_tag else block
            for tag, fs in sets.items():
                for p, r in fs['protocols'].items():
                    L.append(f"| {e} | {tag} | {p} | {r['n_folds']} | {pct(r['segment_accuracy'])} | "
                             f"{pct(r['segment_balanced_accuracy'])} | {pct(r['segment_f1_weighted'])} | "
                             f"{pct(r['subject_majority_vote_accuracy'])} "
                             f"({r['subject_majority_vote_correct']}/{r['n_subjects']}) |")
        L.append('')

    proto_table('protocols', 'Segment-wise vs subject-wise (SMVMD features, 50% and 0% overlap)')
    proto_table('raw_baseline', 'Raw-channel baseline (same 9 features, no SMVMD)', 'raw_overlap50')

    if 'embedding' in s:
        L += ['## Class vs subject structure in the feature space', '',
              '| Exp | Silhouette (class) | Silhouette (subject) | Subject-ID 1-NN acc | Chance |',
              '|---|---|---|---|---|']
        for e, r in s['embedding'].items():
            L.append(f"| {e} | {r['silhouette_by_class']:.3f} | {r['silhouette_by_subject']:.3f} | "
                     f"{pct(r['subject_identification_accuracy_1nn'])} | {pct(r['subject_identification_chance'])} |")
        L.append('')
    if 'mode_counts' in s:
        L += ['## SMVMD modes per segment', '']
        for e, r in s['mode_counts'].items():
            cls = [k for k in r if k not in ('overall', 'mannwhitney_U', 'mannwhitney_p', 'note')]
            L.append(f"- **{e}**: overall mean {r['overall']['mean']:.2f} (sd {r['overall']['std']:.2f}), "
                     f"range {r['overall']['min']}–{r['overall']['max']}, at cap 20: "
                     f"{r['overall']['pct_at_cap_20']:.1f}%; " +
                     '; '.join(f"{c} mean {r[c]['mean']:.2f}" for c in cls) +
                     f"; Mann-Whitney p={r['mannwhitney_p']:.3g} (descriptive)")
        L.append('')
    (OUT / 'results_summary.md').write_text('\n'.join(L), encoding='utf-8')
    logger.info(f"-> {OUT / 'results_summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', default='all',
                    choices=['all', 'reproduction', 'eval', 'raw', 'embed', 'modes', 'md'])
    a = ap.parse_args()
    cfg = load_config('configs/config.yaml')
    set_random_seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    parts = ['reproduction', 'modes', 'embed', 'eval', 'raw'] if a.part == 'all' else [a.part]
    for p in parts:
        t0 = time.perf_counter()
        logger.info(f"===== part: {p} =====")
        {'reproduction': part_reproduction, 'eval': lambda: part_eval(cfg),
         'raw': lambda: part_raw(cfg), 'embed': lambda: part_embed(cfg),
         'modes': lambda: part_modes(cfg), 'md': lambda: None}[p]()
        logger.info(f"===== part {p} done in {time.perf_counter()-t0:.0f}s =====")
        write_markdown()


if __name__ == '__main__':
    main()
