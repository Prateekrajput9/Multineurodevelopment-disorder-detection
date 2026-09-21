"""
evaluation.py - Evaluation, Metrics, and Results Reporting

Paper: "Electroencephalogram-Based Unified Approach for Multiple Neurodevelopmental
        Disorders Detection in Children Using SMVMD"
Authors: Ujjawal Chandela, Kazi Newaj Faisal, Rishi Raj Sharma
IEEE TCDS, 2025

Evaluation metrics [Paper explicit]:
  DS-1: Accuracy, Precision, Recall, F1-score
  DS-2: Accuracy, Precision, Recall, F1-score, MCC, Cohen's Kappa, GDR, G-mean

Additional metrics (NOT from paper, clearly labeled):
  - ROC curve and AUC
  - Precision-Recall curve and AUC

Paper targets:
  DS-1 Rest/Music/Rest+Music: 100% all metrics
  DS-2: Accuracy=99.17%, Precision=99.04%, Recall=99.07%, F1=99.05%
        MCC=0.983, Kappa=0.983, GDR≈99.239, G-mean≈0.9915
"""

import os
import json
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Paper Targets (for comparison)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_TARGETS = {
    'DS1_Rest': {'accuracy': 1.00, 'precision': 1.00, 'recall': 1.00, 'f1': 1.00},
    'DS1_Music': {'accuracy': 1.00, 'precision': 1.00, 'recall': 1.00, 'f1': 1.00},
    'DS1_RestMusic': {'accuracy': 1.00, 'precision': 1.00, 'recall': 1.00, 'f1': 1.00},
    'DS2': {
        'accuracy': 0.9917, 'precision': 0.9904,
        'recall': 0.9907, 'f1': 0.9905,
        'mcc': 0.983, 'kappa': 0.983,
        'gdr': 99.239, 'gmean': 0.9915,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_proba: Optional[np.ndarray] = None,
                         dataset: str = 'DS2',
                         class_names: Optional[List[str]] = None
                         ) -> Dict[str, Any]:
    """Compute all evaluation metrics for one experiment.

    Paper metrics [explicit]:
    - Accuracy = (TP + TN) / (TP + TN + FP + FN)
    - Precision = TP / (TP + FP)
    - Recall = TP / (TP + FN)
    - F1 = 2 × Precision × Recall / (Precision + Recall)
    - MCC (DS-2 only): Matthews Correlation Coefficient
    - Cohen's Kappa (DS-2 only)
    - GDR (DS-2 only): Good Detection Rate
    - G-mean (DS-2 only): √(sensitivity × specificity)

    Additional metrics [NOT from paper, labeled as such]:
    - ROC-AUC
    - PR-AUC

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for positive class (optional).
        dataset: 'DS1' or 'DS2' (determines which metrics to compute).
        class_names: For confusion matrix labeling.

    Returns:
        Dict with all computed metrics.
    """
    if class_names is None:
        class_names = ['NC/TDC (0)', 'ADHD/IDD (1)']

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # Paper metrics
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    results = {
        # Paper-defined metrics
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'confusion_matrix': cm,
        'class_names': class_names,
    }

    # DS-2 specific metrics [Paper explicit]
    if dataset == 'DS2' or dataset.startswith('DS2'):
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)

        from src.utils import compute_gdr, compute_gmean
        if cm.shape == (2, 2):
            gdr = compute_gdr(cm)
            gmean = compute_gmean(cm)
        else:
            gdr = 0.0
            gmean = 0.0

        results.update({
            'mcc': float(mcc),
            'kappa': float(kappa),
            'gdr': float(gdr),
            'gmean': float(gmean),
        })

    # Additional metrics [NOT from paper - clearly labeled]
    if y_proba is not None and cm.shape == (2, 2):
        try:
            roc_auc = roc_auc_score(y_true, y_proba)
            pr_auc = average_precision_score(y_true, y_proba)
            results['roc_auc'] = float(roc_auc)
            results['pr_auc'] = float(pr_auc)
            results['_note_roc_pr'] = (
                "ROC-AUC and PR-AUC are additional reporting metrics. "
                "They are NOT reported in the original paper."
            )
        except Exception as e:
            logger.debug(f"Could not compute ROC/PR AUC: {e}")

    return results


def compare_with_paper_targets(metrics: Dict[str, Any],
                                 experiment: str) -> Dict[str, Any]:
    """Compare computed metrics with paper-reported targets.

    Args:
        metrics: Dict of computed metrics.
        experiment: Key into PAPER_TARGETS (e.g., 'DS2', 'DS1_Rest').

    Returns:
        Dict with comparison results.
    """
    targets = PAPER_TARGETS.get(experiment, {})
    if not targets:
        logger.warning(
            f"No paper targets registered for experiment key '{experiment}'. "
            f"Known keys: {sorted(PAPER_TARGETS)}. The report will show no "
            f"paper comparison for this run."
        )
    comparison = {}

    for metric, target in targets.items():
        if metric in metrics:
            actual = metrics[metric]
            diff = actual - target if not isinstance(actual, np.ndarray) else None
            comparison[metric] = {
                'paper': target,
                'ours': actual,
                'diff': diff,
                'match': abs(diff) < 0.005 if diff is not None else None,
            }

    return comparison


def print_results_table(metrics: Dict[str, Any],
                          experiment: str,
                          comparison: Optional[Dict] = None) -> str:
    """Format results as a comparison table.

    Args:
        metrics: Computed metrics dict.
        experiment: Experiment name for header.
        comparison: Optional comparison with paper targets.

    Returns:
        Formatted string table.
    """
    lines = [
        f"\n{'='*65}",
        f"Results: {experiment}",
        f"{'='*65}",
        f"{'Metric':<20} {'Ours':>12} {'Paper':>12} {'Match':>8}",
        f"{'-'*65}",
    ]

    paper_metrics = ['accuracy', 'precision', 'recall', 'f1',
                      'mcc', 'kappa', 'gdr', 'gmean']

    for m in paper_metrics:
        if m not in metrics:
            continue

        val = metrics[m]
        if isinstance(val, (int, float)):
            val_str = f"{val*100:.2f}%" if m not in ['mcc', 'kappa', 'gmean', 'gdr'] else f"{val:.4f}"
        else:
            continue

        if comparison and m in comparison:
            comp = comparison[m]
            paper_val = comp['paper']
            paper_str = f"{paper_val*100:.2f}%" if m not in ['mcc', 'kappa', 'gmean', 'gdr'] else f"{paper_val:.4f}"
            match_str = "✓" if comp.get('match') else "✗"
        else:
            paper_str = "N/A"
            match_str = "-"

        lines.append(f"{m.upper():<20} {val_str:>12} {paper_str:>12} {match_str:>8}")

    if 'roc_auc' in metrics:
        lines.append(f"{'-'*65}")
        lines.append(f"[Additional - NOT paper metrics]")
        lines.append(f"{'ROC-AUC':<20} {metrics['roc_auc']*100:>11.2f}%")
        if 'pr_auc' in metrics:
            lines.append(f"{'PR-AUC':<20} {metrics['pr_auc']*100:>11.2f}%")

    lines.append(f"{'='*65}")
    table = "\n".join(lines)
    logger.info(table)
    return table


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_paper(cm: np.ndarray,
                                  class_names: List[str],
                                  title: str,
                                  save_path: str) -> None:
    """Plot confusion matrix matching paper style (Fig. 6).

    Args:
        cm: 2×2 confusion matrix.
        class_names: Class label names.
        title: Figure title.
        save_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Display in percent
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Percentage (%)', fontsize=10)

    tick_labels = [f"{cls}\n({cm.sum(axis=1)[i]} samples)"
                   for i, cls in enumerate(class_names)]
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted Label',
        ylabel='True Label',
        title=title,
    )

    thresh = 50.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)",
                    ha='center', va='center', fontsize=12,
                    color='white' if cm_pct[i,j] > thresh else 'black',
                    fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def plot_roc_curve(y_true: np.ndarray,
                    y_proba: np.ndarray,
                    title: str,
                    save_path: str) -> None:
    """Plot ROC curve.

    [NOT a paper metric - additional diagnostic plot]

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities for positive class.
        title: Figure title.
        save_path: Save path.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set(
        xlim=[0, 1], ylim=[0, 1.05],
        xlabel='False Positive Rate',
        ylabel='True Positive Rate',
        title=f'{title}\n[Additional metric - NOT in paper]',
    )
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"ROC curve saved: {save_path}")


def generate_full_report(results_dict: Dict[str, Dict],
                          save_dir: str) -> None:
    """Generate full evaluation report with all experiments.

    Args:
        results_dict: Mapping experiment_name → metrics_dict.
        save_dir: Directory to save report files.
    """
    os.makedirs(save_dir, exist_ok=True)

    report_lines = [
        "# EEG Neurodevelopmental Disorder Detection - Results Report",
        "",
        "## Reproduction of: Chandela, Faisal & Sharma (2025), IEEE TCDS",
        "",
        "**DISCLAIMER:** These are reproduction results. See docs/reproducibility_notes.md",
        "for detailed comparison with the paper's reported values.",
        "",
    ]

    for exp_name, metrics in results_dict.items():
        report_lines.append(f"## {exp_name}")
        report_lines.append("")

        # Paper-defined metrics
        for m in ['accuracy', 'precision', 'recall', 'f1',
                   'mcc', 'kappa', 'gdr', 'gmean']:
            if m in metrics:
                val = metrics[m]
                if not isinstance(val, np.ndarray):
                    report_lines.append(f"- **{m.upper()}**: {val:.4f}")

        # Compare with paper targets
        if exp_name in PAPER_TARGETS:
            comparison = compare_with_paper_targets(metrics, exp_name)
            report_lines.append("")
            report_lines.append("### Comparison with Paper Targets")
            report_lines.append("")
            report_lines.append("| Metric | Paper | Ours | Match |")
            report_lines.append("|--------|-------|------|-------|")
            for m, comp in comparison.items():
                match_emoji = "✅" if comp.get('match') else "⚠️"
                report_lines.append(
                    f"| {m.upper()} | {comp['paper']:.4f} | "
                    f"{comp['ours']:.4f} | {match_emoji} |"
                )

        report_lines.append("")

        # Save confusion matrix
        if 'confusion_matrix' in metrics and metrics['confusion_matrix'] is not None:
            cm = metrics['confusion_matrix']
            cm_path = os.path.join(save_dir, 'confusion_matrices',
                                    f'cm_{exp_name}.png')
            class_names = metrics.get('class_names', ['Class 0', 'Class 1'])
            plot_confusion_matrix_paper(
                cm=cm,
                class_names=class_names,
                title=f"Confusion Matrix - {exp_name}",
                save_path=cm_path,
            )

    # Save report
    report_path = os.path.join(save_dir, 'final_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    logger.info(f"Report saved: {report_path}")

    # Save metrics as JSON
    json_metrics = {}
    for exp, mets in results_dict.items():
        json_metrics[exp] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in mets.items()
            if not isinstance(v, np.ndarray) or k == 'fold_accuracies'
        }

    json_path = os.path.join(save_dir, 'metrics_all.json')
    with open(json_path, 'w') as f:
        json.dump(json_metrics, f, indent=2, default=str)
    logger.info(f"Metrics JSON saved: {json_path}")
