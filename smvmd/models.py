"""
Classification and Evaluation Models Module
===========================================
Unified classification pipelines for pediatric neurodevelopmental disorder detection
supporting exact Table II KNN configurations, SVM, Decision Trees, and Ensembles.
"""

from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    all_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute accuracy, sensitivity, specificity, precision, F1, Cohen's kappa, ROC-AUC."""
    if all_classes is None:
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        if len(unique_classes) == 1:
            all_classes = [0, 1] if unique_classes[0] in (0, 1) else [unique_classes[0]]
        else:
            all_classes = list(unique_classes)

    num_classes = len(all_classes)
    acc = float(accuracy_score(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=all_classes)) if len(np.unique(y_true)) > 1 else 1.0
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    if num_classes <= 2:
        prec = float(precision_score(y_true, y_pred, average="binary", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="binary", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            spec = float(tn / (tn + fp + 1e-12))
            sens = float(tp / (tp + fn + 1e-12))
        else:
            spec = 1.0
            sens = rec

        auc_score = 0.5
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob
                auc_score = float(roc_auc_score(y_true, prob_pos))
            except Exception:
                auc_score = 0.5

        return {
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1_score": f1,
            "cohen_kappa": kappa,
            "roc_auc": auc_score,
            "confusion_matrix": cm,
            "class_names": class_names,
        }
    else:
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        auc_score = 0.5
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                auc_score = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=all_classes))
            except Exception:
                auc_score = 0.5

        sens_list, spec_list = [], []
        for i in range(cm.shape[0]):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - (tp + fn + fp)
            sens_list.append(tp / (tp + fn + 1e-12))
            spec_list.append(tn / (tn + fp + 1e-12))

        return {
            "accuracy": acc,
            "sensitivity": float(np.mean(sens_list)),
            "specificity": float(np.mean(spec_list)),
            "precision": prec_macro,
            "f1_score": f1_macro,
            "f1_weighted": f1_weighted,
            "cohen_kappa": kappa,
            "roc_auc": auc_score,
            "confusion_matrix": cm,
            "class_names": class_names,
        }


def _squared_inverse_weight(distances):
    """Squared inverse distance weighting: w = 1 / (d^2 + 1e-6)"""
    return 1.0 / ((distances ** 2) + 1e-6)


class NeuroDisorderClassifier:
    """
    Wrapper for classification models supporting exact Table II configurations:
    - DS-1 Rest: K=2, cityblock, uniform, standardize=True
    - DS-1 Music: K=2, cityblock, squared_inverse, standardize=False
    - DS-1 Rest+Music: K=1, euclidean, squared_inverse, standardize=True
    - DS-2 ADHD: K=2, cosine, squared_inverse, standardize=False
    """

    def __init__(
        self,
        classifier_type: str = "knn",
        preset: Optional[str] = None,  # 'ds1_rest', 'ds1_music', 'ds1_combined', 'ds2_adhd'
        n_neighbors: int = 2,
        metric: str = "minkowski",
        weights: Union[str, Any] = "distance",
        standardize: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        self.classifier_type = classifier_type.lower()
        self.random_state = random_state

        if preset == "ds1_rest":
            n_neighbors, metric, weights, standardize = 2, "manhattan", "uniform", True
        elif preset == "ds1_music":
            n_neighbors, metric, weights, standardize = 2, "manhattan", _squared_inverse_weight, False
        elif preset == "ds1_combined":
            n_neighbors, metric, weights, standardize = 1, "euclidean", _squared_inverse_weight, True
        elif preset == "ds2_adhd":
            n_neighbors, metric, weights, standardize = 2, "cosine", _squared_inverse_weight, False

        self.standardize = standardize
        self.scaler = StandardScaler() if standardize else None

        if self.classifier_type == "knn":
            self.model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                metric=metric,
                weights=weights,
                **kwargs,
            )
        elif self.classifier_type in ("dt", "decision_tree"):
            self.model = DecisionTreeClassifier(random_state=random_state)
        elif self.classifier_type == "svm":
            self.model = SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 10.0),
                probability=True,
                random_state=random_state,
            )
        elif self.classifier_type in ("rf", "random_forest", "ensemble"):
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 150),
                random_state=random_state,
            )
        elif self.classifier_type in ("mlp", "neural_net"):
            self.model = MLPClassifier(
                hidden_layer_sizes=kwargs.get("hidden_layer_sizes", (64, 32)),
                max_iter=kwargs.get("max_iter", 500),
                random_state=random_state,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_in = self.scaler.fit_transform(X) if self.scaler else X
        self.model.fit(X_in, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_in = self.scaler.transform(X) if self.scaler else X
        return self.model.predict(X_in)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_in = self.scaler.transform(X) if self.scaler else X
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_in)
        else:
            preds = self.predict(X)
            classes = np.unique(preds)
            probs = np.zeros((len(preds), len(classes)))
            for idx, c in enumerate(preds):
                probs[idx, c] = 1.0
            return probs


def evaluate_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    classifier_type: str = "knn",
    n_splits: int = 10,  # 10-fold CV from Section III-A
    random_state: int = 42,
    class_names: Optional[List[str]] = None,
    **clf_kwargs,
) -> Dict[str, Any]:
    """
    Perform Stratified 10-Fold Cross-Validation as per Section III-A.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_y_true, all_y_pred, all_y_prob = [], [], []
    fold_metrics = []
    all_classes = sorted(list(np.unique(y)))

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = NeuroDisorderClassifier(classifier_type=classifier_type, random_state=random_state + fold_idx, **clf_kwargs)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        fold_m = compute_metrics(y_test, y_pred, y_prob, class_names=class_names, all_classes=all_classes)
        fold_metrics.append(fold_m)

    overall_metrics = compute_metrics(
        np.array(all_y_true),
        np.array(all_y_pred),
        np.array(all_y_prob),
        class_names=class_names,
        all_classes=all_classes,
    )

    overall_metrics["accuracy_std"] = float(np.std([m["accuracy"] for m in fold_metrics]))
    overall_metrics["f1_score_std"] = float(np.std([m["f1_score"] for m in fold_metrics]))
    overall_metrics["fold_metrics"] = fold_metrics
    return overall_metrics


def print_classification_report(metrics: Dict[str, Any], title: str = "Classification Report") -> str:
    acc = metrics["accuracy"] * 100
    sens = metrics["sensitivity"] * 100
    spec = metrics["specificity"] * 100
    prec = metrics["precision"] * 100
    f1 = metrics["f1_score"] * 100
    kappa = metrics["cohen_kappa"]
    auc = metrics["roc_auc"] * 100

    acc_std = metrics.get("accuracy_std", 0.0) * 100
    f1_std = metrics.get("f1_score_std", 0.0) * 100

    return f"""
### {title}
| Metric | Value |
| :--- | :--- |
| **Accuracy** | **{acc:.2f}%** +/- {acc_std:.2f}% |
| **Sensitivity (Recall)** | {sens:.2f}% |
| **Specificity** | {spec:.2f}% |
| **Precision** | {prec:.2f}% |
| **F1-Score** | **{f1:.2f}%** +/- {f1_std:.2f}% |
| **Cohen's Kappa (k)** | {kappa:.4f} |
| **ROC-AUC** | {auc:.2f}% |
"""
