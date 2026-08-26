"""Evaluation metrics engine.

SRD Traceability: FR-25, NFR-11
"""

import numpy as np
from typing import Dict, Any, Union
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

def compute_evaluation_metrics(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a single task row (FR-25).
    
    `y_true`: Ground truth binary labels (0 or 1).
    `y_score`: Raw uncalibrated `model_score` predictions (FR-15, FR-16).
    `threshold`: Decision threshold for binarizing predictions.
    
    Safely handles edge cases such as zero positive predictions or single-class targets (NFR-11).
    """
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    
    n_samples = len(y_true)
    n_positives = int(np.sum(y_true == 1))
    prevalence = float(n_positives / n_samples) if n_samples > 0 else 0.0
    
    y_pred = (y_score >= threshold).astype(int)
    
    # Precision, recall, f1 per class
    # zero_division=0 prevents warnings/exceptions on undefined precision/recall
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    
    macro_f1 = float(np.mean(f1))
    
    # Confusion matrix [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    # ROC-AUC (meaningful only if both classes are present)
    roc_auc: Union[float, None] = None
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            roc_auc = None
            
    # PR-AUC (Area under Precision-Recall Curve)
    pr_auc: Union[float, None] = None
    if len(np.unique(y_true)) > 1:
        try:
            precisions, recalls, _ = precision_recall_curve(y_true, y_score)
            pr_auc = float(auc(recalls, precisions))
        except ValueError:
            pr_auc = None
            
    return {
        "n_samples": n_samples,
        "n_positives": n_positives,
        "prevalence": prevalence,
        "class_0_precision": float(p[0]),
        "class_0_recall": float(r[0]),
        "class_0_f1": float(f1[0]),
        "class_1_precision": float(p[1]),
        "class_1_recall": float(r[1]),
        "class_1_f1": float(f1[1]),
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }
