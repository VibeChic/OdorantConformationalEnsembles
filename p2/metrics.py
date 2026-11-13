"""Shared metrics for Paper 2 (macro-AUCPR primary, micro/macro AUROC secondary)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def macro_metrics(y, p):
    """y, p: [M, L] numpy arrays. Labels with no positive are skipped (counted)."""
    aps, aucs = [], []
    for j in range(y.shape[1]):
        pos = y[:, j].sum()
        if pos > 0:
            aps.append(average_precision_score(y[:, j], p[:, j]))
            if pos < y.shape[0]:
                aucs.append(roc_auc_score(y[:, j], p[:, j]))
    out = {
        "macro_aucpr": float(np.mean(aps)) if aps else float("nan"),
        "macro_auroc": float(np.mean(aucs)) if aucs else float("nan"),
        "n_eval_labels": len(aps),
    }
    try:
        out["micro_auroc"] = float(roc_auc_score(y.ravel(), p.ravel()))
    except ValueError:
        out["micro_auroc"] = float("nan")
    return out
