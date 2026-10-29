import numpy as np
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_curve


def fpr_at_tpr(labels, scores, target_tpr=0.95):
    """False-positive rate at the first operating point whose TPR reaches ``target_tpr``."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(fpr[np.searchsorted(tpr, target_tpr, side="left")])


def best_f1(labels, scores, num_thresholds=200):
    candidates = np.linspace(scores.min(), scores.max(), num_thresholds)
    f1s = [f1_score(labels, scores >= t, zero_division=0) for t in candidates]
    i = int(np.argmax(f1s))
    return float(f1s[i]), float(candidates[i])


def detection_metrics(labels, scores, threshold=None):
    """Threshold-free metrics (AUROC, AUPRC, FPR@TPR95), plus F1/accuracy/recall at
    ``threshold`` when given. All values are fractions in [0, 1]."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        raise ValueError("Detection metrics need both safe and unsafe samples")

    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    f1_best, t_best = best_f1(labels, scores)
    out = {
        "n_safe": int((labels == 0).sum()),
        "n_unsafe": int((labels == 1).sum()),
        "auroc": float(auc(fpr, tpr)),
        "auprc": float(auc(recall, precision)),
        "fpr@tpr95": fpr_at_tpr(labels, scores, 0.95),
        "best_f1": f1_best,
        "best_f1_threshold": t_best,
    }
    if threshold is not None:
        pred = scores >= threshold
        out.update({
            "threshold": float(threshold),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "accuracy": float((pred == labels).mean()),
            "recall": float(pred[labels == 1].mean()),
        })
    return out


def accuracy_over_steps(step_scores, labels, threshold):
    """Accuracy if generation were stopped after each step.

    A sample is flagged at step t if its score reached ``threshold`` at any step <= t.

    Args:
        step_scores: [N, T] per-step scores.
        labels: [N] binary labels.

    Returns:
        [T] accuracy after each step.
    """
    flagged = np.maximum.accumulate(np.asarray(step_scores), axis=1) >= threshold
    return (flagged == np.asarray(labels)[:, None].astype(bool)).mean(axis=0)


def balanced_subsample(safe_scores, unsafe_scores, seed=42):
    """Randomly subsample the larger set so both classes have the same size."""
    rng = np.random.RandomState(seed)
    safe_scores, unsafe_scores = np.asarray(safe_scores), np.asarray(unsafe_scores)
    n = min(len(safe_scores), len(unsafe_scores))
    if len(safe_scores) > n:
        safe_scores = rng.choice(safe_scores, n, replace=False)
    if len(unsafe_scores) > n:
        unsafe_scores = rng.choice(unsafe_scores, n, replace=False)
    return safe_scores, unsafe_scores


def format_table(rows, keys=("auroc", "auprc", "fpr@tpr95")):
    """rows: {name: metrics dict} -> fixed-width text table with values in percent."""
    width = max([len("Dataset")] + [len(name) for name in rows]) + 2
    lines = [f"{'Dataset':<{width}}" + "".join(f"{k.upper():>12}" for k in keys)]
    lines.append("-" * len(lines[0]))
    for name, m in rows.items():
        lines.append(f"{name:<{width}}" + "".join(f"{100 * m[k]:>12.2f}" for k in keys))
    return "\n".join(lines)
