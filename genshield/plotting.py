import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(labels, scores, metrics, title, path):
    fpr, tpr, _ = roc_curve(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.5))
    ax_roc.plot(fpr, tpr, lw=2, label=f"AUROC = {100 * metrics['auroc']:.2f}%")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    ax_pr.plot(recall, precision, lw=2, label=f"AUPRC = {100 * metrics['auprc']:.2f}%")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall")
    for ax in (ax_roc, ax_pr):
        ax.set(xlim=(0, 1), ylim=(0, 1.05))
        ax.legend(loc="lower right")
    fig.suptitle(title)
    _save(fig, path)


def plot_score_distribution(labels, scores, threshold, title, path):
    labels, scores = np.asarray(labels), np.asarray(scores)
    bins = np.linspace(scores.min(), scores.max(), 30)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores[labels == 0], bins=bins, alpha=0.6, label="Safe")
    ax.hist(scores[labels == 1], bins=bins, alpha=0.6, label="Unsafe")
    ax.axvline(threshold, color="k", ls="--", label=f"Threshold ({threshold:.2f})")
    ax.set(xlabel="Max concept similarity over steps", ylabel="Count", title=title)
    ax.legend()
    _save(fig, path)


def plot_accuracy_over_steps(accuracies, threshold, title, path):
    steps = np.arange(1, len(accuracies) + 1)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(steps, 100 * np.asarray(accuracies), marker="o", ms=3)
    ax.set(xlabel="Denoising step", ylabel="Accuracy (%)", ylim=(0, 102),
           title=f"{title} (threshold={threshold:.2f})")
    ax.grid(alpha=0.3)
    _save(fig, path)
