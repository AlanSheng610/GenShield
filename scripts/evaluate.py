"""Evaluate a trained detector on the test suites listed in a config.

For every generation the detector score is the maximum concept similarity over all
denoising steps; accuracy-vs-step curves show what early termination would achieve.

    python scripts/evaluate.py --config configs/genshield.yaml --checkpoint checkpoints/genshield/last.ckpt
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from genshield import plotting
from genshield.data import SAFE_LABEL, LatentSequenceDataset, Source
from genshield.detector import GenShield
from genshield.metrics import accuracy_over_steps, detection_metrics, format_table
from genshield.utils import get_device, load_config, set_seed


@torch.no_grad()
def score_suite(shield, concept, dataset, batch_size, num_workers):
    """Returns per-step similarity to ``concept`` [N, T], labels [N], prompts."""
    k = shield.concepts.index(concept)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    scores, labels, prompts = [], [], []
    for latents, batch_labels, batch_prompts in tqdm(loader, leave=False):
        b, t = latents.shape[:2]
        steps = torch.arange(t).repeat(b)
        sim = shield.score(latents.flatten(0, 1), steps)[:, k]  # [B*T]
        scores.append(sim.view(b, t).cpu())
        labels.append(batch_labels)
        prompts.extend(batch_prompts)
    return torch.cat(scores).numpy(), torch.cat(labels).numpy(), prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=None, help="Overrides eval.threshold")
    parser.add_argument("--suites", nargs="*", default=None, help="Only run these suite names")
    parser.add_argument("--batch_size", type=int, default=16, help="Generations per batch")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg["eval"]
    threshold = args.threshold if args.threshold is not None else ecfg["threshold"]
    out_dir = ecfg["output_dir"]
    test_root = os.path.join(cfg["data_root"], "test")
    device = get_device(args.device)
    set_seed(cfg["train"]["seed"])
    os.makedirs(out_dir, exist_ok=True)

    shield = GenShield.from_files(args.checkpoint, cfg["concepts_path"], cfg["timesteps_path"],
                                  concepts=cfg["concepts"], nhead=cfg["model"]["nhead"],
                                  threshold=threshold, device=device)
    rows, groups = {}, {}
    for suite in ecfg["suites"]:
        name, concept = suite["name"], suite["concept"]
        if args.suites and name not in args.suites:
            continue
        limit = suite.get("limit")
        dataset = LatentSequenceDataset([
            Source(os.path.join(test_root, suite.get("safe", ecfg.get("safe"))), SAFE_LABEL, limit),
            Source(os.path.join(test_root, name), 1, limit),
        ])
        step_scores, labels, prompts = score_suite(shield, concept, dataset, args.batch_size, args.num_workers)
        scores = step_scores.max(axis=1)
        metrics = detection_metrics(labels, scores, threshold)
        metrics["accuracy_over_steps"] = accuracy_over_steps(step_scores, labels, threshold).tolist()
        rows[name] = metrics
        if "group" in suite:
            g = groups.setdefault(suite["group"], ([], []))
            g[0].append(labels)
            g[1].append(scores)

        print(f"{name}: AUROC {100 * metrics['auroc']:.2f}  AUPRC {100 * metrics['auprc']:.2f}  "
              f"FPR@TPR95 {100 * metrics['fpr@tpr95']:.2f}  F1@{threshold} {100 * metrics['f1']:.2f}")
        title = f"{name} ({concept})"
        plotting.plot_roc_pr(labels, scores, metrics, title, os.path.join(out_dir, f"{name}_roc_pr.png"))
        plotting.plot_score_distribution(labels, scores, threshold, title, os.path.join(out_dir, f"{name}_scores.png"))
        plotting.plot_accuracy_over_steps(metrics["accuracy_over_steps"], threshold, title,
                                          os.path.join(out_dir, f"{name}_accuracy_vs_step.png"))
        np.savez(os.path.join(out_dir, f"{name}_scores.npz"), step_scores=step_scores, labels=labels,
                 prompts=np.array(prompts, dtype=object))

    for group, (labels, scores) in groups.items():
        rows[f"{group} (pooled)"] = detection_metrics(np.concatenate(labels), np.concatenate(scores), threshold)

    print("\n" + format_table(rows, keys=("auroc", "auprc", "fpr@tpr95", "f1")))
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "threshold": threshold, "results": rows}, f, indent=2)
    print(f"\nPlots and metrics written to {out_dir}/")


if __name__ == "__main__":
    main()
