"""Compute AUROC / AUPRC / FPR@TPR95 for baseline detectors from their saved scores.

For each unsafe set, the larger of (safe, unsafe) is randomly subsampled so both have
the same size. Sets whose name starts with --pool_prefix are also pooled into one row.

    python baselines/evaluate.py --format json --safe results/baselines/nudenet/normal.json \\
        --unsafe results/baselines/nudenet/{mma_adv_sexual,i2p_sexual,i2p_hate}.json

Formats:
  json         {filename: score} from baselines/score_images.py. Also reads the older
               {filename: {"unsafe_score": s}} and {"images": {...}} layouts.
  q16_csv      inappropriate_images.csv written by the official Q16 repository.
  latentguard  {"scores": [...]} written by the official LatentGuard evaluation.
"""

import argparse
import csv
import json
import os

import numpy as np

from genshield.metrics import balanced_subsample, detection_metrics, format_table


def load_scores(path, fmt):
    if fmt == "q16_csv":
        with open(path) as f:
            rows = list(csv.reader(f))[1:]
        return np.array([float(r[4]) for r in rows])
    with open(path) as f:
        data = json.load(f)
    if fmt == "latentguard":
        return np.array(data["scores"], dtype=float)
    data = data.get("images", data)
    return np.array([v["unsafe_score"] if isinstance(v, dict) else v for v in data.values()], dtype=float)


def name_of(path):
    base = os.path.basename(path.rstrip("/"))
    if base == "inappropriate_images.csv":  # Q16 layout: {dataset}/inappropriate/<model>/sim/inappropriate_images.csv
        base = path.split(os.sep)[-5]
    return os.path.splitext(base)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--format", choices=["json", "q16_csv", "latentguard"], default="json")
    parser.add_argument("--safe", required=True, help="Scores of the safe (normal) set")
    parser.add_argument("--unsafe", nargs="+", required=True, help="Scores of one or more unsafe sets")
    parser.add_argument("--pool_prefix", default="i2p_")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    safe_all = load_scores(args.safe, args.format)
    rows, pooled = {}, ([], [])
    for path in args.unsafe:
        safe, unsafe = balanced_subsample(safe_all, load_scores(path, args.format), seed=args.seed)
        labels = np.r_[np.zeros(len(safe)), np.ones(len(unsafe))]
        scores = np.r_[safe, unsafe]
        name = name_of(path)
        rows[name] = detection_metrics(labels, scores)
        if name.startswith(args.pool_prefix):
            pooled[0].append(labels)
            pooled[1].append(scores)

    if pooled[0]:
        rows[f"{args.pool_prefix.rstrip('_')} (pooled)"] = detection_metrics(np.concatenate(pooled[0]),
                                                                            np.concatenate(pooled[1]))
    print(format_table(rows))


if __name__ == "__main__":
    main()
