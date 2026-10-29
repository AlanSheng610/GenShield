"""Train the C3A detector with safety-aware contrastive learning.

    python scripts/train.py --config configs/genshield.yaml
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from genshield.data import SAFE_LABEL, LatentStepDataset, Source
from genshield.embeddings import stack_concepts
from genshield.losses import SupConLoss
from genshield.model import C3A
from genshield.utils import get_device, load_config, set_seed


def expand(dirs, concepts):
    """Expand "{concept}" placeholders: ["gen_{concept}_safe"] -> one entry per concept."""
    out = []
    for d in dirs:
        out.extend([(d.format(concept=c), c) for c in concepts] if "{concept}" in d else [(d, None)])
    return out


def build_sources(cfg):
    root = os.path.join(cfg["data_root"], "train")
    concepts = cfg["concepts"]
    sources = [Source(os.path.join(root, d), SAFE_LABEL) for d, _ in expand(cfg["train"]["safe"], concepts)]
    for d, concept in expand(cfg["train"]["unsafe"], concepts):
        if concept is None:
            raise ValueError(f"Unsafe directory '{d}' must contain '{{concept}}' so it can be labelled")
        sources.append(Source(os.path.join(root, d), concepts.index(concept) + 1))
    return sources


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]
    set_seed(tcfg["seed"])
    device = get_device(args.device)
    os.makedirs(tcfg["output_dir"], exist_ok=True)

    concepts = stack_concepts(torch.load(cfg["concepts_path"], weights_only=True), cfg["concepts"]).to(device)
    timestep_table = torch.load(cfg["timesteps_path"], weights_only=True).to(device)

    dataset = LatentStepDataset(build_sources(cfg), steps=range(*tcfg["steps"]))
    print(f"{len(dataset.items)} generations x {len(dataset.step_ids)} steps; label counts: {dataset.label_counts()}")
    loader = DataLoader(dataset, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=tcfg["num_workers"], pin_memory=device.type == "cuda")

    model = C3A(nhead=cfg["model"]["nhead"], dropout=cfg["model"]["dropout"]).to(device)
    criterion = SupConLoss(temperature=tcfg["temperature"])
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    print(f"C3A parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    with open(os.path.join(tcfg["output_dir"], "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        total = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{tcfg['epochs']}")
        for latents, steps, labels in pbar:
            latents, steps, labels = latents.to(device), steps.to(device), labels.to(device)
            features, feature_labels = model.contrastive_features(latents, timestep_table[steps], labels, concepts)
            loss = criterion(features, feature_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch}: mean loss {total / len(loader):.4f}")
        torch.save(model.state_dict(), os.path.join(tcfg["output_dir"], f"epoch_{epoch}.ckpt"))

    torch.save(model.state_dict(), os.path.join(tcfg["output_dir"], "last.ckpt"))
    print(f"Saved {os.path.join(tcfg['output_dir'], 'last.ckpt')}")


if __name__ == "__main__":
    main()
