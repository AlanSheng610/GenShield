"""Datasets over pre-extracted UNet bottleneck latents.

Each ``.pt`` file holds the mid-block output of one generation (or DDIM inversion)
at every denoising step, shaped [T, 1280, 8, 8]. The file is a dict
``{"latents": ..., "prompt": ..., "seed": ...}``. Files written by the original
research code are also accepted: a bare tensor, or a dict keyed by
``bottleneck_features``.
"""

import glob
import os
import re
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import Dataset

SAFE_LABEL = 0


@dataclass
class Source:
    """A directory of latent files sharing one label (0 = safe, k = k-th concept)."""

    path: str
    label: int
    limit: Optional[int] = None


def _file_index(path):
    numbers = re.findall(r"\d+", os.path.basename(path))
    return int(numbers[-1]) if numbers else -1


def list_latent_files(directory, limit=None):
    files = sorted(glob.glob(os.path.join(directory, "*.pt")), key=_file_index)
    if not files:
        raise FileNotFoundError(f"No latent files (*.pt) found in {directory}")
    return files[:limit] if limit else files


def load_latent_file(path):
    """Returns (latents [T, C, H*W] float tensor, prompt str)."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    prompt = ""
    if isinstance(data, dict):
        prompt = data.get("prompt", "")
        latents = data["latents"] if "latents" in data else data["bottleneck_features"]
    else:
        latents = data
    if latents.dim() == 5:  # [T, 1, C, H, W]
        latents = latents.squeeze(1)
    return latents.reshape(latents.shape[0], latents.shape[1], -1).float(), prompt


class LatentSequenceDataset(Dataset):
    """One item per generation: (latents [T, C, N], label, prompt). Used for evaluation."""

    def __init__(self, sources, steps=None):
        self.items = [(f, s.label) for s in sources for f in list_latent_files(s.path, s.limit)]
        self.steps = list(steps) if steps is not None else None

    def __len__(self):
        return len(self.items)

    def _load(self, idx):
        latents, prompt = load_latent_file(self.items[idx][0])
        if self.steps is not None:
            latents = latents[self.steps]
        return latents, prompt

    def __getitem__(self, idx):
        latents, prompt = self._load(idx)
        return latents, self.items[idx][1], prompt


class LatentStepDataset(LatentSequenceDataset):
    """One item per (generation, denoising step): (latents [C, N], step, label). Used for training."""

    def __init__(self, sources, steps=range(50)):
        super().__init__(sources, steps=None)
        self.step_ids = list(steps)

    def __len__(self):
        return len(self.items) * len(self.step_ids)

    def __getitem__(self, idx):
        file_idx, step_pos = divmod(idx, len(self.step_ids))
        step = self.step_ids[step_pos]
        latents, _ = load_latent_file(self.items[file_idx][0])
        return latents[step], step, self.items[file_idx][1]

    def label_counts(self):
        counts = {}
        for _, label in self.items:
            counts[label] = counts.get(label, 0) + 1
        return counts
