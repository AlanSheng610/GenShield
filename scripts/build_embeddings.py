"""Precompute the concept text embeddings and per-step timestep embeddings.

    python scripts/build_embeddings.py --out_dir data/embeddings
"""

import argparse
import os

import torch

from genshield.embeddings import CONCEPTS, concept_embeddings, timestep_embeddings
from genshield.utils import get_device, load_sd_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--out_dir", default="data/embeddings")
    parser.add_argument("--concepts", nargs="+", default=CONCEPTS)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pipe = load_sd_pipeline(args.model_id, get_device(args.device), dtype=torch.float32)

    concepts = concept_embeddings(pipe.tokenizer, pipe.text_encoder, args.concepts)
    torch.save(concepts, os.path.join(args.out_dir, "concepts.pt"))
    print(f"concepts.pt: {list(concepts)}")

    timesteps = timestep_embeddings(pipe.unet, pipe.scheduler, args.num_inference_steps)
    torch.save(timesteps, os.path.join(args.out_dir, "timesteps.pt"))
    print(f"timesteps.pt: {tuple(timesteps.shape)}")


if __name__ == "__main__":
    main()
