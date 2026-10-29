"""Run SD-1.5 on a list of prompts and save the bottleneck latents of every step.

Each generation is written to ``{out_dir}/latents_{i}.pt`` as
``{"latents": [T, 1280, 8, 8] fp16, "prompt": str, "seed": int}``.

Training pairs (Sec. 3.2): both halves of every pair share the same seeds, so the
safe and unsafe latents start from identical noise::

    python scripts/extract_latents.py --pairs data/prompts/prompts_sexual.csv \\
        --out_dir data/train/gen_sexual_{kind} --num_seeds 5

Benchmarks (I2P, P4D, MMA-Diffusion, ...), optionally taking per-prompt seeds from a column::

    python scripts/extract_latents.py --prompts i2p_benchmark.csv --column prompt \\
        --filter_column categories --filter_value "self-harm" --seed_column sd_seed \\
        --out_dir data/test/i2p_self_harm --save_images
"""

import argparse
import csv
import os

import torch
from tqdm import tqdm

from genshield.hooks import LatentRecorder
from genshield.utils import get_device, load_sd_pipeline


def read_prompts(path, column=None, seed_column=None, guidance_column=None,
                 filter_column=None, filter_value=None):
    """Returns a list of (prompt, seed or None, guidance or None)."""
    if path.endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            return [(line.strip(), None, None) for line in f if line.strip()]
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if filter_column:
        rows = [r for r in rows if filter_value in r[filter_column]]
    column = column or next(iter(rows[0]))
    return [(r[column],
             int(float(r[seed_column])) if seed_column else None,
             float(r[guidance_column]) if guidance_column else None) for r in rows]


def generate(pipe, prompt, seed, guidance_scale, num_steps):
    with LatentRecorder(pipe.unet, classifier_free_guidance=guidance_scale > 1) as rec:
        image = pipe(prompt, generator=torch.Generator("cpu").manual_seed(seed),
                     guidance_scale=guidance_scale, num_inference_steps=num_steps).images[0]
    return rec.stack()[:, 0].half().cpu(), image


def run(pipe, jobs, out_dir, args):
    """jobs: list of (file_index, prompt, seed, guidance)."""
    os.makedirs(out_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(out_dir, "image"), exist_ok=True)
    for idx, prompt, seed, guidance in tqdm(jobs, desc=out_dir):
        path = os.path.join(out_dir, f"latents_{idx}.pt")
        if os.path.exists(path) and not args.overwrite:
            continue
        latents, image = generate(pipe, prompt, seed, guidance, args.num_inference_steps)
        torch.save({"latents": latents, "prompt": prompt, "seed": seed}, path)
        if args.save_images:
            image.save(os.path.join(out_dir, "image", f"{idx}.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs", help="CSV with unsafe_prompt,safe_prompt columns")
    src.add_argument("--prompts", help="CSV or TXT (one prompt per line)")
    parser.add_argument("--out_dir", required=True, help="With --pairs, must contain '{kind}' (safe/unsafe)")
    parser.add_argument("--column", default=None, help="Prompt column (default: first)")
    parser.add_argument("--seed_column", default=None)
    parser.add_argument("--guidance_column", default=None)
    parser.add_argument("--filter_column", default=None, help="Keep rows whose column contains --filter_value")
    parser.add_argument("--filter_value", default=None)
    parser.add_argument("--num_seeds", type=int, default=1, help="Generations per prompt (seeds 0..n-1)")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pipe = load_sd_pipeline(args.model_id, get_device(args.device))

    def jobs_for(prompts):
        jobs = []
        for p_idx, (prompt, seed, guidance) in enumerate(prompts):
            for s in range(args.num_seeds):
                jobs.append((p_idx * args.num_seeds + s, prompt,
                             seed if seed is not None else s,
                             guidance if guidance is not None else args.guidance_scale))
        return jobs

    if args.pairs:
        if "{kind}" not in args.out_dir:
            parser.error("--out_dir must contain '{kind}' when using --pairs")
        with open(args.pairs, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for kind in ("unsafe", "safe"):
            prompts = [(r[f"{kind}_prompt"], None, None) for r in rows]
            run(pipe, jobs_for(prompts), args.out_dir.format(kind=kind), args)
    else:
        prompts = read_prompts(args.prompts, args.column, args.seed_column, args.guidance_column,
                               args.filter_column, args.filter_value)
        run(pipe, jobs_for(prompts), args.out_dir, args)


if __name__ == "__main__":
    main()
