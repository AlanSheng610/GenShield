"""Obtain bottleneck latents of existing images through DDIM inversion (paper Sec. 3.4).

The image is inverted with an empty prompt over --inversion_steps fine-grained steps.
The recorded latents are put back into denoising order and subsampled to the same
number of steps the detector was trained on.

From a directory of images::

    python scripts/invert_images.py --image_dir my_images/ --out_dir data/test/my_images

From UnsafeBench (gated on Hugging Face; run `huggingface-cli login` first)::

    python scripts/invert_images.py --unsafebench --split test --category sexual \\
        --safety_label Unsafe --out_dir data/test/unsafebench_unsafe_sexual
"""

import argparse
import glob
import os

import torch
from diffusers import DDIMInverseScheduler
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from genshield.hooks import LatentRecorder
from genshield.utils import get_device, load_sd_pipeline

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def iter_image_dir(image_dir):
    for path in sorted(glob.glob(os.path.join(image_dir, "*"))):
        if path.lower().endswith(IMAGE_EXTS):
            yield Image.open(path)


def iter_unsafebench(split, category, safety_label):
    from datasets import load_dataset

    for row in load_dataset("yiting/UnsafeBench", split=split):
        if row["safety_label"] == safety_label and row["category"].lower() == category.replace("_", " "):
            yield row["image"]


@torch.no_grad()
def invert(pipe, image, inversion_steps, keep_steps, size):
    x = transforms.ToTensor()(image.convert("RGB").resize((size, size), Image.LANCZOS))[None]
    x = x.to(pipe.device, pipe.unet.dtype) * 2 - 1
    latents = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor

    with LatentRecorder(pipe.unet, classifier_free_guidance=False) as rec:
        pipe(prompt="", guidance_scale=1.0, num_inference_steps=inversion_steps,
             latents=latents, output_type="latent")
    recorded = rec.stack()[:, 0].flip(0)  # inversion runs t: 0 -> T; detector expects T -> 0
    keep = torch.linspace(0, len(recorded) - 1, keep_steps).long()
    return recorded[keep].half().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image_dir")
    src.add_argument("--unsafebench", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--category", help="UnsafeBench category, e.g. sexual, self_harm")
    parser.add_argument("--safety_label", choices=["Safe", "Unsafe"], default="Unsafe")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--inversion_steps", type=int, default=999)
    parser.add_argument("--keep_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    pipe = load_sd_pipeline(args.model_id, get_device(args.device), dtype=torch.float32,
                            scheduler_cls=DDIMInverseScheduler)
    images = (iter_unsafebench(args.split, args.category, args.safety_label) if args.unsafebench
              else iter_image_dir(args.image_dir))

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_images:
        os.makedirs(os.path.join(args.out_dir, "image"), exist_ok=True)
    for idx, image in enumerate(tqdm(images, desc=args.out_dir)):
        latents = invert(pipe, image, args.inversion_steps, args.keep_steps, args.resolution)
        torch.save({"latents": latents, "prompt": ""}, os.path.join(args.out_dir, f"latents_{idx}.pt"))
        if args.save_images:
            image.convert("RGB").resize((args.resolution, args.resolution)).save(os.path.join(args.out_dir, "image", f"{idx}.png"))


if __name__ == "__main__":
    main()
