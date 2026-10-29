"""Score a directory of generated images with a post-hoc safety checker.

Writes ``{image filename: unsafe score}`` as JSON, which baselines/evaluate.py reads.

    python baselines/score_images.py --method nudenet --image_dir data/test/mma_adv_sexual/image \\
        --out results/baselines/nudenet/mma_adv_sexual.json

Methods:
  sd_safety_checker  Stable Diffusion's built-in checker. It only outputs a binary flag,
                     so threshold-free metrics are degenerate (FPR@TPR95 = 100%).
  nudenet            Highest confidence over all NudeNet detections (pip install nudenet).
  clip_nsfw          LAION CLIP-based NSFW detector (pip install tensorflow autokeras
                     git+https://github.com/openai/CLIP.git).

Q16 and LatentGuard are scored with their official repositories; baselines/evaluate.py
reads their output formats directly.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


class SDSafetyChecker:
    def __init__(self, device):
        from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
        from transformers import CLIPImageProcessor

        repo = "CompVis/stable-diffusion-safety-checker"
        self.device = device
        self.checker = StableDiffusionSafetyChecker.from_pretrained(repo).to(device).eval()
        self.processor = CLIPImageProcessor.from_pretrained(repo)

    @torch.no_grad()
    def __call__(self, path):
        image = Image.open(path).convert("RGB")
        clip_input = self.processor(image, return_tensors="pt").pixel_values.to(self.device)
        _, has_nsfw = self.checker(clip_input=clip_input, images=np.array(image)[None].copy())
        return float(has_nsfw[0])


class NudeNet:
    def __init__(self, device):
        from nudenet import NudeDetector

        self.detector = NudeDetector()

    def __call__(self, path):
        return float(max((d["score"] for d in self.detector.detect(path)), default=0.0))


class CLIPNSFW:
    URLS = {
        "ViT-L/14": ("clip_autokeras_binary_nsfw", 768),
        "ViT-B/32": ("clip_autokeras_nsfw_b32", 512),
    }

    def __init__(self, device, clip_model="ViT-L/14"):
        import zipfile
        from urllib.request import urlretrieve

        import autokeras as ak
        import clip
        from tensorflow.keras.models import load_model

        name, _ = self.URLS[clip_model]
        cache = os.path.expanduser(f"~/.cache/clip_nsfw/{clip_model.replace('/', '_')}")
        model_dir = os.path.join(cache, name)
        if not os.path.exists(model_dir):
            os.makedirs(cache, exist_ok=True)
            zip_path = os.path.join(cache, f"{name}.zip")
            urlretrieve(f"https://raw.githubusercontent.com/LAION-AI/CLIP-based-NSFW-Detector/main/{name}.zip", zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(cache)
        self.head = load_model(model_dir, custom_objects=ak.CUSTOM_OBJECTS)
        self.device = device
        self.clip, self.preprocess = clip.load(clip_model, device=device)

    @torch.no_grad()
    def __call__(self, path):
        x = self.preprocess(Image.open(path)).unsqueeze(0).to(self.device)
        emb = self.clip.encode_image(x).float()
        emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()
        return float(self.head.predict(emb, verbose=0)[0][0])


METHODS = {"sd_safety_checker": SDSafetyChecker, "nudenet": NudeNet, "clip_nsfw": CLIPNSFW}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    scorer = METHODS[args.method](args.device)
    paths = sorted(p for p in glob.glob(os.path.join(args.image_dir, "*")) if p.lower().endswith(IMAGE_EXTS))
    scores, elapsed = {}, []
    for path in tqdm(paths, desc=args.method):
        start = time.perf_counter()
        scores[os.path.basename(path)] = scorer(path)
        elapsed.append(time.perf_counter() - start)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Scored {len(scores)} images ({1000 * np.mean(elapsed):.1f} ms/image) -> {args.out}")


if __name__ == "__main__":
    main()
