"""Generation-time defense: run SD-1.5 with GenShield and stop as soon as an unsafe
concept is detected (paper Sec. 3.4).

    python scripts/generate.py --checkpoint checkpoints/genshield/last.ckpt \\
        --prompt "a cozy cabin in a snowy forest" --out cabin.png
"""

import argparse

from genshield.detector import GenShield
from genshield.pipeline import shielded_generate
from genshield.utils import get_device, load_sd_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--concepts_path", default="data/embeddings/concepts.pt")
    parser.add_argument("--timesteps_path", default="data/embeddings/timesteps.pt")
    parser.add_argument("--concepts", nargs="+", default=None, help="Subset of concepts to guard against")
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.87)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--model_id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--out", default="output.png")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    pipe = load_sd_pipeline(args.model_id, device)
    shield = GenShield.from_files(args.checkpoint, args.concepts_path, args.timesteps_path,
                                  concepts=args.concepts, nhead=args.nhead,
                                  threshold=args.threshold, device=device)

    image, report = shielded_generate(pipe, shield, args.prompt, args.seed, args.guidance_scale)
    scores = ", ".join(f"{c}={s:.3f}" for c, s in report["max_scores"].items())
    if image is None:
        print(f"Blocked at step {report['step'] + 1}/50: '{report['concept']}' (score {report['score']:.3f})")
    else:
        image.save(args.out)
        print(f"Safe. Saved {args.out}")
    print(f"Max similarity per concept: {scores}")


if __name__ == "__main__":
    main()
