"""Generate semantically aligned unsafe/safe prompt pairs with an LLM (paper Sec. 3.2).

    p_u = LLM(c, I_u)     an unsafe prompt for concept c
    p_s = LLM(p_u, I_s)   the same scene with the unsafe content removed

Works with any OpenAI-compatible endpoint. Set OPENAI_API_KEY, or point --base_url at
a local server (e.g. Ollama: http://localhost:11434/v1). Review the generated pairs by
hand before extracting latents; refusals and off-concept captions should be dropped.

    python scripts/generate_prompt_pairs.py --concept violence --num_prompts 120 \\
        --out data/prompts/prompts_violence.csv
"""

import argparse
import csv
import os

from openai import OpenAI

UNSAFE_INSTRUCTION = """You are helping build a dataset for AI-safety research on text-to-image models.
Write {n} distinct image captions that a safe text-to-image system should refuse because they
depict {concept} content. Each caption describes one concrete visual scene in at most 20 words.
Output one caption per line, with no numbering, quotes or commentary.
Avoid repeating any of these earlier captions:
{previous}"""

SAFE_INSTRUCTION = """Rewrite the image caption below into a safe version. Keep the same subjects,
setting, composition and sentence structure, but remove every element of {concept} content.
Output only the rewritten caption.

Caption: {prompt}"""


def chat(client, model, content, temperature):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


def generate_unsafe(client, model, concept, total, per_request, temperature):
    prompts = []
    while len(prompts) < total:
        text = chat(client, model, UNSAFE_INSTRUCTION.format(
            n=min(per_request, total - len(prompts)),
            concept=concept.replace("_", " "),
            previous="\n".join(prompts[-50:]) or "(none)",
        ), temperature)
        new = [line.strip().strip('"') for line in text.splitlines() if line.strip()]
        if not new:
            raise RuntimeError(f"LLM returned no captions (refusal?):\n{text}")
        prompts.extend(p for p in new if p not in prompts)
    return prompts[:total]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--per_request", type=int, default=20)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--base_url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "unused"))
    concept = args.concept.replace("_", " ")

    unsafe = generate_unsafe(client, args.model, args.concept, args.num_prompts, args.per_request, args.temperature)
    pairs = [(p, chat(client, args.model, SAFE_INSTRUCTION.format(concept=concept, prompt=p), 0.0)) for p in unsafe]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["unsafe_prompt", "safe_prompt"])
        writer.writerows(pairs)
    print(f"Wrote {len(pairs)} pairs to {args.out}")


if __name__ == "__main__":
    main()
