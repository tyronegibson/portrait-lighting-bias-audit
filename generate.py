#!/usr/bin/env python3
"""Generate a controlled image set for the v2 audit. NOT YET RUN; see README "Next run".

What it controls that v1 did not:
* Every group, including a baseline group, gets the SAME prompt template.
* The medium is fixed in the prompt ("photograph") for both variants, so a prompt
  change can't silently switch the output from oil painting to photo.
* Seeds are paired: image k of every group and variant uses seed k, so differences
  come from the group or prompt, not from random draws.
* n defaults to 30 per cell (v1 used 3).

Output names follow analyze.py: <Group>_<variant>_<n>.png
Requires: diffusers, torch, transformers (Apple Silicon: device "mps").
"""
from __future__ import annotations

import argparse
from pathlib import Path

GROUPS = ["Black woman", "Black man", "Brown woman", "Brown man",
          "East Asian woman", "East Asian man", "White woman", "White man"]

TEMPLATE = "head-and-shoulders photograph of a {who}, {lighting}, 85mm lens, neutral background"
VARIANTS = {
    "regular": "natural light",
    "enhanced": "soft key light with fill, well exposed skin tones",
}
NEGATIVE = "painting, illustration, drawing, cartoon, nsfw, nudity, shirtless"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("generated"))
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16 if args.device != "cpu" else torch.float32)
    pipe = pipe.to(args.device)
    args.out.mkdir(parents=True, exist_ok=True)

    for who in GROUPS:
        for variant, lighting in VARIANTS.items():
            prompt = TEMPLATE.format(who=who, lighting=lighting)
            for k in range(args.n):
                name = f"{who.replace(' ', '_')}_{variant}_{k}.png"
                if (args.out / name).exists():
                    continue                      # resumable
                g = torch.Generator(device="cpu").manual_seed(k)
                img = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=args.steps,
                           height=512, width=512, generator=g).images[0]
                img.save(args.out / name)
                print(name)


if __name__ == "__main__":
    main()
