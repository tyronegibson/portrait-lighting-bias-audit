#!/usr/bin/env python3
"""Re-analyze an image set with both the v1 metric and the v2 face-exposure metrics.

Images must be named <Group>_<variant>_<n>.png (e.g. Black_woman_regular_0.png).
An annotations JSON supplies each image's face box and its medium (photo/painting):

    python3 analyze.py IMAGE_DIR data/v1_test_set_annotations.json --out results/
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import cv2

from exposure_metrics import face_exposure, whole_image_brightness

METRICS = ["v1_brightness", "face_L", "shadow_clip", "highlight_clip", "face_detail", "face_vs_scene"]


def parse_name(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    return "_".join(parts[:-2]), parts[-2]   # group, variant


def bootstrap_diff_ci(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for mean(b) - mean(a). Wide intervals mean 'not enough samples'."""
    rng = random.Random(seed)
    diffs = sorted(
        st.mean(rng.choices(b, k=len(b))) - st.mean(rng.choices(a, k=len(a))) for _ in range(n)
    )
    return diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def analyze(image_dir: Path, annotations: dict) -> list[dict]:
    rows = []
    for path in sorted(image_dir.glob("*.png")):
        ann = annotations[path.stem]
        img = cv2.imread(str(path))
        group, variant = parse_name(path.stem)
        fe = face_exposure(img, tuple(ann["face_box"]))
        rows.append({"image": path.stem, "group": group, "variant": variant, "medium": ann["medium"],
                     "box_source": ann.get("box_source", "manual"),
                     "v1_brightness": whole_image_brightness(img), **fe.as_dict()})
    return rows


def summarize(rows: list[dict]) -> str:
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r["group"], r["variant"])].append(r)
    groups = sorted({r["group"] for r in rows})

    out.append("## Regular vs enhanced prompt, per group (mean of n images)\n")
    out.append("| Group | n | v1 brightness | 95% CI of change | Medium (regular → enhanced) | Face shadow clip | Face highlight clip | Face detail (L* p90-p10) | Face vs scene (L*) |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        reg, enh = by[(g, "regular")], by[(g, "enhanced")]
        def m(rs, k): return st.mean(r[k] for r in rs)
        lo, hi = bootstrap_diff_ci([r["v1_brightness"] for r in reg], [r["v1_brightness"] for r in enh])
        med = lambda rs: f'{sum(r["medium"] == "photo" for r in rs)}/{len(rs)} photo'
        out.append(
            f"| {g.replace('_', ' ')} | {len(reg)}+{len(enh)} "
            f"| {m(reg,'v1_brightness'):.1f} → {m(enh,'v1_brightness'):.1f} "
            f"({(m(enh,'v1_brightness') - m(reg,'v1_brightness')) / m(reg,'v1_brightness') * 100:+.1f}%) "
            f"| {lo:+.1f} to {hi:+.1f} "
            f"| {med(reg)} → {med(enh)} "
            f"| {m(reg,'shadow_clip')*100:.1f}% → {m(enh,'shadow_clip')*100:.1f}% "
            f"| {m(reg,'highlight_clip')*100:.1f}% → {m(enh,'highlight_clip')*100:.1f}% "
            f"| {m(reg,'face_detail'):.1f} → {m(enh,'face_detail'):.1f} "
            f"| {m(reg,'face_vs_scene'):+.1f} → {m(enh,'face_vs_scene'):+.1f} |")

    out.append("\n## By medium, all groups pooled\n")
    out.append("| Medium | n | v1 brightness | Face shadow clip | Face detail | Face vs scene |")
    out.append("|---|---|---|---|---|---|")
    for medium in ("painting", "photo"):
        rs = [r for r in rows if r["medium"] == medium]
        if rs:
            out.append(f"| {medium} | {len(rs)} | {st.mean(r['v1_brightness'] for r in rs):.1f} "
                       f"| {st.mean(r['shadow_clip'] for r in rs)*100:.1f}% "
                       f"| {st.mean(r['face_detail'] for r in rs):.1f} "
                       f"| {st.mean(r['face_vs_scene'] for r in rs):+.1f} |")

    v1_flag = [r for r in rows if r["v1_brightness"] < 100]
    out.append(f"\n## The v1 'poorly lit' rule (brightness < 100)\n")
    out.append(f"Flags {len(v1_flag)} of {len(rows)} images: "
               + ", ".join(f'{r["image"]} ({r["medium"]}, face shadow clip {r["shadow_clip"]*100:.1f}%)' for r in v1_flag))
    haar_fail = [r["image"] for r in rows if r["box_source"] == "manual"]
    out.append(f"\n## Face detector check\n\nOpenCV Haar cascades misplaced the face in {len(haar_fail)} of {len(rows)} images "
               f"(boxes corrected by hand): {', '.join(haar_fail)}.")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("image_dir", type=Path)
    ap.add_argument("annotations", type=Path)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()

    rows = analyze(args.image_dir, json.loads(args.annotations.read_text()))
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    report = summarize(rows)
    (args.out / "summary.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
