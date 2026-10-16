"""
Pick extra training stories for the LLM-free survey prototype from the project's existing
LLM-transformed stories (output/phase2 and output/baseline_*), for the TRAIN split only.

- Only transforms of train stories (transforms of test stories are near-copies of test
  stories and would leak into the test set).
- Up to --per-story distinct versions per train story, spread over the different generators
  (pipeline/baseline x model), exact duplicates removed, at most --max-words words.
- output/ is only read. The picked texts are copied to data/augment-prototype/ (gitignored)
  as <source_story>__aug<k>.txt; prototype/augment_manifest.csv records where each came from.

Usage (from repo root):
    python3 prototype/make_augment_set.py --per-story 5
"""

import argparse
import csv
import hashlib
import random
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "output"
SOURCES = [("phase2", "story_transformed.txt"), ("baseline_zeroshot", "transformed_story.txt"),
           ("baseline_cot", "transformed_story.txt"), ("baseline_fewshot", "transformed_story.txt"),
           ("baseline_iterative", "transformed_story.txt")]


def main():
    parser = argparse.ArgumentParser(description="Pick transformed train stories as extra training data")
    parser.add_argument("--per-story", type=int, default=5)
    parser.add_argument("--max-words", type=int, default=5000)
    parser.add_argument("--dest", type=str, default=str(REPO / "data" / "augment-prototype"))
    parser.add_argument("--manifest", type=str, default=str(REPO / "prototype" / "augment_manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(REPO / "prototype" / "split.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    train = {r["story_name"]: r["label"] for r in rows if r["split"] == "train"}

    # candidates per train story, grouped by generator (source folder + model folder)
    cands = defaultdict(lambda: defaultdict(list))
    seen = set()
    for folder, name in SOURCES:
        for p in sorted((OUT / folder).rglob(name)):
            story = next((part for part in p.parts if part in train), None)
            if story is None:
                continue
            text = p.read_text(encoding="utf-8", errors="replace")
            words = len(text.split())
            digest = hashlib.md5(text.encode()).hexdigest()
            if words < 50 or words > args.max_words or digest in seen:
                continue
            seen.add(digest)
            generator = f"{folder}/{p.relative_to(OUT / folder).parts[0]}"
            cands[story][generator].append((p, words))

    rng = random.Random(args.seed)
    dest = Path(args.dest)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit(f"{dest} is not empty - refusing to overwrite")
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for story in sorted(train):
        gens = list(cands[story].values())
        for g in gens:
            rng.shuffle(g)
        rng.shuffle(gens)
        picked = []
        while len(picked) < args.per_story and any(gens):       # round-robin over generators for variety
            for g in gens:
                if g and len(picked) < args.per_story:
                    picked.append(g.pop())
        for i, (p, words) in enumerate(picked, 1):
            name = f"{story}__aug{i}"
            shutil.copy2(p, dest / f"{name}.txt")
            manifest.append({"story_name": name, "source_story": story, "source_label": train[story],
                             "origin": str(p.relative_to(REPO)), "words": words})

    with open(args.manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    per = defaultdict(int)
    for m in manifest:
        per[m["source_story"]] += 1
    print(f"{len(manifest)} extra stories from {len(per)} train stories "
          f"(per story: min {min(per.values())}, max {max(per.values())}) -> {dest} | manifest {args.manifest}")


if __name__ == "__main__":
    main()
