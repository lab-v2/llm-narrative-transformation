"""
Build the train/test split for the LLM-free survey prototype.

Per source (individualistic and collectivistic stories, split independently):
- Drop the n_drop longest stories (by whitespace word count).
- Length-stratified split: sort by word count, cut into n_test near-equal
  blocks, and pick 1 story per block (seeded) for test, so train and test
  have matched length distributions.

  individualistic: 90 -> drop 10 longest (>= 3,480 words) -> 60 train / 20 test
  collectivistic:  28 -> drop 2 longest                   -> 20 train /  6 test
      (The_Strangers_That_Came_to_Town: 6,113 words and an identical copy is in
       the individualistic folder; The_Christmas_Monks: also in the
       individualistic folder under the same name)

Copies the .txt files into data/train-prototype/ and data/test-prototype/
(data/ is gitignored) and writes prototype/split.csv as the tracked record.
Re-running is safe: files already in place are kept, missing ones are copied,
and it stops if a folder holds a story that is not part of this split.

Usage (from repo root):
    python3 prototype/make_split.py
"""

import csv
import random
import shutil
import sys
from pathlib import Path

TRAIN_DIR = Path("data/train-prototype")
TEST_DIR = Path("data/test-prototype")
SPLIT_CSV = Path("prototype/split.csv")
SEED = 42

# (label, source folder, n_drop longest, n_test)
SOURCES = [
    ("individualistic", Path("data/individualistic-rags-to-riches-stories"), 10, 20),
    ("collectivistic", Path("data/collectivistic-stories-all"), 2, 6),
]


def word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").split())


def length_stratified_test(kept: list, n_test: int) -> set:
    """kept is sorted by length; cut into n_test near-equal blocks, pick 1 per block."""
    rng = random.Random(SEED)
    bounds = [round(i * len(kept) / n_test) for i in range(n_test + 1)]
    return {rng.choice(kept[lo:hi])[1] for lo, hi in zip(bounds, bounds[1:])}


def main():
    rows, plan = [], []  # plan: (source path, destination folder)
    for label, src_dir, n_drop, n_test in SOURCES:
        stories = sorted((word_count(p), p) for p in src_dir.glob("*.txt"))
        kept, dropped = stories[:-n_drop], stories[-n_drop:]
        test = length_stratified_test(kept, n_test)

        for words, path in kept:
            split = "test" if path in test else "train"
            plan.append((path, TEST_DIR if split == "test" else TRAIN_DIR))
            rows.append({"story_name": path.stem, "label": label, "words": words, "split": split})
        for words, path in dropped:
            rows.append({"story_name": path.stem, "label": label, "words": words, "split": "dropped"})

    names = [p.name for p, _ in plan]
    if len(names) != len(set(names)):
        sys.exit(f"Duplicate story names across sources: {sorted({n for n in names if names.count(n) > 1})}")

    for d in (TRAIN_DIR, TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)
        expected = {p.name for p, dest in plan if dest == d}
        stray = {p.name for p in d.glob("*.txt")} - expected
        if stray:
            sys.exit(f"{d} has stories that are not in this split: {sorted(stray)}")

    n_copied = 0
    for path, dest in plan:
        if not (dest / path.name).exists():
            shutil.copy2(path, dest / path.name)
            n_copied += 1

    with open(SPLIT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["story_name", "label", "words", "split"])
        writer.writeheader()
        writer.writerows(rows)

    for label, *_ in SOURCES:
        counts = {s: sum(r["label"] == label and r["split"] == s for r in rows) for s in ("train", "test", "dropped")}
        print(f"{label:16} train={counts['train']:3d}  test={counts['test']:3d}  dropped={counts['dropped']:3d}")
    print(f"copied {n_copied} new files -> {TRAIN_DIR}, {TEST_DIR} | record -> {SPLIT_CSV}")


if __name__ == "__main__":
    main()
