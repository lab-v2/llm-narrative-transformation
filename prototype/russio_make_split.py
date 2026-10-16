"""
Train/test split of the russio documents for LLM-to-classifier distillation, stratified by
language AND the LLM's answer, so every language keeps its yes/no mix in both parts.

Needs every document's survey JSON (survey_story.py on data/russio-prototype/docs). Writes:
  data/russio-prototype/split.csv     story_name, label (= language, used to stratify CV folds),
                                      answer, split, doc_id, outlet, tier, words
  data/russio-prototype/{train,test}/ the documents' text files
  <survey-dir>/../{train,test}/       the matching survey JSONs (so load_data can pair them)
data/ is gitignored.

Usage (from repo root):
    python3 prototype/russio_make_split.py --survey-dir output_test/prototype/russio/survey_war/gpt-5.4/all
"""

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "data" / "russio-prototype"


def main():
    parser = argparse.ArgumentParser(description="Stratified train/test split of the russio documents")
    parser.add_argument("--survey-dir", type=str, required=True, help="Survey JSONs for all documents")
    parser.add_argument("--question-id", type=str, default="1")
    parser.add_argument("--test-share", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    survey_dir = Path(args.survey_dir)
    with open(BASE / "manifest.csv", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    answers, missing = {}, []
    for r in manifest:
        p = survey_dir / f"{r['name']}.json"
        if not p.exists():
            missing.append(r["name"])
            continue
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        if s.get("n_failed", 1) != 0:
            missing.append(r["name"])
            continue
        answers[r["name"]] = s["ratings"][args.question_id]
    if missing:
        raise SystemExit(f"{len(missing)} documents have no complete survey yet, e.g. {missing[:5]}")

    strata = defaultdict(list)
    for r in manifest:
        strata[(r["lang"], answers[r["name"]])].append(r["name"])
    rng = random.Random(args.seed)
    test = set()
    for key in sorted(strata):
        names = sorted(strata[key])
        rng.shuffle(names)
        test.update(names[:round(len(names) * args.test_share)])

    for part in ("train", "test"):
        for d in (BASE / part, survey_dir.parent / part):
            if d.exists() and any(d.iterdir()):
                raise SystemExit(f"{d} is not empty - refusing to overwrite")
            d.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in manifest:
        part = "test" if r["name"] in test else "train"
        shutil.copy2(BASE / "docs" / f"{r['name']}.txt", BASE / part / f"{r['name']}.txt")
        shutil.copy2(survey_dir / f"{r['name']}.json", survey_dir.parent / part / f"{r['name']}.json")
        rows.append({"story_name": r["name"], "label": r["lang"], "answer": answers[r["name"]], "split": part,
                     "doc_id": r["doc_id"], "outlet": r["outlet"], "tier": r["tier"], "words": r["words_exported"]})
    with open(BASE / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    c = Counter((r["split"], r["answer"]) for r in rows)
    for part in ("train", "test"):
        n = c[(part, 0)] + c[(part, 1)]
        print(f"{part:5}: {n:5} documents | yes {c[(part, 1)]:5} ({c[(part, 1)] / n:.1%}) | no {c[(part, 0)]:5}")
    print(f"\n  {'lang':>4} | {'train yes/total':>15} | {'test yes/total':>14}")
    for lang in sorted({r['label'] for r in rows}, key=lambda l: -sum(1 for r in rows if r["label"] == l)):
        t = [r for r in rows if r["label"] == lang]
        tr = [r for r in t if r["split"] == "train"]; te = [r for r in t if r["split"] == "test"]
        print(f"  {lang:>4} | {sum(r['answer'] for r in tr):>6}/{len(tr):<5} {sum(r['answer'] for r in tr)/len(tr):>3.0%} | "
              f"{sum(r['answer'] for r in te):>5}/{len(te):<5} {sum(r['answer'] for r in te)/len(te):>3.0%}")
    print(f"\nsplit -> {BASE / 'split.csv'} | texts -> {BASE}/train, test | surveys -> {survey_dir.parent}/train, test")


if __name__ == "__main__":
    main()
