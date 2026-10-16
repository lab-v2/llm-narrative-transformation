"""
Build the 20-question russio package for the clustering work: one row per article, in the same order
everywhere, with the LLM's 20 yes/no answers, the article's OpenAI embedding and its details.

Reads
  data/russio-prototype/split.csv                         article order, language, split
  data/russio-relevance-data/russio_all.jsonl            date, outlet, tier, title
  output_test/prototype/russio/survey_war/gpt-5.4/        question 1
  output_test/prototype/russio/survey_extra/gpt-5.4/      questions 2-4
  output_test/prototype/russio/survey_q5_20/gpt-5.4/      questions 5-20
  output_test/prototype/russio/embeddings/text-embedding-3-large/whole.npy

Writes data/russio-prototype/package_20q/ (gitignored: the data stays out of the public repo)
  articles.csv          row, doc_id, split, language, outlet, tier, pub_date, words, title
  answers_llm_20q.csv   doc_id, q1 ... q20 (1 = yes, 0 = no)
  answers_llm_20q.npy   (7784, 20) uint8, same rows as articles.csv
  embeddings.npy        (7784, 3072) float32, same rows (text-embedding-3-large, whole article)
  questions_20q.json    what q1 ... q20 mean
  README.md

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_make_package.py
"""

import csv
import json
import shutil
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"
PKG = DATA / "package_20q"
SURVEYS = [("survey_war", ["1"]), ("survey_extra", ["2", "3", "4"]), ("survey_q5_20", [str(i) for i in range(5, 21)])]
Q = [str(i) for i in range(1, 21)]


def main():
    PKG.mkdir(parents=True, exist_ok=True)
    with open(DATA / "split.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    info = {}
    with open(REPO / "data" / "russio-relevance-data" / "russio_all.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                info[str(d["doc_id"])] = d

    answers = np.zeros((len(rows), len(Q)), dtype=np.uint8)
    for i, r in enumerate(rows):
        for folder, qids in SURVEYS:
            with open(OUT / folder / "gpt-5.4" / r["split"] / f"{r['story_name']}.json", encoding="utf-8") as f:
                s = json.load(f)
            if s["n_failed"] or any(s["ratings"].get(q) not in (0, 1) for q in qids):
                raise SystemExit(f"{r['story_name']}: missing answers in {folder}")
            for q in qids:
                answers[i, int(q) - 1] = s["ratings"][q]

    with open(PKG / "articles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "doc_id", "split", "language", "outlet", "tier", "pub_date", "words", "title"])
        for i, r in enumerate(rows):
            d = info[r["doc_id"]]
            w.writerow([i, r["story_name"], r["split"], r["label"], d["outlet"], d["tier"], d["pub_date"], r["words"], d["title"]])
    with open(PKG / "answers_llm_20q.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id"] + [f"q{q}" for q in Q])
        for r, a in zip(rows, answers):
            w.writerow([r["story_name"]] + a.tolist())
    np.save(PKG / "answers_llm_20q.npy", answers)

    emb_dir = OUT / "embeddings" / "text-embedding-3-large"
    with open(emb_dir / "whole_meta.csv", encoding="utf-8") as f:
        assert [m["story_name"] for m in csv.DictReader(f)] == [r["story_name"] for r in rows], "embedding order != split.csv"
    shutil.copyfile(emb_dir / "whole.npy", PKG / "embeddings.npy")
    shutil.copyfile(REPO / "prototype" / "russio_questions_20q.json", PKG / "questions_20q.json")

    questions = json.load(open(PKG / "questions_20q.json", encoding="utf-8"))["questions"]
    yes = answers.mean(0)
    qlines = "\n".join(f"| q{q['id']} | {q['component']} | {q['kind']} | {yes[q['id'] - 1]:.0%} | {q['relevance_question']['question']} |"
                       for q in questions)
    (PKG / "README.md").write_text(f"""# russio 20-question package (internal, do not publish)

{len(rows):,} Russian state-media articles in 8 languages, mostly 15 Sep - 11 Oct 2026. Every file has one row
per article in the SAME order (row 0 = first line of articles.csv).

| File | Shape | What it is |
|---|---|---|
| articles.csv | {len(rows):,} rows | doc_id, train/test split, language, outlet, tier, publication date, word count, title |
| answers_llm_20q.csv / .npy | {len(rows):,} x 20 | the LLM's yes/no answers (1 = yes, 0 = no) to the 20 questions |
| embeddings.npy | {len(rows):,} x 3,072 | OpenAI text-embedding-3-large embedding of the whole article (title + text) |
| questions_20q.json | 20 | the questions (q1 ... q20) |

Answers: gpt-5.4, temperature 0, seed 42, asked in three calls per article (q1; q2-q4; q5-q20).
Embeddings: 51 articles longer than 8,191 tokens are embedded from their first 8,191 tokens.
The train/test split is the one the student models were trained on; the student models' own
answers for the 20 questions will be added once they are trained.

## Questions

| | Short name | Kind | Yes (all articles) | Question |
|---|---|---|---|---|
{qlines}

## Loading

```python
import numpy as np, pandas as pd
articles = pd.read_csv("articles.csv")
answers = np.load("answers_llm_20q.npy")      # (N, 20) 0/1
emb = np.load("embeddings.npy")               # (N, 3072)
```
""", encoding="utf-8")
    print(f"package -> {PKG}")
    for q in questions:
        print(f"  q{q['id']:<3} {yes[q['id'] - 1]:>5.1%}  {q['component']}")


if __name__ == "__main__":
    main()
