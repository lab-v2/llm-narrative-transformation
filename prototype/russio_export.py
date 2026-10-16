"""
Export the russio documents (data/russio-relevance-data/russio_all.jsonl) as one text file per
document, so the survey, embedding and prediction scripts can read them like the stories.

Each file is the title, a blank line, then the cleaned text: data/russio-prototype/docs/doc_<id>.txt
(data/ is gitignored). data/russio-prototype/manifest.csv records each document's metadata and the
existing narrative/framing/identity scores (empty where not scored).

Usage (from repo root):
    python3 prototype/russio_export.py
"""

import argparse
import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
META = ["doc_id", "outlet", "tier", "lang", "pub_date", "url", "word_count",
        "narrative_score", "framing_score", "identity_score", "any_relevant"]


def main():
    parser = argparse.ArgumentParser(description="Export russio documents as text files + manifest")
    parser.add_argument("--input", type=str, default=str(REPO / "data" / "russio-relevance-data" / "russio_all.jsonl"))
    parser.add_argument("--dest", type=str, default=str(REPO / "data" / "russio-prototype"))
    args = parser.parse_args()

    dest = Path(args.dest)
    docs = dest / "docs"
    if docs.exists() and any(docs.iterdir()):
        raise SystemExit(f"{docs} is not empty - refusing to overwrite")
    docs.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            name = f"doc_{r['doc_id']:05d}"
            text = f"{(r.get('title') or '').strip()}\n\n{r['cleaned'].strip()}\n"
            (docs / f"{name}.txt").write_text(text, encoding="utf-8")
            rows.append({"name": name, "words_exported": len(text.split()), **{k: r.get(k) for k in META}})

    with open(dest / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} documents -> {docs} | manifest {dest / 'manifest.csv'}")


if __name__ == "__main__":
    main()
