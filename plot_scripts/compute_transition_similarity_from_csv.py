"""
Compute cosine similarity from existing transition CSVs.

Reads:
  original_transitions.csv
  transformed_transitions.csv

Writes:
  transition_similarity.txt
"""

import argparse
from pathlib import Path

import csv


def load_matrix(path: Path):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "entity":
            raise ValueError(f"Unexpected header in {path}")
        cols = header[1:]
        rows = {}
        def to_float(val: str) -> float:
            try:
                return float(val)
            except Exception:
                return 0.0
        for row in reader:
            if not row:
                continue
            entity = row[0]
            values = [to_float(x) if x else 0.0 for x in row[1:]]
            rows[entity] = dict(zip(cols, values))
    return rows, cols


def align_and_vectorize(a_rows, a_cols, b_rows, b_cols):
    all_cols = sorted(set(a_cols) | set(b_cols))
    all_rows = sorted(set(a_rows.keys()) | set(b_rows.keys()))
    vec_a = []
    vec_b = []
    for entity in all_rows:
        a_vals = a_rows.get(entity, {})
        b_vals = b_rows.get(entity, {})
        for col in all_cols:
            vec_a.append(a_vals.get(col, 0.0))
            vec_b.append(b_vals.get(col, 0.0))
    return vec_a, vec_b


def cosine_similarity(vec_a, vec_b):
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    denom = (norm_a ** 0.5) * (norm_b ** 0.5)
    if denom == 0:
        return 0.0
    return dot / denom


def main():
    parser = argparse.ArgumentParser(description="Compute transition similarity from CSVs.")
    parser.add_argument("--original", required=True, help="Path to original_transitions.csv")
    parser.add_argument("--transformed", required=True, help="Path to transformed_transitions.csv")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to transition_similarity.txt to write",
    )
    args = parser.parse_args()

    orig_rows, orig_cols = load_matrix(Path(args.original))
    trans_rows, trans_cols = load_matrix(Path(args.transformed))

    vec_orig, vec_trans = align_and_vectorize(orig_rows, orig_cols, trans_rows, trans_cols)
    sim = cosine_similarity(vec_orig, vec_trans)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"cosine_similarity={sim:.6f}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
