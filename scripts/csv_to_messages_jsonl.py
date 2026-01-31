#!/usr/bin/env python3
"""
Convert a CSV dataset into a JSONL file with ChatML-style messages.

Each JSONL line looks like:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def _build_messages(
    prompt: str,
    completion: str,
    system_prompt: str | None,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": completion})
    return messages


def _iter_rows(
    csv_path: Path,
    prompt_col: str,
    completion_col: str,
    system_col: str | None,
) -> Iterable[Dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = (row.get(prompt_col) or "").strip()
            completion = (row.get(completion_col) or "").strip()
            system_prompt = (row.get(system_col) or "").strip() if system_col else ""
            yield {
                "prompt": prompt,
                "completion": completion,
                "system_prompt": system_prompt,
            }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert CSV to JSONL with ChatML-style messages."
    )
    parser.add_argument("--input-csv", required=True, help="Path to input CSV")
    parser.add_argument("--output-jsonl", required=True, help="Path to output JSONL")
    parser.add_argument("--prompt-col", default="user_prompt", help="Prompt column name")
    parser.add_argument(
        "--completion-col", default="assistant_output", help="Completion column name"
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Global system prompt to include (ignored if --system-col is set)",
    )
    parser.add_argument(
        "--system-col",
        default="",
        help="Optional system prompt column name (overrides --system-prompt per row)",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Skip rows with empty prompt or completion",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_jsonl = Path(args.output_jsonl)
    system_col = args.system_col.strip() or None
    system_prompt_default = args.system_prompt.strip()

    if not input_csv.exists():
        raise FileNotFoundError(f"CSV not found: {input_csv}")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in _iter_rows(input_csv, args.prompt_col, args.completion_col, system_col):
            prompt = row["prompt"]
            completion = row["completion"]
            if args.drop_empty and (not prompt or not completion):
                continue
            system_prompt = row["system_prompt"] or system_prompt_default
            record = {"messages": _build_messages(prompt, completion, system_prompt)}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records to {output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
