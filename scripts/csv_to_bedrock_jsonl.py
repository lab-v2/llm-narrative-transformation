#!/usr/bin/env python3
"""
Convert a CSV file with prompt/completion columns to JSONL for Bedrock.

Each output line will be:
{"prompt": "...", "completion": "..."}
"""

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CSV to Bedrock JSONL.")
    parser.add_argument("--csv", required=True, help="Path to input CSV")
    parser.add_argument("--jsonl", required=True, help="Path to output JSONL")
    parser.add_argument("--prompt-col", default="user_prompt", help="CSV column for prompt")
    parser.add_argument("--completion-col", default="assistant_output", help="CSV column for completion")
    parser.add_argument(
        "--format",
        choices=["prompt_completion", "conversation"],
        default="prompt_completion",
        help="Output schema: prompt_completion or conversation",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    jsonl_path = Path(args.jsonl)

    with csv_path.open("r", encoding="utf-8", newline="") as f_in, jsonl_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)
        if args.prompt_col not in reader.fieldnames or args.completion_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must include columns '{args.prompt_col}' and '{args.completion_col}'. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            prompt = row.get(args.prompt_col, "")
            completion = row.get(args.completion_col, "")
            if args.format == "conversation":
                record = {
                    "schemaVersion": "bedrock-conversation-2024",
                    "messages": [
                        {"role": "user", "content": [{"text": prompt}]},
                        {"role": "assistant", "content": [{"text": completion}]},
                    ],
                }
            else:
                record = {"prompt": prompt, "completion": completion}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
