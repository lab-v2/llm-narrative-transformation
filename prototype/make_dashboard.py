"""
Build the interactive results dashboard (one self-contained HTML file) from the
summary.json that evaluate.py writes.

Shows: headline numbers, agreement with the LLM survey (QWK with bootstrap ranges and
CV QWK), paired differences, all accuracy scores, time and cost per story, per-question
QWK heatmap, confusion matrices per question and model, results by story type, and a
story explorer. The page follows the viewer's light/dark theme.

Usage (from repo root):
    python3 prototype/make_dashboard.py \
        --summary output_test/prototype/evaluation/test/summary.json \
        --output output_test/prototype/evaluation/test/dashboard.html
"""

import argparse
import json
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"


def main():
    parser = argparse.ArgumentParser(description="Build the results dashboard from evaluate.py's summary.json")
    parser.add_argument("--summary", type=str, required=True, help="summary.json written by evaluate.py")
    parser.add_argument("--output", type=str, required=True, help="Where to write the dashboard .html")
    args = parser.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        data = json.load(f)
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")  # safe inside <script>
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload, 1)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out} ({out.stat().st_size / 1024:.0f} KB, {len(data['models'])} models, "
          f"{data['n_stories']} stories)")


if __name__ == "__main__":
    main()
