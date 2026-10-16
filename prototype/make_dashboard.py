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
    parser.add_argument("--hide", type=str, nargs="*", default=[],
                        choices=["kappa", "agreement", "scores", "speed", "questions", "confusion", "groups", "stories"],
                        help="Parts to leave out, e.g. a short version for sharing: "
                             "--hide kappa agreement scores questions groups stories")
    args = parser.parse_args()

    with open(args.summary, encoding="utf-8") as f:
        data = json.load(f)
    data["hide"] = args.hide
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")  # safe inside <script>
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", payload, 1)
    if data.get("n_classes") == 2:  # yes/no survey: its own name, so the two dashboards are easy to tell apart
        html = html.replace("LLM-Free Survey Scorecard", "LLM-Free Yes/No Scorecard")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out} ({out.stat().st_size / 1024:.0f} KB, {len(data['models'])} models, "
          f"{data['n_stories']} stories)")


if __name__ == "__main__":
    main()
