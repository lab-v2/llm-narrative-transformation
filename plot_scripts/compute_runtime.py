"""
compute_runtime.py

Computes per-story transformation runtime for iteration_0 (fallback to iteration_1, 2, ...)
based on timestamps in cost_tracking.json and cost_tracking_transformation.json.

Runtime = transform_timestamp (last entry) - survey_timestamp (first entry in cost_tracking.json)

Folder structure:
  single   → output/phase2/{model_name}/{problem}/{story}/
  multiple → output/phase2/{model_name}-original/{problem}/{story}/

Usage (run from project root):
  python plot_scripts/compute_runtime.py --mode single   --problem forward --model-name gpt-4o
  python plot_scripts/compute_runtime.py --mode multiple --problem inverse --model-name gpt-4o
"""

import argparse
import json
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ==========================================================
# Helpers
# ==========================================================

def parse_timestamp(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str)


def load_json(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_cost_files(story_dir: Path, max_iterations: int = 10):
    """
    Search iteration_0, iteration_1, ... until both cost files are found.
    Returns (cost_tracking dict, cost_tracking_transformation dict, iteration index)
    or (None, None, None) if not found in any iteration.
    """
    for i in range(max_iterations):
        iter_dir = story_dir / f"iteration_{i}"
        ct_path  = iter_dir / "cost_tracking.json"
        ctt_path = iter_dir / "cost_tracking_transformation.json"
        if ct_path.exists() and ctt_path.exists():
            return load_json(ct_path), load_json(ctt_path), i
    return None, None, None


def compute_runtime(ct: dict, ctt: dict, mode: str):
    """
    Returns (survey_timestamp_str, transform_timestamp_str, runtime_seconds).
      single   → transformation end = per_story_breakdown[0]['timestamp']
      multiple → transformation end = per_story_breakdown[-1]['timestamp']
    """
    survey_ts_str    = ct['per_story_breakdown'][0]['timestamp']
    survey_ts        = parse_timestamp(survey_ts_str)

    if mode == 'single':
        transform_ts_str = ctt['per_story_breakdown'][0]['timestamp']
    else:
        transform_ts_str = ctt['per_story_breakdown'][-1]['timestamp']

    transform_ts    = parse_timestamp(transform_ts_str)
    runtime_seconds = (transform_ts - survey_ts).total_seconds()

    return survey_ts_str, transform_ts_str, runtime_seconds


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser(
        description='Compute per-story runtime from cost tracking files.'
    )
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], required=True,
                        help='single: single-call transformer | multiple: iterative transformer')
    parser.add_argument('--problem', type=str, choices=['forward', 'inverse'], required=True,
                        help='Problem type: forward or inverse')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Model name (e.g. gpt-4o, gpt-4o-mini)')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Root output directory (default: output)')
    parser.add_argument('--results-dir', type=str, default='output_analysis/runtime_csvs',
                        help='Directory to save CSVs (default: output_analysis/runtime_csvs)')
    parser.add_argument('--max-iterations', type=int, default=10,
                        help='Max iterations to search per story (default: 10)')

    args = parser.parse_args()

    # Resolve model folder:
    # single   → output/phase2/{model_name}/
    # multiple → output/phase2/{model_name}-original/
    phase2_dir  = Path(args.output_dir) / "phase2"
    folder_name = f"{args.model_name}-original" if args.mode == "multiple" else args.model_name
    model_dir   = phase2_dir / folder_name

    if not phase2_dir.exists():
        print(f"Error: phase2 directory not found: {phase2_dir}")
        return

    if not model_dir.exists():
        print(f"Error: model directory not found: {model_dir}")
        return

    problem_dir = model_dir / args.problem
    if not problem_dir.exists():
        print(f"Error: problem directory not found: {problem_dir}")
        return

    story_dirs = sorted([d for d in problem_dir.iterdir() if d.is_dir()])
    print(f"\nModel:   {args.model_name}")
    print(f"Problem: {args.problem} | Mode: {args.mode}")
    print(f"Stories found: {len(story_dirs)}")
    print("=" * 50)

    rows = []

    for story_dir in story_dirs:
        story_name = story_dir.name

        ct, ctt, iter_used = find_cost_files(story_dir, args.max_iterations)

        if ct is None:
            print(f"  SKIPPING {story_name}: cost files not found in any iteration")
            continue

        try:
            survey_ts, transform_ts, runtime_sec = compute_runtime(ct, ctt, args.mode)
        except (KeyError, IndexError, ValueError) as e:
            print(f"  SKIPPING {story_name}: error parsing timestamps — {e}")
            continue

        runtime_min     = round(runtime_sec / 60, 2)
        model_from_json = ct.get('model', args.model_name)  # authoritative model name

        rows.append({
            'story_name':          story_name,
            'model':               model_from_json,
            'problem_type':        args.problem,
            'mode':                args.mode,
            'iteration_used':      iter_used,
            'survey_timestamp':    survey_ts,
            'transform_timestamp': transform_ts,
            'runtime_seconds':     round(runtime_sec, 2),
            'runtime_minutes':     runtime_min,
        })

        print(f"  ✓ {story_name:<35} iter={iter_used}  {runtime_sec:.1f}s ({runtime_min}m)")

    if not rows:
        print("\nNo data collected. Check your output directory structure.")
        return

    # -------------------------------------------------------
    # Save per-story CSV
    # -------------------------------------------------------
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv = results_dir / f"runtime_{args.model_name}_{args.mode}_{args.problem}.csv"
    fieldnames = [
        'story_name', 'model', 'problem_type', 'mode', 'iteration_used',
        'survey_timestamp', 'transform_timestamp', 'runtime_seconds', 'runtime_minutes'
    ]

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Per-story CSV  → {out_csv}")

    # -------------------------------------------------------
    # Save summary CSV
    # -------------------------------------------------------
    runtimes = [r['runtime_seconds'] for r in rows]
    mean_sec = sum(runtimes) / len(runtimes)
    std_sec  = (sum((x - mean_sec) ** 2 for x in runtimes) / len(runtimes)) ** 0.5

    summary_csv = results_dir / f"runtime_{args.model_name}_{args.mode}_{args.problem}_summary.csv"
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model', 'problem_type', 'mode', 'n_stories',
            'mean_runtime_seconds', 'std_runtime_seconds',
            'mean_runtime_minutes', 'std_runtime_minutes',
            'min_runtime_seconds',  'max_runtime_seconds',
        ])
        writer.writeheader()

        groups = defaultdict(list)
        for r in rows:
            groups[r['model']].append(r['runtime_seconds'])

        for model, times in sorted(groups.items()):
            n    = len(times)
            mean = sum(times) / n
            std  = (sum((x - mean) ** 2 for x in times) / n) ** 0.5
            writer.writerow({
                'model':                model,
                'problem_type':         args.problem,
                'mode':                 args.mode,
                'n_stories':            n,
                'mean_runtime_seconds': round(mean, 2),
                'std_runtime_seconds':  round(std, 2),
                'mean_runtime_minutes': round(mean / 60, 2),
                'std_runtime_minutes':  round(std  / 60, 2),
                'min_runtime_seconds':  round(min(times), 2),
                'max_runtime_seconds':  round(max(times), 2),
            })

    print(f"✓ Summary CSV    → {summary_csv}")
    print(f"\n{'='*50}")
    print(f"SUMMARY: {len(rows)} stories | mean={mean_sec:.1f}s ({mean_sec/60:.2f}m) | std={std_sec:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
