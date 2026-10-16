"""
Combine three yes/no LLM survey runs into majority-vote answers (steadier labels).

For each story and question: the majority of the three 0/1 answers. Each output JSON has the
same shape as survey_story.py's (so every script can read it) plus:
  votes       {question_id: [run1, run2, run3]}
  yes_share   {question_id: fraction of runs that said yes (0, 1/3, 2/3, 1)}
  timing / tokens / cost_usd   summed over the three calls (a majority-of-3 survey costs 3 calls)
Also prints how consistent the LLM is across runs.

Usage (from repo root):
    python3 prototype/make_majority_labels.py --split train
    python3 prototype/make_majority_labels.py --split test
"""

import argparse
import itertools
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
P = REPO / "output_test" / "prototype"
RUNS = [P / "survey_binary" / "gpt-6-sol", P / "survey_binary_run2" / "gpt-6-sol", P / "survey_binary_run3" / "gpt-6-sol"]


def main():
    parser = argparse.ArgumentParser(description="Majority vote of three yes/no survey runs")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: output_test/prototype/survey_binary_majority/gpt-6-sol/<split>")
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else P / "survey_binary_majority" / "gpt-6-sol" / args.split
    out.mkdir(parents=True, exist_ok=True)

    runs = [{p.stem: json.load(open(p, encoding="utf-8")) for p in (r / args.split).glob("*.json")} for r in RUNS]
    stories = sorted(set(runs[0]))
    missing = [(i + 1, s) for i, r in enumerate(runs) for s in stories if s not in r or r[s].get("n_failed", 1) != 0]
    if missing:
        raise SystemExit(f"Incomplete runs (run, story): {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    pair_agree, unanimous, changed, total = {pair: 0 for pair in itertools.combinations(range(3), 2)}, 0, 0, 0
    for s in stories:
        r = [run[s] for run in runs]
        qids = sorted(r[0]["ratings"], key=int)
        votes = {q: [x["ratings"][q] for x in r] for q in qids}
        majority = {q: int(sum(v) >= 2) for q, v in votes.items()}
        for q, v in votes.items():
            total += 1
            unanimous += len(set(v)) == 1
            changed += majority[q] != v[0]
            for a, b in pair_agree:
                pair_agree[(a, b)] += v[a] == v[b]
        result = {k: r[0][k] for k in ("story_name", "story_file", "model", "problem_type", "questions_file", "answer_scale")}
        result.update({
            "label_rule": "majority of 3 yes/no runs",
            "runs": [str(run_dir / args.split / f"{s}.json") for run_dir in RUNS],
            "ratings": majority,
            "votes": votes,
            "yes_share": {q: round(sum(v) / 3, 4) for q, v in votes.items()},
            "n_failed": 0,
            "tokens": {k: sum((x["tokens"].get(k) or 0) for x in r) for k in ("input", "output", "reasoning", "total")},
            "timing": {k: round(sum(x["timing"][k] for x in r), 3) for k in ("llm_latency_s", "story_wall_s")},
            "cost_usd": round(sum(x["cost_usd"] or 0 for x in r), 6),
        })
        with open(out / f"{s}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"{args.split}: {len(stories)} stories x {len(qids)} questions = {total} answers -> {out}")
    for (a, b), n in pair_agree.items():
        print(f"  run {a + 1} vs run {b + 1}: same answer {n / total:.1%}")
    print(f"  all three runs agree: {unanimous / total:.1%} | majority differs from run 1: {changed / total:.1%}")


if __name__ == "__main__":
    main()
