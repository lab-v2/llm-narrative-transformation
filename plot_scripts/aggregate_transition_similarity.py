"""
Aggregate transition similarity averages from existing transition_similarity.txt files.

Scans: output_analysis/entity_transitions/evaluation_data/<model>/<group>/<story>/{abduction,baseline}/transition_similarity.txt
Outputs a TSV table to stdout.
"""

from pathlib import Path


BASE = Path(
    "/home/dbavikad/leibniz/llm-narrative-transformation/output_analysis/entity_transitions/evaluation_data"
)


def parse_similarity(path: Path):
    try:
        text = path.read_text(encoding="utf-8").strip()
        return float(text.split("=")[-1])
    except Exception:
        return None


def main():
    rows = []
    for model_dir in sorted([p for p in BASE.iterdir() if p.is_dir()]):
        model = model_dir.name
        for group in ["collectivistic", "individualistic"]:
            group_dir = model_dir / group
            if not group_dir.exists():
                continue

            sims_abd = []
            sims_base = []

            for story_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()]):
                abd_sim = story_dir / "abduction" / "transition_similarity.txt"
                base_sim = story_dir / "baseline" / "transition_similarity.txt"

                if abd_sim.exists():
                    val = parse_similarity(abd_sim)
                    if val is not None:
                        sims_abd.append(val)
                if base_sim.exists():
                    val = parse_similarity(base_sim)
                    if val is not None:
                        sims_base.append(val)

            if sims_abd:
                rows.append(
                    (model, group, "abduction", sum(sims_abd) / len(sims_abd), len(sims_abd))
                )
            if sims_base:
                rows.append(
                    (model, group, "baseline", sum(sims_base) / len(sims_base), len(sims_base))
                )

    print("Model\tGroup\tKind\tAvgCosine\tN")
    for model, group, kind, avg, n in sorted(rows):
        print(f"{model}\t{group}\t{kind}\t{avg:.6f}\t{n}")


if __name__ == "__main__":
    main()
