import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # Always use a non-interactive backend for CLI runs.
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "diagnosis_scores.png"
DEFAULT_MEDIAN_OUTPUT = SCRIPT_DIR / "diagnosis_scores_median.png"
DEFAULT_MODE_OUTPUT = SCRIPT_DIR / "diagnosis_scores_mode.png"
SCORE_PATTERN = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)")
EXPERT_RATING_PATTERN = re.compile(
    r"rating[\s:_]*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE
)


def normalize_story_name(raw_name: Optional[str], fallback: str) -> str:
    """Return a cleaned story name used for alignment and display."""
    name = (raw_name or fallback).strip()
    name = re.sub(r"\s+", " ", name)
    return name


def extract_score(raw_response: str, question_id: int, story_file: Path) -> float:
    """Pull the numeric score from the start of the GPT response block."""
    match = SCORE_PATTERN.match(raw_response or "")
    if match:
        return float(match.group(1))

    rating_match = EXPERT_RATING_PATTERN.search(raw_response or "")
    if rating_match:
        return float(rating_match.group(1))

    raise ValueError(
        f"Could not find a numeric score for question {question_id} "
        f"in {story_file.name}"
    )


def read_story_scores(story_path: Path) -> Tuple[str, List[float]]:
    """Return the story name and the list of raw scores from all questions."""
    with story_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    fallback_name = story_path.stem.replace("_", " ")
    story_name = normalize_story_name(payload.get("story_name"), fallback_name)
    responses = payload.get("questions_and_answers") or []
    if not responses:
        raise ValueError(f"No survey entries found in {story_path}")

    scores: List[float] = []
    for entry in responses:
        try:
            scores.append(
                extract_score(
                    entry.get("gpt_response", ""),
                    entry.get("id", -1),
                    story_path,
                )
            )
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

    return story_name, scores


def gather_story_scores(survey_dir: Path) -> List[Tuple[str, List[float]]]:
    """Load every survey JSON file and return the raw scores per story."""
    survey_files = sorted(survey_dir.glob("*.json"))
    if not survey_files:
        raise FileNotFoundError(f"No JSON files found in {survey_dir}")

    story_scores: List[Tuple[str, List[float]]] = []
    for story_path in survey_files:
        story_scores.append(read_story_scores(story_path))
    return story_scores


def resolve_experiment_dir(raw_path: Path) -> Path:
    """Resolve the experiment directory regardless of the caller's working dir."""
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    candidates.append((Path.cwd() / raw_path).resolve())
    candidates.append((REPO_ROOT / raw_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not locate experiment directory at {raw_path}. "
        "Tried absolute path, relative to the current directory, and relative to the repo root."
    )


def gather_iter_story_scores(
    experiment_dir: Path,
) -> dict[str, List[Tuple[str, List[float]]]]:
    """Collect story scores for every iter_* survey inside experiment_dir."""
    iter_dirs = sorted(
        d for d in experiment_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")
    )
    if not iter_dirs:
        raise FileNotFoundError(
            f"No iter_* directories found inside {experiment_dir}"
        )

    iter_story_scores: dict[str, List[Tuple[str, List[float]]]] = {}
    for iter_dir in iter_dirs:
        survey_dir = iter_dir / "survey"
        if not survey_dir.exists():
            continue
        iter_story_scores[iter_dir.name] = gather_story_scores(survey_dir)

    if not iter_story_scores:
        raise FileNotFoundError(
            f"No survey directories found under {experiment_dir}"
        )

    return iter_story_scores


def compute_mode(values: Sequence[float]) -> float:
    """Return the mode with deterministic tie-breaking (smallest value wins)."""
    counts = Counter(values)
    max_freq = max(counts.values())
    candidates = [val for val, freq in counts.items() if freq == max_freq]
    return min(candidates)


def summarize_scores(
    story_scores: Iterable[Tuple[str, Sequence[float]]],
    reducer: Callable[[Sequence[float]], float],
) -> List[Tuple[str, float]]:
    """Apply reducer (mean/median/etc.) to each story's score list."""
    return [(story, reducer(scores)) for story, scores in story_scores]


def aggregate_best_story_scores(
    iter_story_scores: dict[str, List[Tuple[str, List[float]]]],
    reducer: Callable[[Sequence[float]], float],
) -> List[Tuple[str, float]]:
    """Compute the best reduced score per story across all iterations."""
    best_scores: dict[str, Tuple[float, str]] = {}
    for iter_name, story_scores in iter_story_scores.items():
        summaries = summarize_scores(story_scores, reducer)
        for story, value in summaries:
            record = best_scores.get(story)
            if record is None or value > record[0]:
                best_scores[story] = (value, iter_name)

    return [
        (story, best_scores[story][0])
        for story in sorted(best_scores.keys())
    ]


def draw_bar_plot(
    scores: List[Tuple[str, float]],
    output_path: Path,
    show: bool,
    y_label: str,
    label: str,
    secondary_scores: Optional[List[Tuple[str, float]]] = None,
    secondary_label: str = "Expert",
) -> None:
    """Draw and optionally display grouped bar plots for abduction vs expert."""
    if not scores:
        raise ValueError("No scores provided for plotting.")

    primary_map = {story: value for story, value in scores}
    secondary_map: dict[str, float] = {}
    if secondary_scores:
        for story, value in secondary_scores:
            secondary_map[story] = value

    ordered_names = list(
        dict.fromkeys(
            list(primary_map.keys()) + list(secondary_map.keys())
        )
    )

    primary_values = [primary_map.get(name) for name in ordered_names]
    secondary_values = (
        [secondary_map.get(name) for name in ordered_names]
        if secondary_scores
        else None
    )

    x_positions = list(range(len(ordered_names)))
    width = 0.35 if secondary_scores else 0.6

    def _heights(values: List[Optional[float]]) -> List[float]:
        return [val if val is not None else 0.0 for val in values]

    fig, ax = plt.subplots(figsize=(12, 6))

    primary_bars = ax.bar(
        [x - (width / 2 if secondary_scores else 0) for x in x_positions],
        _heights(primary_values),
        width,
        label=label,
        color="#4e79a7",
    )

    secondary_bars = None
    if secondary_values:
        secondary_bars = ax.bar(
            [x + width / 2 for x in x_positions],
            _heights(secondary_values),
            width,
            label=secondary_label,
            color="#e15759",
        )

    ax.set_ylabel(y_label)
    ax.set_xlabel("Story")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_names, rotation=35, ha="right")
    max_value = max(
        [val for val in primary_values if val is not None]
        + (
            [val for val in secondary_values if val is not None]
            if secondary_values
            else []
        )
    )
    ax.set_ylim(0, max_value + 1)
    ax.legend(loc="upper left")

    def annotate(bars, values):
        if bars is None:
            return
        for rect, val in zip(bars, values):
            if val is None:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.05,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    annotate(primary_bars, primary_values)
    if secondary_values:
        annotate(secondary_bars, secondary_values)

    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a bar chart where the x-axis lists the story names and the y-axis "
            "shows the average GPT score pulled from each survey JSON file."
        )
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help=(
            "Path to the experiment directory (e.g., experiment_results/exp_it5_seg100_feat20_temp0.0). "
            "The script will look for iter_*/survey folders inside this path."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Where to save the generated figure (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--median-output",
        type=Path,
        default=DEFAULT_MEDIAN_OUTPUT,
        help=(
            "Where to save the median score figure "
            f"(default: {DEFAULT_MEDIAN_OUTPUT})"
        ),
    )
    parser.add_argument(
        "--mode-output",
        type=Path,
        default=DEFAULT_MODE_OUTPUT,
        help=(
            "Where to save the mode score figure "
            f"(default: {DEFAULT_MODE_OUTPUT})"
        ),
    )
    # parser.add_argument(
    #     "--expert-dir",
    #     type=Path,
    #     default=None,
    #     help=(
    #         "Directory containing expert survey JSON files. "
    #         "Set to 'none' to skip expert plotting."
    #     ),
    # )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = resolve_experiment_dir(args.experiment_dir)
    iter_story_scores = gather_iter_story_scores(experiment_dir)

    expert_avg_scores: Optional[List[Tuple[str, float]]] = None
    # if args.expert_dir:
    #     expert_dir = resolve_experiment_dir(args.expert_dir)
    #     expert_story_scores = gather_story_scores(expert_dir)
    #     expert_avg_scores = summarize_scores(expert_story_scores, mean)

    average_scores = aggregate_best_story_scores(iter_story_scores, mean)
    draw_bar_plot(
        average_scores,
        args.output,
        args.show,
        "Avg Diagnosis Score",
        "Abduction",
        # secondary_scores=expert_avg_scores,
        # secondary_label="Expert",
    )
    print(
        f"Wrote best average-score bar chart for {len(average_scores)} stories "
        f"to {args.output.resolve()}"
    )

    median_scores = aggregate_best_story_scores(iter_story_scores, median)
    draw_bar_plot(
        median_scores,
        args.median_output,
        args.show,
        "Median Diagnosis Score",
        "Abduction",
    )
    print(
        f"Wrote best median-score bar chart for {len(median_scores)} stories "
        f"to {args.median_output.resolve()}"
    )

    mode_scores = aggregate_best_story_scores(iter_story_scores, compute_mode)
    draw_bar_plot(
        mode_scores,
        args.mode_output,
        args.show,
        "Mode Diagnosis Score",
        "Abduction",
    )
    print(
        f"Wrote best mode-score bar chart for {len(mode_scores)} stories "
        f"to {args.mode_output.resolve()}"
    )


if __name__ == "__main__":
    main()
