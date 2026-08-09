# CONNECT — LLM Narrative Transformation

CONNECT is a **neurosymbolic pipeline that rewrites stories to shift their cultural-values
framing** along an individualistic ↔ collectivistic axis. It combines LLM surveys with
symbolic reasoning (PyReason) and abduction to decide *which* parts of a story to change,
then uses an LLM to rewrite only those parts, iterating until the target framing is reached.

Two transformation directions are supported:

| `--problem` | Direction | Trains on | Transforms |
|-------------|-----------|-----------|------------|
| `inverse`   | individualistic → collectivistic | collectivistic corpus | an individualistic story |
| `forward`   | collectivistic → individualistic | individualistic corpus | a collectivistic story |

> ⚠️ **Doc caveat:** the `--help` / docstring wording for `forward` vs `inverse` in `main.py`
> and `baseline_transform.py` is inconsistent and in places reversed. The table above reflects
> what the **code actually does** (which `*_questions.json` file and training corpus each
> problem type selects). Trust the table, not the argparse help strings.

---

## The two routes

### Route 1 — Neurosymbolic pipeline (`main.py`) — the main entry point

Runs in two phases.

**Phase 1 (training)** — learn symbolic rules from a corpus:
1. `story_loader.py` — load the training stories for the chosen direction
2. `llm_survey.py` — survey each story on 20 cultural features via an LLM
3. `rule_learner.py` — convert survey scores into PyReason logic rules
   → writes `<output-dir>/<model>/<problem>/learned_rules/pyreason_rules.txt`

**Phase 2 (transformation)** — iteratively rewrite one test story (up to `--max-iterations`):
1. `llm_survey.py` — survey the current story
2. `graph_builder.py` — build PyReason ground atoms from the survey
3. `pyreason_runner.py` — run PyReason with the learned rules → trace files
4. `abduction.py` — rank features by the gap between current and max-possible corpus
   similarity (stops early when all gaps < 0.01)
5. `story_transformer.py` — rewrite the top-`k` features' segments, feed the result into
   the next iteration
   → writes `<output-dir>/phase2/<model>/<problem>/<story>/iteration_N/...`

### Route 2 — Baselines (four prompting techniques)

The comparison baselines: **no PyReason, no abduction.** Four LLM-only prompting techniques,
each a standalone script that transforms a story then surveys the result. All reuse the
original survey from Phase 2's `iteration_0`, so **Phase 2 must have run first** for the same
story/model/problem. See "Baseline prompting techniques" below for details:

| Technique | Script | Output dir |
|-----------|--------|------------|
| Zero-Shot | `baseline_transform.py` | `output/baseline_zeroshot` (also the legacy `output/baseline`) |
| Chain-of-Thought | `baseline_transform_cot.py` | `output/baseline_cot` |
| Few-Shot | `baseline_transform_fewshot.py` | `output/baseline_fewshot` |
| Self-Refine (iterative) | `baseline_transform_iterative.py` | `output/baseline_iterative` |

---

## Setup

```bash
pip install -r requirements.txt      # (requirements1.txt is a newer/alternate pin set)
```

Create a `.env` in the repo root with the provider keys you need (surveys/transforms go
through LiteLLM, so any supported provider works):

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
XAI_API_KEY=...
AWS_ACCESS_KEY_ID=...        # for Bedrock models
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=...
```

Python 3.10.

---

## How `--data-dir` works (important, non-obvious)

`--data-dir` does **not** point directly at a folder of stories. It points at a **parent
folder** that must contain:

- a corpus subfolder whose **name is hardcoded** in `story_loader.py`:
  - `inverse` → `<data-dir>/collectivistic-stories-all/`
  - `forward` → `<data-dir>/individualistic-rags-to-riches-stories-subset-subset/`
- the two questions files: `<data-dir>/collectivistic_questions.json` and
  `<data-dir>/individualistic_questions.json`

To train on a **custom / smaller** corpus without editing code, make a fresh parent dir with
those exact subfolder names and the questions files, then pass it as `--data-dir`. (See
`test_data/` in this repo for a 3-stories-per-direction example.)

---

## Quickstart (isolated run — never touches paper results)

Phase 1 writes rules under `--output-dir`, and Phase 2 reads rules only from `--rules`. So
pointing both at a **fresh** location keeps existing results in `output/` completely
untouched. This is the recommended way to experiment.

```bash
# --- INVERSE: individualistic -> collectivistic ---

# Phase 1: learn NEW rules into an isolated dir
python main.py --phase 1 --problem inverse --model gpt-4o \
  --data-dir test_data --output-dir output_test/phase1

# Phase 2: transform a fresh test story using those NEW rules
python main.py --phase 2 --problem inverse --model gpt-4o \
  --story test_inputs/individualistic/Ursula_Burns.txt \
  --rules output_test/phase1/gpt-4o/inverse/learned_rules/pyreason_rules.txt \
  --output-dir output_test \
  --max-iterations 3 --top-k 20
```

```bash
# --- FORWARD: collectivistic -> individualistic ---

python main.py --phase 1 --problem forward --model gpt-4o \
  --data-dir test_data --output-dir output_test/phase1

python main.py --phase 2 --problem forward --model gpt-4o \
  --story test_inputs/collectivistic/Back_To_The_Wall.txt \
  --rules output_test/phase1/gpt-4o/forward/learned_rules/pyreason_rules.txt \
  --output-dir output_test \
  --max-iterations 3 --top-k 20
```

Notes:
- Only **Phase 1** needs `--data-dir` to select the training corpus. Phase 2 uses `--data-dir`
  only to find the `*_questions.json` files (default `data` is fine).
- If you omit `--rules` in Phase 2, it auto-resolves to a **hardcoded** `output/phase1/...`
  path (see gotcha below) — pass `--rules` explicitly when working in an isolated dir.

### Baseline for the same story

```bash
python baseline_transform.py \
  --story test_inputs/individualistic/Ursula_Burns.txt \
  --problem inverse --model gpt-4o \
  --phase2-dir output_test/phase2 \
  --output-dir output_test/baseline
```

---

## Batch scripts

- `run_phase2_all_stories.sh` — loop Phase 2 over every `*.txt` in a story directory
  (`_grok`, `_llama`, `_llm4-cn`, `_singleprompt` are per-model/variant copies).
- `run_phase2_hyperparameter*.sh` — sweep iteration / segment / feature settings.
- `baseline_all_stories*.sh` — batch the baseline route.

Edit the `MODEL` / `PROBLEM` / `OUTPUT_DIR` variables at the top of each script.

---

## Adding test stories & running the full test set

This is the runbook for **growing the test set and (re)running both routes** (our approach +
baseline) over it. It works because both batch scripts glob **every `*.txt`** in the story
directory and **skip stories that already have results** — so you just drop new files in and
re-run; old stories are skipped, new ones are processed.

**1. Where test stories live (drop new `.txt` files here).** The story dir is chosen by
`$PROBLEM` inside the batch scripts:

| Direction | `--problem` | Test-story directory | New stories should be |
|-----------|-------------|----------------------|-----------------------|
| C→I | `forward` | `data/collectivistic-stories-all/`               | collectivistic |
| I→C | `inverse` | `data/individualistic-rags-to-riches-stories-subset/` | individualistic |

**2. Phase-1 rules must already exist for that model + direction — but you do NOT retrain for
new test stories.** Phase 2 learns nothing from the test story; it reads the rules at
`output/phase1/<model>/<problem>/learned_rules/pyreason_rules.txt` (the batch runs
`main_single.py` without `--rules`, so it auto-resolves to that hardcoded path). If those
rules exist (they do for the paper models), just run Phase 2. If you're adding a **new model**,
run Phase 1 once first (see Quickstart).

**3. Run our approach (Phase 2) over the whole dir.** Edit `PROBLEM` / `MODEL` at the top of
`run_phase2_all_stories.sh`, then:
```bash
bash run_phase2_all_stories.sh          # skips already-processed, does only the new stories
```
Writes `output/phase2/<model>/<problem>/<story>/iteration_*/…`.

**4. Run the baseline over the whole dir — AFTER step 3.** The baseline reuses Phase 2's
`iteration_0` survey, so Phase 2 must have run first for the same story/model/problem. Edit
`PROBLEM` / `MODEL` in `baseline_all_stories.sh`, then:
```bash
bash baseline_all_stories.sh            # reads output/phase2/…, writes output/baseline/…
```

**5. Re-run the evaluation pipeline — new stories are picked up automatically.** Every eval
script globs the output/story dirs, so once the new stories exist you just re-run the same
commands from the sections below (ratings → optimal iteration → divergences / BERTScore /
comparison → plots / tables). Nothing is hardcoded to the old story list.

> ⚠️ **Train/test leakage caveat.** `create_train_test_split.py` writes
> `data/train_test_split.csv`, but the batch scripts transform **all** `.txt` in the folder,
> ignoring that split. When you add stories, make sure any story you evaluate as *test* is not
> also in the *training* corpus that Phase 1 learned rules from for the relevant direction
> (the training-corpus folder names are hardcoded — see "How `--data-dir` works"). Keep the
> split file authoritative and, if in doubt, train Phase 1 on a corpus dir that excludes your
> test stories.

---

## Baseline prompting techniques

Four LLM-only baselines, each a standalone transform script (mirrors `baseline_transform.py`)
plus 6 per-model batch runners. All share the same args
(`--story --problem --model --temperature --max-tokens --data-dir --phase2-dir --output-dir`),
write to a technique-specific `output/baseline_<tech>/<model>/<problem>/<story>/`, and survey
only the final story. Prereq for all: the Phase-2 `iteration_0` survey must exist.

| Technique | Transform script | How it prompts |
|-----------|------------------|----------------|
| **Zero-Shot** | `baseline_transform.py` | single prompt: `Make the following story more {target}` |
| **Chain-of-Thought** | `baseline_transform_cot.py` | reason step-by-step, emit story in `<rewritten_story>` tags (parsed out) |
| **Few-Shot** | `baseline_transform_fewshot.py` | prepend expert example pairs from `data/fewshot_examples/<problem>/`, then the zero-shot instruction |
| **Self-Refine** | `baseline_transform_iterative.py` | 3 stateless calls: zero-shot → critique(original+transformed) → refine(transformed+critique, tag-parsed) |

### Batch runners (one per model, problem type as arg)

Each technique has 6 runners `baseline_all_stories_<tech>_<model>.sh` for
`{gpt4o, gpt52, grok, claude, llama, deepseek}`. Problem type is the **first arg**
(`forward` default, or `inverse`); each skips already-done stories **and** stories not in the
test set (no Phase-2 survey). Run one per terminal:

```bash
# <tech> ∈ {cot, fewshot, iterative}; zero-shot uses baseline_all_stories*.sh
bash baseline_all_stories_cot_grok.sh            # forward
bash baseline_all_stories_iterative_claude.sh inverse
```

Notes:
- **Few-shot examples** live in `data/fewshot_examples/<forward|inverse>/example*/{original,transformed}.txt`
  (held out from the test set; gitignored under `data/`, so copy them out-of-band to other machines).
- **CoT & Self-Refine** wrap the final story in `<rewritten_story>` tags and parse it out, so
  `transformed_story.txt` is clean (no model-specific preamble/commentary); the raw response is
  kept in `transformation_raw.txt` / `refine_raw.txt`.
- **Self-Refine** is the priciest (~4 calls/story: zero-shot + critique + refine + survey).

---

## Per-baseline evaluation (technique dimension)

Every eval script takes a `--baseline-dir` (or `--base-dir`) so you can compute KL / BERTScore /
Improvement % for **each** baseline against abduction, writing to technique-suffixed dirs so
nothing collides. The abduction side is identical across techniques; only the baseline box/column
changes. Example uses `<tech>` ∈ `{zeroshot, cot, fewshot, iterative}` and the matching
`output/baseline_<tech>` source (zero-shot = `output/baseline_zeroshot`).

**KL divergence** (per model × problem):
```bash
python plot_scripts/create_divergence_csv.py --model <M> --problem <P> \
  --baseline-dir output/baseline_<tech> --output-dir output_analysis/divergences_<tech>
python plot_scripts/plot_kl_boxplot_optimal_updated.py --model <M> --problem <P> \
  --divergence-dir output_analysis/divergences_<tech> --output-dir output_analysis/plots_<tech> \
  --baseline-label "CoT" --max-y-tick 5.0     # label per technique; 5.0 fits all 4 baselines
```

**BERTScore F1** (per model):
```bash
python extract_evaluation_stories.py --model_name <sanitized-model> \
  --baseline-dir output/baseline_<tech> --output-dir evaluation_data_<tech> --force
python plot_scripts/bertscore_analysis.py {collectivistic|individualistic} <sanitized-model> \
  --data-root evaluation_data_<tech> \
  --output-csv output_analysis/bert_scores_csv_<tech>/bertscore_results_<sanitized-model>_<forward|inverse>.csv
python plot_scripts/bertscore_f1_boxplot.py --model <M> --problem <P> \
  --bertscore-dir output_analysis/bert_scores_csv_<tech> --output-dir output_analysis/plots_<tech> \
  --baseline-label "CoT"     # y-axis default 0.75–1.01 fits all baselines
# run_bertscore_all_baselines.sh batches all 4 techniques × 6 models × 2 dirs (resumable)
```

**Improvement % table** (per model × problem):
```bash
python plot_scripts/extract_ratings_baseline.py --model <M> --problem <P> \
  --base-dir output/baseline_<tech> --output-dir output_analysis/baseline_<tech>
python plot_scripts/create_model_comparison_csv.py --models <all models> --problem <P> \
  --baseline-dir output_analysis/baseline_<tech> --output-dir output_analysis/cross_model_summary_<tech>
python plot_scripts/generate_results_tables.py --models <all models> \
  --summary-dir output_analysis/cross_model_summary_<tech> --output-dir output_analysis/tables_<tech>
```

Notes:
- **KL y-axis:** the paper zero-shot plots use `--max-y-tick 4.0`; the 4-baseline comparison uses
  a uniform **5.0** (covers the shared abduction outlier at ~4.31 — the peak is an abduction value,
  identical in every plot).
- **BERTScore is slow** (loads RoBERTa; ~90s per model/direction, no API). Use
  `run_bertscore_all_baselines.sh` and let it run.
- **Gotcha:** the CoT batch runners predate the "not-in-test-set" skip check, so a CoT-inverse run
  over the 90-story dir can leave partial dirs (only `original_story.txt`) that break
  `extract_ratings_baseline`. Delete any `output/baseline_cot/*/*/*/` dir lacking
  `transformed_survey.json` before extracting ratings.
- All suffixed outputs (`output_analysis/*_<tech>`, `evaluation_data_<tech>`) are gitignored.

### Combined comparison tables

Once the per-baseline KL / BERTScore / improvement outputs exist (the three blocks above,
run for all techniques), build the combined baseline-vs-abduction tables — rows = models,
columns = `Zero-Shot / CoT / Few-Shot / Self-Refine / Abduction` — for all three metrics in
one shot:

```bash
python plot_scripts/generate_baseline_comparison_tables.py
#   → output_analysis/tables_baselines/improvement_{forward,inverse}.csv
#   → output_analysis/tables_baselines/kl_median_{forward,inverse}.csv
#   → output_analysis/tables_baselines/bertscore_f1_median_{forward,inverse}.csv
```

- **improvement** = avg gap-to-target closed on the median rating (%, higher = better shift);
  **kl_median** = median KL (lower = better fidelity); **bertscore_f1_median** = median F1
  (higher = better fidelity).
- The **Abduction** column is identical across techniques (it's the method, independent of the
  baseline).
- Reads the suffixed dirs under `--analysis-dir` (default `output_analysis`); override
  `--models` / `--metrics` / `--output-dir` as needed.
- **Reading the results:** baselines tend to win the *improvement* (they shift aggressively),
  but abduction wins *KL and BERTScore* almost everywhere — i.e. baselines trade fidelity for
  shift, abduction keeps both.

#### Current results

Generated by the command above (median rating for improvement/KL; median F1 for BERTScore).
`Abduction` = our method. *GPT-4o reflects post-hoc edits and is excluded from paper validation.*

**Average Improvement % (higher = better shift) — Forward (C→I)**

| Model | Zero-Shot | Few-Shot | CoT | Self-Refine | Abduction |
|-------|-----------|----------|-----|-------------|-----------|
| GPT-4o* | 26.73 | 69.68 | 94.78 | 75.83 | 97.12 |
| GPT-5.2 | 33.38 | 60.88 | 70.56 | 64.75 | 70.24 |
| Grok-4 | 61.50 | 81.50 | 82.67 | 92.67 | 94.33 |
| Claude-4.5 | 48.63 | 77.34 | 80.19 | 82.69 | 67.34 |
| Llama-4 | 45.76 | 58.26 | 50.54 | 55.05 | 46.54 |
| DeepSeek-R1 | 56.79 | 90.12 | 80.09 | 66.67 | 71.91 |

**Average Improvement % (higher = better shift) — Inverse (I→C)**

| Model | Zero-Shot | Few-Shot | CoT | Self-Refine | Abduction |
|-------|-----------|----------|-----|-------------|-----------|
| GPT-4o* | 75.00 | 74.26 | 92.86 | 94.64 | 95.39 |
| GPT-5.2 | 46.64 | 61.39 | 70.03 | 90.22 | 85.16 |
| Grok-4 | 86.63 | 78.44 | 92.26 | 98.51 | 97.32 |
| Claude-4.5 | 75.02 | 64.81 | 86.18 | 96.58 | 96.13 |
| Llama-4 | 71.43 | 80.06 | 65.48 | 83.04 | 85.71 |
| DeepSeek-R1 | 87.67 | 92.43 | 90.05 | 98.81 | 73.68 |

**KL median (lower = better fidelity) — Forward (C→I)**

| Model | Zero-Shot | Few-Shot | CoT | Self-Refine | Abduction |
|-------|-----------|----------|-----|-------------|-----------|
| GPT-4o* | 1.21 | 1.56 | 1.79 | 1.96 | 0.72 |
| GPT-5.2 | 1.57 | 1.60 | 1.66 | 1.88 | 0.81 |
| Grok-4 | 1.48 | 1.58 | 0.97 | 1.85 | 0.93 |
| Claude-4.5 | 0.76 | 1.24 | 1.41 | 1.44 | 0.80 |
| Llama-4 | 1.61 | 1.78 | 1.85 | 1.85 | 0.82 |
| DeepSeek-R1 | 2.43 | 2.55 | 2.73 | 2.72 | 1.02 |

**KL median (lower = better fidelity) — Inverse (I→C)**

| Model | Zero-Shot | Few-Shot | CoT | Self-Refine | Abduction |
|-------|-----------|----------|-----|-------------|-----------|
| GPT-4o* | 1.66 | 1.70 | 1.98 | 2.30 | 1.06 |
| GPT-5.2 | 1.10 | 1.12 | 1.16 | 1.66 | 1.08 |
| Grok-4 | 1.68 | 1.07 | 0.99 | 1.76 | 0.67 |
| Claude-4.5 | 0.75 | 0.46 | 1.21 | 1.55 | 1.16 |
| Llama-4 | 1.48 | 1.62 | 1.74 | 1.75 | 0.94 |
| DeepSeek-R1 | 2.30 | 2.28 | 2.49 | 2.97 | 1.40 |

**BERTScore F1 median (higher = better fidelity) — Forward (C→I)**

| Model | Zero-Shot | Few-Shot | CoT | Self-Refine | Abduction |
|-------|-----------|----------|-----|-------------|-----------|
| GPT-4o* | 0.899 | 0.874 | 0.863 | 0.858 | 0.937 |
| GPT-5.2 | 0.854 | 0.856 | 0.869 | 0.826 | 0.926 |
| Grok-4 | 0.870 | 0.857 | 0.943 | 0.853 | 0.931 |
| Claude-4.5 | 0.930 | 0.910 | 0.915 | 0.899 | 0.947 |
| Llama-4 | 0.870 | 0.858 | 0.847 | 0.860 | 0.953 |
| DeepSeek-R1 | 0.831 | 0.825 | 0.832 | 0.822 | 0.924 |

**BERTScore F1 median (higher = better fidelity) — Inverse (I→C)**

| Model | Zero-Shot | Few-Shot | CoT | Self-Refine | Abduction |
|-------|-----------|----------|-----|-------------|-----------|
| GPT-4o* | 0.887 | 0.878 | 0.860 | 0.861 | 0.852 |
| GPT-5.2 | 0.894 | 0.884 | 0.879 | 0.845 | 0.900 |
| Grok-4 | 0.873 | 0.921 | 0.918 | 0.887 | 0.925 |
| Claude-4.5 | 0.931 | 0.940 | 0.903 | 0.876 | 0.917 |
| Llama-4 | 0.886 | 0.876 | 0.869 | 0.873 | 0.912 |
| DeepSeek-R1 | 0.854 | 0.850 | 0.847 | 0.832 | 0.918 |

### Significance tests (Wilcoxon signed-rank)

Per-story **paired** tests of Abduction vs each baseline (same test stories transformed by both),
run per `LLM × direction × metric × baseline` on all three metrics (diagnosis median rating, KL,
BERTScore F1):

```bash
python plot_scripts/wilcoxon_tests.py
#   → output_analysis/tables_baselines/wilcoxon_all.csv          (long: 144 tests)
#   → output_analysis/tables_baselines/wilcoxon_{forward,inverse}.csv  (paper layout)
```

Design (fixed):
- **Two-sided** `scipy.stats.wilcoxon`; report median Δ (abduction − baseline) **and** direction.
- **Holm** step-down correction **within each `LLM × direction × metric` family** across the 4
  baseline comparisons; both `p_raw` and `p_holm` are reported.
- **Effect size** = matched-pairs **rank-biserial correlation**, oriented so **+ = abduction better**
  (`effect_r`). Cite this + `p_holm` for direction/magnitude — do **not** read direction off median Δ.
- Significance stars off `p_holm`: `*<0.05  **<0.01  ***<0.001`, else `n.s.`

Gotchas / interpretation:
- **Diagnosis scores are discrete (1–5)**, so the median paired difference is frequently `0.0`
  even when the test is significant — that's why `effect_r` (not median Δ) determines `favors`.
- A non-significant result is **"no significant difference detected"**, *not* evidence of
  equivalence — phrase accordingly (no TOST run).
- Headline pattern from this dataset: abduction is **decisive on fidelity** (KL & BERTScore
  significant in its favor in ~16–22 of 24 cells/metric, r≈0.85–0.97, survives Holm) and at
  **parity on shift** (diagnosis mostly `n.s.` or mixed) — the Pareto result.



`plot_baselines_combined.py` draws one subfigure per model with **all 4 baselines
+ Abduction-Guided** in a single 5-box boxplot — for KL and BERTScore F1. This is the
paper's headline comparison (supersedes the per-baseline 2-box plots in `plots_<tech>/`).

Single figure:
```bash
python plot_scripts/plot_baselines_combined.py --metric {kl|bertscore} --model <M> --problem <P>
#   → figures-paper1/{kl_divergence|bertscore_f1}_<problem>_<sanitized-model>.png
```

All 24 (2 metrics × 6 models × 2 directions):
```bash
MODELS=(gpt-4o gpt-5.2 xai/grok-4-fast-reasoning claude-sonnet-4-5 \
        bedrock/us.meta.llama4-maverick-17b-instruct-v1:0 bedrock/us.deepseek.r1-v1:0)
for metric in kl bertscore; do
  for M in "${MODELS[@]}"; do for P in forward inverse; do
    python plot_scripts/plot_baselines_combined.py --metric "$metric" --model "$M" --problem "$P"
  done; done
done
```

- **Box order** (fixed, weakest→strongest, Ours last): `ZS, FS, CoT, SR, Abduction-Guided` —
  matches the tables above and lets the eye end on the method.
- **Colors:** the 4 baselines are green; **Abduction-Guided is the only red/filled box** (so
  "Ours" is visually distinct). X-ticks are abbreviated `ZS / FS / CoT / SR`.
- **Y-axis:** KL `0–5` (`--max-y-tick`, covers the shared abduction outlier ~4.31);
  BERTScore `0.75–1.01` (`--y-min` / `--y-max`).
- **Inputs:** reads the per-technique dirs under `--analysis-dir` (default `output_analysis`) —
  `divergences_<tech>/` + `phase2/…/optimal_iterations.csv` for KL, `bert_scores_csv_<tech>/`
  for BERTScore. Run the per-baseline eval steps above first.
- Filenames are suffixed with the model name, for dropping straight into the paper as subfigures.
  `figures-paper1/` is gitignored (regenerate anywhere with the loop above).

---

## Evaluation plots

### KL divergence boxplot

Compares lexical KL(transformed ‖ original) for the **baseline** vs the **abduction-guided**
route (lower = closer to the original story). The plot is built in three steps, run **per
model and per problem type** (`forward` / `inverse`). Example below uses grok-4; swap
`--model` / `--problem` for others.

```bash
# 1. Compute per-story, per-iteration KL (and JS) values from the transformed stories
#    → writes output_analysis/divergences/<model>/<problem>/kl_divergence_transformed_original.csv
python plot_scripts/create_divergence_csv.py \
  --model xai/grok-4-fast-reasoning --problem forward

# 2. Pick the optimal iteration per story (from the phase-2 ratings CSVs)
#    → writes output_analysis/phase2/<model>/<problem>/optimal_iterations.csv
python plot_scripts/find_optimal_iteration.py \
  --model xai/grok-4-fast-reasoning --problem forward --max-iterations 3

# 3. Draw the boxplot (Baseline vs Abduction-Guided) at the optimal iteration
#    → writes output_analysis/plots/<model>/<problem>/kl_divergence_boxplot_optimal.png
python plot_scripts/plot_kl_boxplot_optimal_updated.py \
  --model xai/grok-4-fast-reasoning --problem forward --max-y-tick 4.0
```

Notes:
- **Use `plot_kl_boxplot_optimal_updated.py`** (the `_updated` version). It forces a fixed
  y-axis of `0` to `--max-y-tick` (4.0), which is the range used in the paper. The plain
  `plot_kl_boxplot_optimal.py` auto-scales the y-axis for non-Llama models and does **not**
  match.
- Steps 1 and 2 are independent and can run in either order; step 3 needs both their CSVs.
  Step 2 reads the phase-2 ratings CSVs (`original_ratings.csv`,
  `transformed_ratings_iteration_*.csv`) produced by the ratings-extraction step.
- Model names with `/` or `:` are sanitized to `-` in output paths (e.g.
  `xai/grok-4-fast-reasoning` → `output_analysis/.../xai-grok-4-fast-reasoning/...`).
- To experiment without overwriting paper plots, add `--output-dir output_test/plots` to
  step 3 (and `--output-dir output_test/divergences` to step 1).

### BERTScore F1 boxplot

Compares the semantic similarity (BERTScore F1) of the transformed story to the original for
the **baseline** vs the **abduction-guided** route (higher = more of the original meaning
retained). Unlike KL, BERTScore reads from a separate `evaluation_data/` folder that must be
populated first, and that step reuses the `optimal_iterations.csv` from the KL pipeline. Run
**per model**; example uses grok-4.

```bash
# Prereq: optimal_iterations.csv must already exist for BOTH forward and inverse
#   (run find_optimal_iteration.py first — same as KL step 2)

# 1. Copy each story's original + optimal-iteration abduction + baseline into evaluation_data/
#    → evaluation_data/<model>/{collectivistic,individualistic}/<story>/
#         {original_story,abduction_transformed,baseline_transformed}.txt
python extract_evaluation_stories.py --model_name xai-grok-4-fast-reasoning

# 2. Compute BERTScore P/R/F1 (original vs transformed), one call per direction
#    → output_analysis/bert_scores_csv/bertscore_results_<model>_<direction>.csv
python plot_scripts/bertscore_analysis.py collectivistic  xai-grok-4-fast-reasoning   # forward
python plot_scripts/bertscore_analysis.py individualistic xai-grok-4-fast-reasoning   # inverse

# 3. Draw the F1 boxplot (Baseline vs Abduction-Guided)
#    → output_analysis/plots/<model>/<problem>/bertscore_f1_boxplot.png
python plot_scripts/bertscore_f1_boxplot.py --model xai/grok-4-fast-reasoning --problem forward
python plot_scripts/bertscore_f1_boxplot.py --model xai/grok-4-fast-reasoning --problem inverse
```

Notes:
- **Naming differs between steps.** Steps 1 and 2 take the **already-sanitized** folder name
  (`xai-grok-4-fast-reasoning`); step 3 takes `--model` and sanitizes it itself, so
  `xai/grok-4-fast-reasoning` is fine there.
- **Step 2's first positional is the *original* direction:** `collectivistic` → forward,
  `individualistic` → inverse (the script infers the transform target and the CSV's
  `_forward` / `_inverse` suffix from it).
- The F1 boxplot y-axis is **hardcoded to 0.75–1.01** (no `--max-y-tick` flag; that's the
  paper range).
- To experiment without overwriting paper plots, add `--output-dir output_test/plots` to
  step 3.

---

## Results tables (Improvement %)

The paper's "Average improvement over original stories" tables (forward C→I and inverse I→C)
report, per model, the **% of the gap-to-target closed** on the **Median** cultural rating,
averaged over all stories:

- **Forward** (target = 1.0, lower is better): `(original − value) / (original − 1.0) × 100`
- **Inverse** (target = 5.0, higher is better): `(value − original) / (5.0 − original) × 100`
- **Zero-Shot** row uses the baseline value; **Abduction** row uses the transformed
  (optimal-iteration) value.

Pipeline (per model; example uses grok-4). Steps 1–2 are the same ratings/optimal-iteration
CSVs used by the KL plots:

```bash
# 1. Aggregate per-story ratings into Mean/Median/Mode
#    → output_analysis/phase2/<model>/<problem>/{original_ratings,transformed_ratings_iteration_*}.csv
#    → output_analysis/baseline/<model>/<problem>/baseline_transformed_ratings.csv
python plot_scripts/extract_ratings.py          --model xai/grok-4-fast-reasoning --problem forward --max-iterations 3
python plot_scripts/extract_ratings_baseline.py --model xai/grok-4-fast-reasoning --problem forward

# 2. Pick the optimal iteration per story
#    → output_analysis/phase2/<model>/<problem>/optimal_iterations.csv
python plot_scripts/find_optimal_iteration.py --model xai/grok-4-fast-reasoning --problem forward --max-iterations 3

# 3. Assemble the per-story original/baseline/transformed comparison (writes mean, median, mode)
#    → output_analysis/cross_model_summary/<problem>/model_comparison_median.csv
python plot_scripts/create_model_comparison_csv.py --models xai/grok-4-fast-reasoning --problem forward
python plot_scripts/create_model_comparison_csv.py --models xai/grok-4-fast-reasoning --problem inverse

# 4. Compute the Improvement (%) tables (forward + inverse in one call)
#    → output_analysis/tables/table_{forward,inverse}_results.csv
python plot_scripts/generate_results_tables.py --models xai/grok-4-fast-reasoning
```

Notes:
- **The table script reads only `model_comparison_median.csv`** (median, not mean/mode),
  even though step 3 writes all three.
- **`--models` strings must match exactly between steps 3 and 4.** The CSV column prefix is
  the raw model name (the short-name mapping in the code is disabled, so `model_short = model`),
  i.e. columns are literally `xai/grok-4-fast-reasoning_original`. Pass the identical string
  to both scripts.
- To verify without touching paper artifacts, point steps 3–4 at test dirs
  (`--output-dir output_test/cross_model_summary` and `--output-dir output_test/tables`,
  with a matching `--summary-dir` on step 4).

### Fine-tuned model tables (diagnosis-model dimension)

The fine-tuned-model tables (LLM1–LLM4, **inverse only**) come from a **parallel copy** of the
pipeline in `plot_scripts_standalone/`. The difference: the model that *transforms* is
separated from the model that *diagnoses* (surveys/rates), so every output path gains a
`/{survey_model}/` level. That's why there are two tables for the same models — one per
diagnosis model:

- `gpt-4o-mini-2024-07-18` diagnosis → `output_analysis_standalone/tables/gpt-4o-mini-2024-07-18/individualistic/table_results.csv`
- `gpt-5.2` diagnosis → `output_analysis_standalone/tables/gpt-5.2/individualistic/table_results.csv`

Paper name → directory:

| Paper | `--tuned-model-name(s)` value | Fine-tuning |
|-------|------------------------------|-------------|
| LLM1    | `gpt-4o-mini-2024-07-18`                              | none (base) |
| LLM2    | `ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi`     | collectivistic narratives |
| LLM3    | `ft-gpt-4o-mini-2024-07-18-syracuse-university-llm3-D75Ahi1l`     | abduction-guided shifts |
| LLM4-CH | `ft-gpt-4o-mini-2024-07-18-syracuse-university-llm4-D5NuhUdq`     | Chinese recipes (Chinese) |
| LLM4-EN | `ft-gpt-4o-mini-2024-07-18-syracuse-university-llm4-en-D6rDdxB0`  | Chinese recipes (English) |

Pipeline (I→C / `individualistic`). The `evaluation_data/<tuned_model>/` folders must already
be populated (`extract_evaluation_stories.py`, see BERTScore section). Set `SURVEY` to the
diagnosis model and re-run the whole block for each one (`gpt-4o-mini-2024-07-18`, then
`gpt-5.2`):

```bash
SURVEY=gpt-5.2   # or gpt-4o-mini-2024-07-18
MODELS=(
  gpt-4o-mini-2024-07-18
  ft-gpt-4o-mini-2024-07-18-syracuse-university-llm2-D5JJuHZi
  ft-gpt-4o-mini-2024-07-18-syracuse-university-llm3-D75Ahi1l
  ft-gpt-4o-mini-2024-07-18-syracuse-university-llm4-D5NuhUdq
  ft-gpt-4o-mini-2024-07-18-syracuse-university-llm4-en-D6rDdxB0
)

# 1+2. Per tuned model: survey the pre-extracted stories with the DIAGNOSIS model, then aggregate ratings
#      survey   → survey/<tuned>/<survey>/individualistic/<story>/survey_*.json
#      ratings  → output_analysis_standalone/survey/<tuned>/<survey>/individualistic/{original_ratings,baseline_transformed_ratings,abduction_ratings}.csv
#                 (this is create_model_comparison's default --survey-dir, so the steps chain with no extra flags)
for M in "${MODELS[@]}"; do
  python llm_survey_standalone.py \
    --tuned-model-name "$M" --model-name-for-survey "$SURVEY" \
    --original-stories-type individualistic \
    --questions-file data/collectivistic_questions.json
  python plot_scripts_standalone/extract_ratings_standalone.py \
    --tuned-model-name "$M" --model-name-for-survey "$SURVEY" \
    --original-stories-type individualistic
done

# 3. Assemble the per-story comparison across ALL tuned models (note: --tuned-model-names, plural)
#    → output_analysis_standalone/cross_model_summary/<survey>/individualistic/model_comparison_median.csv
python plot_scripts_standalone/create_model_comparison_csv_standalone.py \
  --tuned-model-names "${MODELS[@]}" --model-name-for-survey "$SURVEY" \
  --original-stories-type individualistic

# 4. Compute the Improvement (%) table
#    → output_analysis_standalone/tables/<survey>/individualistic/table_results.csv
python plot_scripts_standalone/generate_results_tables_standalone.py \
  --tuned-model-names "${MODELS[@]}" --model-name-for-survey "$SURVEY" \
  --original-stories-type individualistic
```

Notes:
- **Same Improvement (%) formula** as the main results tables (gap-to-target closed on the
  Median rating); the only new axis is `--model-name-for-survey` (the diagnosis model).
- **Singular vs plural:** steps 1–2 take `--tuned-model-name` (one model, run in a loop);
  steps 3–4 take `--tuned-model-names` (the full list at once).
- These are **inverse only** (`--original-stories-type individualistic`,
  `data/collectivistic_questions.json`).

---

## Key arguments (`main.py`)

| Arg | Applies to | Meaning |
|-----|-----------|---------|
| `--phase {1,2}` | both | 1 = train rules, 2 = transform a story |
| `--problem {forward,inverse}` | both | direction (see table above) |
| `--model` | both | LLM id (default `gpt-4o`; any LiteLLM provider) |
| `--temperature` | both | LLM temperature (default 0.7) |
| `--data-dir` | both | parent folder of corpora + questions files |
| `--output-dir` | both | where results are written |
| `--story` | Phase 2 | path to the single story to transform |
| `--rules` | Phase 2 | learned rules file (auto-resolved if omitted) |
| `--max-iterations` | Phase 2 | max transform iterations (default 3) |
| `--top-k` | Phase 2 | features rewritten per iteration (default 3) |
| `--skip-survey` | both | reuse saved surveys instead of calling the LLM — **behaves differently per phase** (see below) |

### `--skip-survey` behaviour (differs by phase)

Surveying is the expensive step (a 20-feature LLM call per story). This flag reuses saved
survey JSON to avoid paying again — but the two phases handle it differently:

- **Phase 2 — per-iteration existence check.** For each iteration, if a `survey.json`
  already exists in that iteration folder it is loaded (no LLM call); if it is missing the
  LLM is called to survey. So: *has survey → skip, missing → run.* On a brand-new story with
  no saved survey, the flag effectively does nothing.
- **Phase 1 — blunt all-or-nothing switch, no per-story check.**
  - *Without* the flag: surveys **all** training stories (no reuse, even if some already
    have surveys).
  - *With* `--skip-survey`: surveys **none** and just reads whatever survey JSONs already
    exist in `<output-dir>/<model>/<problem>/survey_results/`. A story whose survey is
    missing is **not** back-filled by the LLM — it is simply absent from rule learning, and
    if the folder is empty `rule_learner.py` has nothing to work with and fails.

---

## Repo layout (orientation)

- **Pipeline core:** `main.py`, `story_loader.py`, `llm_survey.py`, `rule_learner.py`,
  `graph_builder.py`, `pyreason_runner.py`, `abduction.py`, `story_transformer.py`
- **Variants:** `*_single.py` (single-prompt), `*_hf.py` / `hf_inference.py` (local
  HuggingFace models), `main_single.py`, `main_hf.py`
- **Baseline:** `baseline_transform.py`
- **Data prep / fine-tuning:** `create_*dataset*.py`, `fine_tune.py`, `train_llm2.py`,
  `create_train_test_split.py`
- **Evaluation / plots:** `plot_scripts/`, `plot_scripts_standalone/`,
  `extract_*.py`, `compute_*.py`
- **Data:** `data/` (corpora + `*_questions.json`); `test_data/` and `test_inputs/` are the
  small isolated fixtures used by the quickstart above.

---

## Known gotchas (worth a future cleanup)

1. **`forward`/`inverse` help text is inconsistent/reversed** in `main.py` and
   `baseline_transform.py` docstrings. The code behaviour is authoritative (see table).
2. **`--data-dir` is a parent folder, not a stories folder** — the actual corpus subfolder
   name is hardcoded in `story_loader.py`. Consider refactoring to accept a real stories path.
3. **Phase 2 auto-rules default is hardcoded to `output/phase1/...`**, which does *not* match
   where Phase 1 writes when you pass a custom `--output-dir`. Always pass `--rules`
   explicitly for isolated runs.