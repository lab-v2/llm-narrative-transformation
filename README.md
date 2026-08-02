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

### Route 2 — Baseline (`baseline_transform.py`)

The comparison baseline: **no PyReason, no abduction, no iteration.** A single prompt
("Make the following story more {individualistic|collectivistic}"), then survey the result.
It reuses the original survey from Phase 2's `iteration_0`, so **Phase 2 must have run first**
for the same story/model/problem.

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

## Different prompting techniques (design note — not yet implemented)

> This section is a **build plan**, not a runbook. As of today the only non-neurosymbolic
> route is the **hardcoded zero-shot baseline**; few-shot / CoT / self-refine / etc. do **not**
> exist in the code yet. Read this before implementing so the outputs slot into the existing
> evaluation pipeline for free.

**What exists today.** `baseline_transform.py` builds a single fixed prompt and has **no
strategy flag**:
```python
# baseline_transform.py → create_baseline_prompt()
prompt = f"Make the following story more {target}:\n\n{story_text}"
# system_prompt = "You are a helpful assistant."
```
Args are only `--story --problem --model --data-dir --output-dir --temperature`.

**To add prompting techniques, two pieces are needed:**

1. **Generation.** Add a `--strategy` (e.g. `zero_shot` | `few_shot` | `cot` | `self_refine`)
   to `baseline_transform.py` (or a sibling `prompt_transform.py`) that swaps the
   system/user prompt and any exemplars. Run it over the **full** test set (old + new) per
   model × direction, the same way `baseline_all_stories.sh` loops.

2. **Output layout = the whole trick.** The eval scripts locate the baseline at
   `output/baseline/<model>/<problem>/<story>/transformed_story.txt`. To reuse ratings /
   divergence / BERTScore / tables **without editing them**, treat "technique" as a new path
   dimension — mirror what the standalone pipeline did for the *survey model*:
   `output/<strategy>/<model>/<problem>/<story>/transformed_story.txt`.
   Then either (a) point each eval script's `--base-dir` / `--output-dir` at the strategy dir
   where those flags already exist (`extract_ratings_baseline.py --base-dir`,
   `create_divergence_csv.py`, etc.), or (b) thread a `--strategy` arg through
   `extract_ratings` → `create_model_comparison_csv` → `generate_results_tables`, exactly like
   `--model-name-for-survey` is threaded through the `plot_scripts_standalone/` copies. Option
   (b) gives you clean per-technique tables with the existing gap-closed Improvement % formula.

**Order of operations when you get to it:** finish "Adding test stories" above so the test set
is final → run each strategy's generation over the full set → run the eval pipeline once per
strategy → the results tables/plots gain a technique dimension.

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