# Baseline prompt templates (Appendix C)

Verbatim prompt templates for the four LLM-only baselines, extracted from the transform scripts.
Placeholders:

- `{target}` / `{source}` — direction-dependent framing:
  - **Forward (C→I):** `{target}` = *individualistic*, `{source}` = *collectivistic*
  - **Inverse (I→C):** `{target}` = *collectivistic*, `{source}` = *individualistic*
- `{story}` — the input story text.
- All models are called at temperature 0.7. Transformation calls use max_tokens 8096;
  the diagnosis/survey step (separate, not shown here) uses max_tokens 4096.

Files:
- `zero_shot.md` — Zero-Shot (`baseline_transform.py`)
- `few_shot.md` — Few-Shot (`baseline_transform_fewshot.py`)
- `cot.md` — Chain-of-Thought (`baseline_transform_cot.py`)
- `self_refine.md` — Self-Refine, three prompts (`baseline_transform_iterative.py`)
