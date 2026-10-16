"""
Combine the russio results for the 4 questions (1 = war relevance, 2-4 from russio_questions_extra.json)
into one 4-question survey and one 4-question prediction set per approach, so evaluate.py can score them
together (per question, overall, per language, speed).

Each approach has one SEPARATE model per question (models/<name>/ for question 1, models/<name>_q<id>/ for
2-4). Answering all 4 questions for one article then costs:
  emb_lr, emb_mlp, chunk_transformer   ONE embedding request (shared by the 4 models) + 4 model predictions
  xlmr_base                            4 forward passes (4 fine-tuned models), no API call

LLM reference: question 1 and questions 2-4 were asked in two separate calls. A deployed survey would
ask all 4 in one call, so its time and cost are taken from the questions 2-4 call (the article dominates
the input, so a 4th answer adds very little); the two measured calls are kept under timing.measured_calls.
This keeps the student models' speed-ups conservative.

Writes (row order and names as in data/russio-prototype/split.csv):
  prototype/russio_questions_all4.json                      the 4-question questionnaire (for evaluate.py)
  output_test/prototype/russio/survey_4q/gpt-5.4/test/       4-question LLM answers per test article
  output_test/prototype/russio/predictions/<name>_4q/test/   4-question predictions per test article
  output_test/prototype/russio/models/<name>_4q/             config with each question's CV result

Usage (from repo root, with the ConnectPyreason .venv310 active):
    python prototype/russio_merge_questions.py
"""

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "russio-prototype"
OUT = REPO / "output_test" / "prototype" / "russio"
QUESTIONS = ["1", "2", "3", "4"]
APPROACHES = {  # folder name -> (label, how the 4 models share work per article)
    "emb_lr": ("OpenAI text-embedding-3-large + logistic regression", "shared_embedding"),
    "emb_mlp": ("OpenAI text-embedding-3-large + MLP", "shared_embedding"),
    "xlmr_base": ("fine-tuned xlm-roberta-base", "separate_passes"),
    "chunk_transformer": ("transformer (2 layers, d=256) over OpenAI chunk embeddings", "shared_embedding"),
}


def folder(name: str, q: str) -> str:
    return name if q == "1" else f"{name}_q{q}"


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_questionnaire() -> Path:
    war = load(REPO / "prototype" / "russio_questions_war.json")
    extra = load(REPO / "prototype" / "russio_questions_extra.json")
    path = REPO / "prototype" / "russio_questions_all4.json"
    out = {"source": "LLM-free survey prototype, russio dataset: the 4 yes/no questions (1 from russio_questions_war.json, "
                     "2-4 from russio_questions_extra.json)",
           "text_label": "TEXT", "questions": war["questions"] + extra["questions"]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return path


def main():
    qfile = write_questionnaire()
    with open(DATA / "split.csv", encoding="utf-8") as f:
        test = [r["story_name"] for r in csv.DictReader(f) if r["split"] == "test"]

    # ---------- LLM: the 4 answers per test article ----------
    sdir = OUT / "survey_4q" / "gpt-5.4" / "test"
    sdir.mkdir(parents=True, exist_ok=True)
    for doc in test:
        war = load(OUT / "survey_war" / "gpt-5.4" / "test" / f"{doc}.json")
        extra = load(OUT / "survey_extra" / "gpt-5.4" / "test" / f"{doc}.json")
        merged = {**extra, "questions_file": str(qfile.relative_to(REPO)),
                  "ratings": {**war["ratings"], **extra["ratings"]},
                  "n_failed": war["n_failed"] + extra["n_failed"],
                  "timing": {**extra["timing"], "note": "one-call estimate = the questions 2-4 call",
                             "measured_calls": {"question_1": war["timing"], "questions_2_4": extra["timing"]}},
                  "cost_usd": extra["cost_usd"]}
        with open(sdir / f"{doc}.json", "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"LLM survey, 4 questions: {len(test)} test articles -> {sdir}")

    # ---------- students: 4 separate models per approach ----------
    for name, (label, sharing) in APPROACHES.items():
        dirs = {q: OUT / "predictions" / folder(name, q) / "test" for q in QUESTIONS}
        missing = [q for q, d in dirs.items() if not d.exists() or len(list(d.glob("*.json"))) != len(test)]
        if missing:
            print(f"{name}: skipped, no complete test predictions yet for question(s) {', '.join(missing)}")
            continue
        model_dir = OUT / "models" / f"{name}_4q"
        pred_dir = OUT / "predictions" / f"{name}_4q" / "test"
        model_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        first = {q: load(next(dirs[q].glob("*.json"))) for q in QUESTIONS}
        configs = {}
        for q in QUESTIONS:
            m = first[q]["model"]["file"]
            configs[q] = load(Path(m).with_name(Path(m).stem + "_config.json"))
        model_file = model_dir / f"{name}_4q_final.pt"
        cfg = {"model": f"{label}, one model per question", "per_question_models": {q: first[q]["model"]["file"] for q in QUESTIONS},
               "n_classes": 2, "answer_values": [0, 1], "answer_scale": "yes/no (1/0)", "question_ids": QUESTIONS,
               "cv_per_question": {q: configs[q]["cv_mean"] for q in QUESTIONS},
               "cv_mean": {k: round(float(np.mean([configs[q]["cv_mean"][k] for q in QUESTIONS])), 4)
                           for k in configs["1"]["cv_mean"]}}
        with open(model_dir / f"{name}_4q_final_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        info = {"name": cfg["model"], "file": str(model_file),
                "n_parameters": int(sum(first[q]["model"]["n_parameters"] for q in QUESTIONS)),
                "size_bytes": int(sum(first[q]["model"]["size_bytes"] for q in QUESTIONS))}

        for doc in test:
            p = {q: load(dirs[q] / f"{doc}.json") for q in QUESTIONS}
            predict_s = sum(p[q]["timing"]["predict_latency_s"] for q in QUESTIONS)
            if sharing == "shared_embedding":             # one embedding call serves all 4 models
                embed_s = p["1"]["timing"]["embed_latency_s"]
                wall_s, cost = embed_s + predict_s, p["1"]["cost_usd"]
            else:                                         # 4 separate end-to-end passes
                embed_s = 0.0
                wall_s, cost = sum(p[q]["timing"]["story_wall_s"] for q in QUESTIONS), 0.0
            out = {**{k: v for k, v in p["1"].items() if k not in ("ratings", "probabilities", "timing", "model", "cost_usd")},
                   "model": info,
                   "ratings": {q: p[q]["ratings"][q] for q in QUESTIONS},
                   "probabilities": {q: p[q]["probabilities"][q] for q in QUESTIONS},
                   "n_failed": sum(p[q]["n_failed"] for q in QUESTIONS),
                   "timing": {"story_wall_s": round(wall_s, 6), "embed_latency_s": round(embed_s, 6),
                              "predict_latency_s": round(predict_s, 6)},
                   "cost_usd": cost}
            with open(pred_dir / f"{doc}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"{name}: 4 questions merged -> {pred_dir}")


if __name__ == "__main__":
    main()
