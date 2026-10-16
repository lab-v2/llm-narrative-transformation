"""
Build a yes/no version of a survey questionnaire for the LLM-free survey prototype.

Each question keeps its id, component and question text word for word (the questions
are already phrased as yes/no questions); only the 1-5 rating scale and the
excerpt-quoting instruction are replaced by "1 = Yes, 0 = No".

Usage (from repo root):
    python3 prototype/make_binary_questions.py \
        --input data/collectivistic_questions.json \
        --output prototype/collectivistic_questions_binary.json
"""

import argparse
import json
from pathlib import Path

INSTRUCTIONS = [
    "You are an expert literary analyst.",
    "Given a story, answer the question about its narrative perspective with yes or no.",
    "Answers:",
    "1 = Yes",
    "0 = No",
]


def main():
    parser = argparse.ArgumentParser(description="Make a yes/no version of a survey questionnaire")
    parser.add_argument("--input", type=str, default="data/collectivistic_questions.json")
    parser.add_argument("--output", type=str, default="prototype/collectivistic_questions_binary.json")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        source = json.load(f)

    questions = []
    for q in source["questions"]:
        out = {"id": q["id"], "component": q["component"]}
        for key, value in q.items():
            if key.endswith("_question"):
                out[key] = {
                    "question": value["question"],
                    "type": "binary",
                    "answers": {"1": "Yes", "0": "No"},
                    "instructions": INSTRUCTIONS,
                }
        questions.append(out)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"source": args.input, "questions": questions}, f, indent=2, ensure_ascii=False)
    print(f"{len(questions)} questions -> {args.output}")


if __name__ == "__main__":
    main()
