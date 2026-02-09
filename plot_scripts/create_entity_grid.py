"""
Entity Grid Creation Script

Builds an entity grid (sentences x entities) for one or more stories.
Each cell is labeled as:
  S = entity appears as sentence subject
  O = entity appears as sentence object
  X = entity appears in another role
  - = entity absent

Outputs CSV files per input and a shared entity list when requested.
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import spacy

logger = logging.getLogger(__name__)

# ==========================================================
# Entity Grid Helpers
# ==========================================================

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECT_DEPS = {"dobj", "pobj", "iobj", "obj", "dative", "attr", "oprd"}

ROLE_ORDER = {"S": 3, "O": 2, "X": 1, "-": 0}


def normalize_span(span, include_pronouns=False):
    tokens = []
    for token in span:
        if token.is_space or token.is_punct:
            continue
        if not include_pronouns and token.pos_ == "PRON":
            continue
        if token.is_stop and token.pos_ in {"DET", "ADP", "PRON"}:
            continue
        lemma = token.lemma_.lower().strip()
        if lemma:
            tokens.append(lemma)

    if not tokens:
        head = span.root
        if include_pronouns or head.pos_ != "PRON":
            lemma = head.lemma_.lower().strip()
            if lemma:
                tokens = [lemma]

    if not tokens:
        return None

    return " ".join(tokens)


def role_for_span(span):
    dep = span.root.dep_
    if dep in SUBJECT_DEPS:
        return "S"
    if dep in OBJECT_DEPS:
        return "O"
    return "X"


def update_role(existing, new):
    if ROLE_ORDER[new] > ROLE_ORDER.get(existing, 0):
        return new
    return existing


def extract_sentence_mentions(doc, include_pronouns=False):
    sents = list(doc.sents)
    chunks_by_sent = defaultdict(list)
    ents_by_sent = defaultdict(list)

    for chunk in doc.noun_chunks:
        chunks_by_sent[chunk.sent.start].append(chunk)

    for ent in doc.ents:
        ents_by_sent[ent.sent.start].append(ent)

    mentions_per_sent = []
    counts = defaultdict(int)

    for sent in sents:
        sent_key = sent.start
        mentions = {}
        spans = chunks_by_sent.get(sent_key, []) + ents_by_sent.get(sent_key, [])

        for span in spans:
            key = normalize_span(span, include_pronouns=include_pronouns)
            if not key:
                continue
            role = role_for_span(span)
            mentions[key] = update_role(mentions.get(key, "-"), role)
            counts[key] += 1

        mentions_per_sent.append((sent, mentions))

    return sents, mentions_per_sent, counts


def select_entities(counts, min_count=1, max_entities=50):
    items = [(entity, cnt) for entity, cnt in counts.items() if cnt >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    entities = [entity for entity, _ in items]
    if max_entities is not None:
        entities = entities[:max_entities]
    return entities


def write_entity_grid_csv(output_path, sentences, mentions_per_sent, entities):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["sentence_index", "sentence_text"] + entities

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for idx, (sent, mentions) in enumerate(mentions_per_sent):
            row = [str(idx), '"' + sent.text.replace('"', '""') + '"']
            for entity in entities:
                row.append(mentions.get(entity, "-"))
            f.write(",".join(row) + "\n")


def write_entity_list(output_path, entities):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entity in entities:
            f.write(entity + "\n")


# ==========================================================
# Main
# ==========================================================


def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' not found. "
            f"Install it or choose another with --model. Error: {exc}"
        ) from exc


def build_grid_for_file(nlp, input_path, include_pronouns, min_count, max_entities):
    text = Path(input_path).read_text(encoding="utf-8")
    doc = nlp(text)
    sentences, mentions_per_sent, counts = extract_sentence_mentions(
        doc, include_pronouns=include_pronouns
    )
    entities = select_entities(counts, min_count=min_count, max_entities=max_entities)
    return sentences, mentions_per_sent, counts, entities


def main():
    parser = argparse.ArgumentParser(description="Create entity grids for story files.")
    parser.add_argument("--original", type=str, help="Path to original story text file.")
    parser.add_argument("--transformed", type=str, help="Path to transformed story text file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_analysis/entity_grids",
        help="Directory to write entity grid CSVs.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="en_core_web_sm",
        help="spaCy model name to use.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum number of mentions to include an entity.",
    )
    parser.add_argument(
        "--max-entities",
        type=int,
        default=50,
        help="Maximum number of entities to include (most frequent).",
    )
    parser.add_argument(
        "--include-pronouns",
        action="store_true",
        help="Include pronouns as entities.",
    )
    parser.add_argument(
        "--shared-entities",
        action="store_true",
        help="Use a shared entity list across inputs for aligned grids.",
    )

    args = parser.parse_args()

    inputs = []
    if args.original:
        inputs.append(("original", args.original))
    if args.transformed:
        inputs.append(("transformed", args.transformed))

    if not inputs:
        raise SystemExit("No inputs provided. Use --original and/or --transformed.")

    nlp = load_spacy_model(args.model)

    grids = {}
    global_counts = defaultdict(int)

    for label, path in inputs:
        sentences, mentions_per_sent, counts, entities = build_grid_for_file(
            nlp,
            path,
            include_pronouns=args.include_pronouns,
            min_count=args.min_count,
            max_entities=args.max_entities,
        )
        grids[label] = (sentences, mentions_per_sent, counts, entities)
        for entity, cnt in counts.items():
            global_counts[entity] += cnt

    shared_entities = None
    if args.shared_entities and len(inputs) > 1:
        shared_entities = select_entities(
            global_counts, min_count=args.min_count, max_entities=args.max_entities
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, _ in inputs:
        sentences, mentions_per_sent, counts, entities = grids[label]
        entity_list = shared_entities if shared_entities is not None else entities

        csv_path = output_dir / f"{label}_entity_grid.csv"
        entities_path = output_dir / f"{label}_entities.txt"

        write_entity_grid_csv(csv_path, sentences, mentions_per_sent, entity_list)
        write_entity_list(entities_path, entity_list)

        logger.info(f"Wrote {csv_path}")
        logger.info(f"Wrote {entities_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
