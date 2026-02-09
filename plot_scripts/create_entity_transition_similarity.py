"""
Entity Transition Similarity Script

Computes entity grid role transitions across adjacent sentences and compares
original vs transformed stories using cosine similarity.

Roles:
  S = subject
  O = object
  X = other
  - = absent
"""

import argparse
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import spacy

logger = logging.getLogger(__name__)

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "expl"}
OBJECT_DEPS = {"dobj", "pobj", "iobj", "obj", "dative", "attr", "oprd"}

ROLE_ORDER = {"S": 3, "O": 2, "X": 1, "-": 0}
ROLES = ["S", "O", "X", "-"]


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

        mentions_per_sent.append(mentions)

    return sents, mentions_per_sent, counts


def select_entities(counts, min_count=1, max_entities=50):
    items = [(entity, cnt) for entity, cnt in counts.items() if cnt >= min_count]
    items.sort(key=lambda x: (-x[1], x[0]))
    entities = [entity for entity, _ in items]
    if max_entities is not None:
        entities = entities[:max_entities]
    return entities


def build_entity_roles_for_text(nlp, text, include_pronouns, min_count, max_entities):
    doc = nlp(text)
    sents, mentions_per_sent, counts = extract_sentence_mentions(
        doc, include_pronouns=include_pronouns
    )
    entities = select_entities(counts, min_count=min_count, max_entities=max_entities)
    return sents, mentions_per_sent, entities, counts


def compute_entity_transitions(mentions_per_sent, entities):
    transitions = {entity: defaultdict(int) for entity in entities}

    for idx in range(len(mentions_per_sent) - 1):
        current = mentions_per_sent[idx]
        nxt = mentions_per_sent[idx + 1]
        for entity in entities:
            role_a = current.get(entity, "-")
            role_b = nxt.get(entity, "-")
            transitions[entity][f"{role_a}->{role_b}"] += 1

    return transitions


def transition_vector(transitions):
    role_pairs = [f"{a}->{b}" for a in ROLES for b in ROLES]
    vector = np.zeros(len(role_pairs), dtype=float)
    index = {pair: i for i, pair in enumerate(role_pairs)}

    for entity, counts in transitions.items():
        for pair, cnt in counts.items():
            vector[index[pair]] += cnt

    return vector, role_pairs


def cosine_similarity(vec_a, vec_b):
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def write_transition_csv(path, transitions, role_pairs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        header = ["entity"] + role_pairs
        f.write(",".join(header) + "\n")
        for entity, counts in sorted(transitions.items()):
            row = [entity]
            for pair in role_pairs:
                row.append(str(counts.get(pair, 0)))
            f.write(",".join(row) + "\n")


def load_spacy_model(model_name):
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' not found. "
            f"Install it or choose another with --model. Error: {exc}"
        ) from exc


def main():
    parser = argparse.ArgumentParser(
        description="Compute entity transition vectors and cosine similarity."
    )
    parser.add_argument("--original", type=str, required=True, help="Original story file.")
    parser.add_argument("--transformed", type=str, required=True, help="Transformed story file.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_analysis/entity_transitions",
        help="Directory to write transition CSVs.",
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
        help="Use a shared entity list across inputs for aligned transitions.",
    )

    args = parser.parse_args()

    nlp = load_spacy_model(args.model)

    original_text = Path(args.original).read_text(encoding="utf-8")
    transformed_text = Path(args.transformed).read_text(encoding="utf-8")

    (
        _orig_sents,
        orig_mentions,
        orig_entities,
        orig_counts,
    ) = build_entity_roles_for_text(
        nlp,
        original_text,
        include_pronouns=args.include_pronouns,
        min_count=args.min_count,
        max_entities=args.max_entities,
    )
    (
        _trans_sents,
        trans_mentions,
        trans_entities,
        trans_counts,
    ) = build_entity_roles_for_text(
        nlp,
        transformed_text,
        include_pronouns=args.include_pronouns,
        min_count=args.min_count,
        max_entities=args.max_entities,
    )

    if args.shared_entities:
        merged_counts = defaultdict(int)
        for entity, cnt in orig_counts.items():
            merged_counts[entity] += cnt
        for entity, cnt in trans_counts.items():
            merged_counts[entity] += cnt
        shared_entities = select_entities(
            merged_counts, min_count=args.min_count, max_entities=args.max_entities
        )
        orig_entities = shared_entities
        trans_entities = shared_entities

    orig_transitions = compute_entity_transitions(orig_mentions, orig_entities)
    trans_transitions = compute_entity_transitions(trans_mentions, trans_entities)

    orig_vector, role_pairs = transition_vector(orig_transitions)
    trans_vector, _ = transition_vector(trans_transitions)

    similarity = cosine_similarity(orig_vector, trans_vector)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_transition_csv(output_dir / "original_transitions.csv", orig_transitions, role_pairs)
    write_transition_csv(
        output_dir / "transformed_transitions.csv", trans_transitions, role_pairs
    )

    summary_path = output_dir / "transition_similarity.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"cosine_similarity={similarity:.6f}\n")

    logger.info(f"Cosine similarity: {similarity:.6f}")
    logger.info(f"Wrote {summary_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
