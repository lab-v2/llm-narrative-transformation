"""
Rule Learner Module

Learns PyReason rules from LLM survey results.
Creates rules mapping feature scores to corpus similarity.

The rules follow the format:
corpus(X, feature):[confidence,confidence] <-1
    individualistic_feature(X, feature):[score,score],
    story_name(X):[1,1],
    feature_predicate(feature):[1,1]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ==========================================================
# Configuration
# ==========================================================

# Score ranges for rule bodies
# Each range represents a bucket of normalized scores
BODY_RANGES = [
    [0.0, 0.0],  # Rating 5
    [0.25, 0.25],  # Rating 4
    [0.5, 0.5],  # Rating 3
    [0.75, 0.75],  # Rating 2
    [1.0, 1.0]  # Rating 1
]

# Minimum confidence to create a rule (proportion of stories)
MIN_CONFIDENCE = 0.0

# Minimum number of stories supporting a rule
MIN_SUPPORT = 0


# ==========================================================
# Rating Normalization
# ==========================================================

def normalize_rating(rating: int) -> float:
    """
    Convert rating (1-5) to normalized score (0.0-1.0).

    Mapping:
        1 (entirely individual)  → 1.0
        2 (primarily individual) → 0.75
        3 (balanced)             → 0.5
        4 (primarily group)      → 0.25
        5 (entirely group)       → 0.0

    Args:
        rating: Rating value (1-5)

    Returns:
        Normalized score (0.0-1.0)
    """
    mapping = {
        1: 1.0,
        2: 0.75,
        3: 0.5,
        4: 0.25,
        5: 0.0
    }
    return mapping.get(rating, 0.5)  # Default to 0.5 if unknown


def feature_to_predicate(feature: str) -> str:
    """
    Convert feature name to predicate format.

    Replaces spaces and hyphens with underscores, converts to lowercase.

    Args:
        feature: Feature name (e.g., "Internal Goals")

    Returns:
        Predicate format (e.g., "internal_goals")
    """
    return feature.replace(' ', '_').replace('-', '_').replace('‑', '_').lower()


# ==========================================================
# Survey Loading
# ==========================================================

def load_survey_results(survey_dir: Path) -> Tuple[List[Dict], List[str]]:
    """
    Load all survey results from directory.

    Args:
        survey_dir: Path to survey results directory

    Returns:
        Tuple of (list of survey results, list of story names)
    """
    if not survey_dir.exists():
        raise FileNotFoundError(f"Survey directory not found: {survey_dir}")

    survey_files = sorted(survey_dir.glob("*.json"))

    if len(survey_files) == 0:
        raise ValueError(f"No survey JSON files found in {survey_dir}")

    logger.info(f"Found {len(survey_files)} survey result files")

    survey_results = []
    story_names = []

    for survey_file in survey_files:
        try:
            with open(survey_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
            survey_results.append(result)
            story_names.append(result["story_name"])
        except Exception as e:
            logger.error(f"Failed to load {survey_file.name}: {e}")
            continue

    logger.info(f"Successfully loaded {len(survey_results)} survey results")
    return survey_results, story_names


# ==========================================================
# Feature Score Extraction
# ==========================================================

def extract_feature_scores(survey_results: List[Dict]) -> Tuple[Dict, set, set]:
    """
    Extract feature scores from survey results.

    Args:
        survey_results: List of survey result dictionaries

    Returns:
        Tuple of:
            - feature_scores: {story_name: {feature: normalized_score}}
            - stories: Set of story names
            - features: Set of feature names (in predicate format)
    """
    feature_scores = defaultdict(dict)
    stories = set()
    features = set()

    for result in survey_results:
        story_name = result["story_name"]
        stories.add(story_name)

        for qa in result.get("questions_and_answers", []):
            # Get component name and convert to predicate format
            component = qa.get("component", "")
            feature = feature_to_predicate(component)

            # Get rating and normalize
            rating = qa.get("rating")
            if rating is None:
                logger.warning(
                    f"No rating found for {component} in story {story_name}, skipping"
                )
                continue

            score = normalize_rating(rating)

            feature_scores[story_name][feature] = score
            features.add(feature)

    logger.info(f"Extracted scores for {len(stories)} stories and {len(features)} features")

    return feature_scores, stories, features


# ==========================================================
# Rule Learning
# ==========================================================

def learn_rules_from_scores(
        feature_scores: Dict,
        stories: set,
        features: set,
        body_ranges: List[List[float]] = BODY_RANGES,
        min_confidence: float = MIN_CONFIDENCE,
        min_support: int = MIN_SUPPORT
) -> List[str]:
    """
    Learn PyReason rules from feature scores.

    Args:
        feature_scores: {story_name: {feature: score}}
        stories: Set of story names
        features: Set of feature names
        body_ranges: List of [min, max] ranges for body predicates
        min_confidence: Minimum confidence to create a rule
        min_support: Minimum number of stories supporting a rule

    Returns:
        List of rule strings
    """
    learned_rules = []
    total_stories = len(stories)

    logger.info(f"Learning rules for {len(features)} features...")

    for feature in sorted(features):
        logger.debug(f"Processing feature: {feature}")

        for range_min, range_max in body_ranges:
            # Count stories with feature score in [range_min, range_max]
            stories_in_range = 0

            for story in stories:
                if story in feature_scores and feature in feature_scores[story]:
                    score = feature_scores[story][feature]
                    if range_min <= score <= range_max:
                        stories_in_range += 1

            # Calculate confidence
            confidence = stories_in_range / total_stories if total_stories > 0 else 0.0

            # Only create rule if meets criteria
            if confidence >= min_confidence and stories_in_range >= min_support:
                # Create rule text
                rule = (
                    f"corpus(X, {feature}):[{confidence:.2f},{confidence:.2f}] <-1 "
                    f"individualistic_feature(X, {feature}):[{range_min},{range_max}], "
                    f"story_name(X):[1,1], "
                    f"{feature}({feature}):[1,1]"
                )

                learned_rules.append(rule)

                logger.debug(
                    f"  Range [{range_min},{range_max}]: "
                    f"{stories_in_range}/{total_stories} stories → "
                    f"confidence {confidence:.2f}"
                )
            else:
                logger.debug(
                    f"  Range [{range_min},{range_max}]: "
                    f"{stories_in_range}/{total_stories} stories → "
                    f"SKIPPED (conf={confidence:.2f}, support={stories_in_range})"
                )

    logger.info(f"Learned {len(learned_rules)} rules")
    return learned_rules


# ==========================================================
# Rule Saving
# ==========================================================

def save_rules(rules: List[str], output_file: Path):
    """
    Save learned rules to text file.

    Args:
        rules: List of rule strings
        output_file: Path to output file
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for rule in rules:
            f.write(rule + '\n')

    logger.info(f"Saved {len(rules)} rules to {output_file}")


# ==========================================================
# Main Function
# ==========================================================

def learn_rules(
        survey_results_dir: str,
        problem_type: str,
        model: str,
        output_dir: str
) -> str:
    """
    Learn PyReason rules from survey results.

    Args:
        survey_results_dir: Path to directory containing survey JSON files
        problem_type: "forward" or "inverse"
        model: Model name (for directory structure)
        output_dir: Base output directory

    Returns:
        Path to saved rules file
    """
    logger.info("=" * 60)
    logger.info("RULE LEARNING")
    logger.info("=" * 60)

    survey_dir = Path(survey_results_dir)

    # Check for failed stories
    failed_stories_file = survey_dir.parent / "failed_stories.json"
    if failed_stories_file.exists():
        with open(failed_stories_file, 'r') as f:
            failed_data = json.load(f)
            failed_stories = failed_data.get("failed_stories", [])
            if failed_stories:
                logger.warning(
                    f"⚠️  {len(failed_stories)} stories failed during survey and "
                    f"will be excluded from rule learning:"
                )
                for story in failed_stories:
                    logger.warning(f"  - {story}")

    # Load survey results
    logger.info(f"Loading survey results from: {survey_dir}")
    survey_results, story_names = load_survey_results(survey_dir)

    # Extract feature scores
    logger.info("Extracting feature scores...")
    feature_scores, stories, features = extract_feature_scores(survey_results)

    # Learn rules
    logger.info("Learning PyReason rules...")
    logger.info(f"Configuration:")
    logger.info(f"  - Body ranges: {BODY_RANGES}")
    logger.info(f"  - Min confidence: {MIN_CONFIDENCE}")
    logger.info(f"  - Min support: {MIN_SUPPORT}")

    rules = learn_rules_from_scores(
        feature_scores=feature_scores,
        stories=stories,
        features=features,
        body_ranges=BODY_RANGES,
        min_confidence=MIN_CONFIDENCE,
        min_support=MIN_SUPPORT
    )

    # Sanitize model name for directory
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Save rules
    rules_dir = Path(output_dir) / sanitized_model / problem_type / "learned_rules"
    rules_file = rules_dir / "pyreason_rules.txt"

    save_rules(rules, rules_file)

    # Summary
    logger.info("=" * 60)
    logger.info("RULE LEARNING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Stories processed: {len(stories)}")
    logger.info(f"Features: {len(features)}")
    logger.info(f"Rules learned: {len(rules)}")
    logger.info(f"Rules saved to: {rules_file}")
    logger.info("=" * 60)

    # Show sample rules
    if rules:
        logger.info("\nSample rules:")
        for i, rule in enumerate(rules[:5], 1):
            logger.info(f"  {i}. {rule}")
        if len(rules) > 5:
            logger.info(f"  ... and {len(rules) - 5} more")

    return str(rules_file)


# ==========================================================
# For Direct Execution (Testing)
# ==========================================================

if __name__ == "__main__":
    # Example usage for testing
    import sys

    if len(sys.argv) < 4:
        print("Usage: python rule_learner.py <survey_dir> <problem_type> <model>")
        print("Example: python rule_learner.py output/phase1/gpt-4o/forward/survey_results forward gpt-4o")
        sys.exit(1)

    survey_dir = sys.argv[1]
    problem_type = sys.argv[2]
    model = sys.argv[3]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    rules_file = learn_rules(
        survey_results_dir=survey_dir,
        problem_type=problem_type,
        model=model,
        output_dir="output/phase1"
    )

    print(f"\n✓ Rules saved to: {rules_file}")