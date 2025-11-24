"""
Network-Wide Degree Centrality Computation Script

Computes network-wide degree centrality (Freeman, 1979) for bipartite graphs
representing segment-feature relationships.

Formula: C_G = Σ(d*_G - d_i) / ((N_G - 1)(N_G - 2))

Where:
- d*_G = maximum degree in network
- d_i = degree of node i
- N_G = total number of nodes
- Sum over all nodes

Range: [0, 1]
- 0 = Perfectly decentralized (all nodes equal degree)
- 1 = Perfect star (one central node)

Higher centrality indicates fewer segments cover many features (more centralized).

Output: CSV saved to output_analysis/causal_graphs/{model}/{problem}/
"""

import json
import logging
import argparse
from pathlib import Path
import pandas as pd
import networkx as nx
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==========================================================
# Bipartite Graph Construction
# ==========================================================

def build_bipartite_graph(segments_metadata: dict) -> nx.Graph:
    """
    Build bipartite graph from segments metadata.

    Args:
        segments_metadata: Dict from segments_metadata.json

    Returns:
        NetworkX Graph
    """
    G = nx.Graph()

    story_data = segments_metadata.get('story', {})

    for feature_name, feature_data in story_data.items():
        # Add feature node
        G.add_node(feature_name, bipartite=1, node_type='feature')

        excerpts = feature_data.get('excerpts', [])

        for excerpt in excerpts:
            # Use truncated version as ID
            segment_id = excerpt[:50] + "..." if len(excerpt) > 50 else excerpt

            # Add segment node
            G.add_node(segment_id, bipartite=0, node_type='segment')

            # Add edge
            G.add_edge(segment_id, feature_name)

    return G


# ==========================================================
# Network-Wide Degree Centrality
# ==========================================================

def compute_bipartite_centralities(G: nx.Graph) -> tuple:
    """
    Compute network-wide degree centrality separately for each partition
    of the bipartite graph, plus overall centrality.

    Returns three centrality scores:
    1. Segment centrality (based on segment node degrees only)
    2. Feature centrality (based on feature node degrees only)
    3. Overall centrality (based on all nodes)

    Formula (Freeman, 1979): C = Σ(d* - d_i) / ((N - 1)(N - 2))

    Args:
        G: NetworkX bipartite graph

    Returns:
        Tuple of (segment_centrality, feature_centrality, overall_centrality)
    """
    # Separate nodes by type
    segments = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
    features = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]

    # Get all degrees
    degrees = dict(G.degree())

    # Compute segment centrality (using only segment nodes)
    if len(segments) >= 2:
        segment_degrees = [degrees[s] for s in segments]
        d_star_segments = max(segment_degrees)
        sum_diff_segments = sum(d_star_segments - d for d in segment_degrees)
        N_segments = len(segments)
        denominator_segments = (N_segments - 1) * (N_segments - 2)

        if denominator_segments > 0:
            C_segments = sum_diff_segments / denominator_segments
        else:
            C_segments = 0.0
    else:
        C_segments = 0.0

    # Compute feature centrality (using only feature nodes)
    if len(features) >= 2:
        feature_degrees = [degrees[f] for f in features]
        d_star_features = max(feature_degrees)
        sum_diff_features = sum(d_star_features - d for d in feature_degrees)
        N_features = len(features)
        denominator_features = (N_features - 1) * (N_features - 2)

        if denominator_features > 0:
            C_features = sum_diff_features / denominator_features
        else:
            C_features = 0.0
    else:
        C_features = 0.0

    # Compute overall centrality (all nodes together)
    all_degrees = list(degrees.values())
    if len(all_degrees) >= 2:
        d_star_overall = max(all_degrees)
        sum_diff_overall = sum(d_star_overall - d for d in all_degrees)
        N_total = len(all_degrees)
        denominator_overall = (N_total - 1) * (N_total - 2)

        if denominator_overall > 0:
            C_overall = sum_diff_overall / denominator_overall
        else:
            C_overall = 0.0
    else:
        C_overall = 0.0

    return C_segments, C_features, C_overall


# ==========================================================
# Main Processing
# ==========================================================

def compute_centralities(
    model: str,
    problem_type: str,
    phase2_dir: str = "output/phase2",
    output_dir: str = "output_analysis/causal_graphs"
):
    """
    Compute network-wide centrality for all stories.

    Args:
        model: Model name
        problem_type: "forward" or "inverse"
        phase2_dir: Phase 2 results directory
        output_dir: Output directory for CSV
    """
    logger.info("=" * 60)
    logger.info("COMPUTING NETWORK-WIDE DEGREE CENTRALITY")
    logger.info("=" * 60)
    logger.info(f"Model: {model}")
    logger.info(f"Problem: {problem_type}")

    # Sanitize model name
    sanitized_model = model.replace("/", "-").replace(":", "-")

    # Find all story directories
    results_dir = Path(phase2_dir) / sanitized_model / problem_type
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    story_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(story_dirs)} stories")

    if len(story_dirs) == 0:
        logger.error("No story directories found")
        return

    # Process each story
    results = []

    for story_dir in sorted(story_dirs):
        story_name = story_dir.name
        logger.info(f"\nProcessing: {story_name}")

        # Load segments metadata from iteration_0
        segments_file = story_dir / "iteration_0" / "segments_metadata.json"
        if not segments_file.exists():
            logger.warning(f"  Segments metadata not found, skipping")
            continue

        try:
            with open(segments_file, 'r', encoding='utf-8') as f:
                segments_metadata = json.load(f)

            # Build bipartite graph
            G = build_bipartite_graph(segments_metadata)

            # Compute all three centrality scores
            segment_centrality, feature_centrality, overall_centrality = compute_bipartite_centralities(G)

            # Log details
            segments = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 0]
            features = [n for n, d in G.nodes(data=True) if d.get('bipartite') == 1]
            n_segments = len(segments)
            n_features = len(features)
            n_edges = G.number_of_edges()

            segment_degrees = [G.degree(s) for s in segments]
            feature_degrees = [G.degree(f) for f in features]
            max_segment_degree = max(segment_degrees) if segment_degrees else 0
            max_feature_degree = max(feature_degrees) if feature_degrees else 0

            logger.info(f"  Segments: {n_segments}, Features: {n_features}, Edges: {n_edges}")
            logger.info(f"  Max segment degree: {max_segment_degree}")
            logger.info(f"  Max feature degree: {max_feature_degree}")
            logger.info(f"  Segment centrality: {segment_centrality:.4f}")
            logger.info(f"  Feature centrality: {feature_centrality:.4f}")
            logger.info(f"  Overall centrality: {overall_centrality:.4f}")

            results.append({
                'story_name': story_name,
                'segment_centrality': round(segment_centrality, 4),
                'feature_centrality': round(feature_centrality, 4),
                'overall_centrality': round(overall_centrality, 4),
                'num_segments': n_segments,
                'num_features': n_features,
                'num_edges': n_edges,
                'max_segment_degree': max_segment_degree,
                'max_feature_degree': max_feature_degree
            })

        except Exception as e:
            logger.error(f"  Failed to process: {e}")
            continue

    if not results:
        logger.error("No results to save")
        return

    # Create output directory
    output_path = Path(output_dir) / sanitized_model / problem_type
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df = pd.DataFrame(results)
    csv_file = output_path / "network_centrality.csv"
    df.to_csv(csv_file, index=False)

    logger.info("\n" + "=" * 60)
    logger.info("COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Stories processed: {len(results)}")
    logger.info(f"CSV saved to: {csv_file}")
    logger.info("=" * 60)

    # Summary statistics
    logger.info("\nSegment Centrality Statistics:")
    logger.info(f"  Mean: {df['segment_centrality'].mean():.4f}")
    logger.info(f"  Median: {df['segment_centrality'].median():.4f}")
    logger.info(f"  Min: {df['segment_centrality'].min():.4f}")
    logger.info(f"  Max: {df['segment_centrality'].max():.4f}")
    logger.info(f"  Std: {df['segment_centrality'].std():.4f}")

    logger.info("\nFeature Centrality Statistics:")
    logger.info(f"  Mean: {df['feature_centrality'].mean():.4f}")
    logger.info(f"  Median: {df['feature_centrality'].median():.4f}")
    logger.info(f"  Min: {df['feature_centrality'].min():.4f}")
    logger.info(f"  Max: {df['feature_centrality'].max():.4f}")
    logger.info(f"  Std: {df['feature_centrality'].std():.4f}")

    logger.info("\nOverall Centrality Statistics:")
    logger.info(f"  Mean: {df['overall_centrality'].mean():.4f}")
    logger.info(f"  Median: {df['overall_centrality'].median():.4f}")
    logger.info(f"  Min: {df['overall_centrality'].min():.4f}")
    logger.info(f"  Max: {df['overall_centrality'].max():.4f}")
    logger.info(f"  Std: {df['overall_centrality'].std():.4f}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Compute network-wide degree centrality for bipartite graphs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compute for gpt-4o forward problem
  python plot_scripts/compute_network_centrality.py --model gpt-4o --problem forward
  
  # Compute for Claude inverse problem
  python plot_scripts/compute_network_centrality.py --model claude-sonnet-4-5 --problem inverse
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name (e.g., gpt-4o, claude-sonnet-4-5)'
    )

    parser.add_argument(
        '--problem',
        type=str,
        choices=['forward', 'inverse'],
        required=True,
        help='Problem type: forward or inverse'
    )

    parser.add_argument(
        '--phase2-dir',
        type=str,
        default='output/phase2',
        help='Phase 2 results directory (default: output/phase2)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output_analysis/causal_graphs',
        help='Output directory (default: output_analysis/causal_graphs)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Compute centralities
    try:
        compute_centralities(
            model=args.model,
            problem_type=args.problem,
            phase2_dir=args.phase2_dir,
            output_dir=args.output_dir
        )

        logger.info("\n✓ Computation complete!")

    except Exception as e:
        logger.error(f"Computation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()