"""
Bipartite Graph Visualization Script

Creates bipartite graphs showing segment-feature relationships.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple
from collections import defaultdict

import networkx as nx
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Plot config
FIGURE_SIZE = (14, 10)
COLOR_SEGMENT = '#3498DB'
COLOR_FEATURE = '#E74C3C'


def build_bipartite_graph(segments_metadata: Dict) -> Tuple:
    G = nx.Graph()
    segment_to_features = defaultdict(set)
    feature_to_segments = defaultdict(set)

    story_data = segments_metadata.get('story', {})

    for feature_name, feature_data in story_data.items():
        G.add_node(feature_name, bipartite=1, node_type='feature')
        excerpts = feature_data.get('excerpts', [])

        for excerpt in excerpts:
            segment_id = excerpt[:50] + "..." if len(excerpt) > 50 else excerpt
            G.add_node(segment_id, bipartite=0, node_type='segment')
            G.add_edge(segment_id, feature_name)
            segment_to_features[segment_id].add(feature_name)
            feature_to_segments[feature_name].add(segment_id)

    analytics = {
        'total_segments': len(segment_to_features),
        'total_features': len(feature_to_segments),
        'total_edges': G.number_of_edges()
    }

    return G, analytics


def visualize_bipartite_graph(G, analytics, story_name, model, problem_type, output_path):
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    segments = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 0}
    features = {n for n, d in G.nodes(data=True) if d.get('bipartite') == 1}
    pos = nx.bipartite_layout(G, segments, align='vertical', scale=2)

    nx.draw_networkx_nodes(G, pos, nodelist=segments, node_color=COLOR_SEGMENT, node_size=800, node_shape='s',
                           alpha=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=features, node_color=COLOR_FEATURE, node_size=1200, node_shape='o',
                           alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='#95A5A6', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)

    problem_desc = "Collectivistic → Individualistic" if problem_type == "forward" else "Individualistic → Collectivistic"
    ax.set_title(f'{story_name}: {problem_desc}\n({model})', fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_bipartite_graphs(model, problem_type, phase2_dir="output/phase2",
                            output_dir="output_analysis/causal_graphs"):
    sanitized_model = model.replace("/", "-").replace(":", "-")
    results_dir = Path(phase2_dir) / sanitized_model / problem_type
    story_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    graph_output_dir = Path(output_dir) / sanitized_model / problem_type

    for story_dir in sorted(story_dirs):
        story_name = story_dir.name
        segments_file = story_dir / "iteration_0" / "segments_metadata.json"
        if not segments_file.exists():
            continue

        with open(segments_file, 'r') as f:
            segments_metadata = json.load(f)

        G, analytics = build_bipartite_graph(segments_metadata)
        plot_file = graph_output_dir / f"{story_name}_bipartite.png"
        visualize_bipartite_graph(G, analytics, story_name, model, problem_type, plot_file)


def main():
    parser = argparse.ArgumentParser(description='Create bipartite graphs')
    parser.add_argument('--model', required=True)
    parser.add_argument('--problem', choices=['forward', 'inverse'], required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    create_bipartite_graphs(args.model, args.problem)


if __name__ == "__main__":
    main()