"""
Test script for Kaggle playlist-based model.

For now, this just loads and displays the graph structure.
The actual model will be implemented later.
"""

import argparse
import logging

import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test Kaggle playlist model")
    parser.add_argument(
        "--graph", type=str, default="data/kaggle/playlist_graph.pt", help="Path to playlist graph"
    )

    args = parser.parse_args()

    # Load graph
    logger.info(f"Loading graph from {args.graph}")
    graph = torch.load(args.graph, weights_only=False)

    # Display graph info
    print("\nPlaylist Graph Structure:")
    print("-" * 50)
    print(f"Node types: {graph.node_types}")
    print(f"Edge types: {graph.edge_types}")
    print()

    print("Node counts:")
    for node_type in graph.node_types:
        print(f"  {node_type}: {graph[node_type].num_nodes:,}")
    print()

    print("Edge counts:")
    for edge_type in graph.edge_types:
        print(f"  {edge_type}: {graph[edge_type].edge_index.shape[1]:,}")
    print()

    print("Node feature dimensions:")
    for node_type in graph.node_types:
        if hasattr(graph[node_type], "x") and graph[node_type].x is not None:
            print(f"  {node_type}: {graph[node_type].x.shape}")

    print("\nNote: Playlist-based model training will be implemented in the next phase.")


if __name__ == "__main__":
    main()
