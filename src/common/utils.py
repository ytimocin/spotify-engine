"""Utility functions for the Spotify Engine project."""

import torch


def create_node_indices(graph) -> dict:
    """Create node indices dictionary for graph operations.

    Args:
        graph: PyTorch Geometric HeteroData graph

    Returns:
        Dictionary mapping node types to their indices
    """
    indices = {
        "user": torch.arange(graph["user"].num_nodes, dtype=torch.long),
        "song": torch.arange(graph["song"].num_nodes, dtype=torch.long),
        "artist": torch.arange(graph["artist"].num_nodes, dtype=torch.long),
    }

    # Add genre indices if present in graph
    if "genre" in graph.node_types:
        indices["genre"] = torch.arange(graph["genre"].num_nodes, dtype=torch.long)

    return indices
