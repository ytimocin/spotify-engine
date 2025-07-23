"""
Utility functions for sparse embedding operations.

Split from models.py to reduce complexity.
"""

from typing import Dict

import torch
from torch_geometric.data import HeteroData


def get_initial_node_embeddings(
    model, node_indices: Dict[str, torch.Tensor], graph: HeteroData
) -> Dict[str, torch.Tensor]:
    """
    Get initial embeddings for specified nodes.

    Args:
        model: The PlaylistGAT model
        node_indices: Dict mapping node types to indices
        graph: Full graph for features

    Returns:
        Dict of initial embeddings by node type
    """
    h_dict = {}

    # Playlist embeddings
    if "playlist" in node_indices:
        idx = node_indices["playlist"]
        emb = model.playlist_embedding(idx)

        if hasattr(graph["playlist"], "x") and graph["playlist"].x is not None:
            feat = model.playlist_feature_proj(graph["playlist"].x[idx])
            h_dict["playlist"] = torch.cat([emb, feat], dim=-1)
        else:
            h_dict["playlist"] = model.playlist_init_proj(emb)

    # Track embeddings
    if "track" in node_indices:
        idx = node_indices["track"]
        if hasattr(graph["track"], "x") and graph["track"].x is not None:
            h_dict["track"] = model.track_init_proj(graph["track"].x[idx])

    # Artist embeddings
    if "artist" in node_indices:
        idx = node_indices["artist"]
        emb = model.artist_embedding(idx)

        if hasattr(graph["artist"], "x") and graph["artist"].x is not None:
            feat = model.artist_feature_proj(graph["artist"].x[idx])
            h_dict["artist"] = torch.cat([emb, feat], dim=-1)
        else:
            h_dict["artist"] = model.artist_init_proj(emb)

    # Genre embeddings
    if "genre" in node_indices:
        idx = node_indices["genre"]
        if hasattr(graph["genre"], "x") and graph["genre"].x is not None:
            h_dict["genre"] = model.genre_init_proj(graph["genre"].x[idx])

    return h_dict


def apply_output_projections(model, h_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Apply output projections to node embeddings.

    Args:
        model: The PlaylistGAT model
        h_dict: Dict of embeddings by node type

    Returns:
        Dict of projected embeddings
    """
    result = {}
    for node_type, embeddings in h_dict.items():
        if node_type in model.output_projection:
            result[node_type] = model.output_projection[node_type](embeddings)
        else:
            result[node_type] = embeddings
    return result
