"""
Sparse operations for efficient Kaggle model training.

This module provides optimized sparse embedding and computation
functions to speed up training by only computing what's needed.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class SparseEmbeddingModule(nn.Module):
    """Handles sparse embedding operations for playlist GAT model."""

    def __init__(self, model):
        """Initialize with reference to main model."""
        super().__init__()
        self.model = model

    def get_playlist_embeddings(
        self, playlist_indices: torch.Tensor, graph_features: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Get initial embeddings for specific playlists.

        Args:
            playlist_indices: Indices of playlists to embed
            graph_features: Optional pre-extracted features

        Returns:
            Initial playlist embeddings (before GAT)
        """
        # Get learned embeddings
        embeddings = self.model.playlist_embedding(playlist_indices)

        # Project to initial hidden dimension
        h = self.model.playlist_init_proj(embeddings)

        # Add features if available
        if graph_features is not None:
            feat_proj = self.model.playlist_feature_proj(graph_features)
            h = torch.cat([h, feat_proj], dim=-1)

        return h

    def get_track_embeddings(
        self, track_indices: torch.Tensor, graph_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get initial embeddings for specific tracks.

        Args:
            track_indices: Indices of tracks to embed
            graph_features: Track audio features

        Returns:
            Initial track embeddings (before GAT)
        """
        # Tracks only use features, no learned embeddings
        return self.model.track_init_proj(graph_features)

    def compute_final_embeddings(
        self,
        initial_embeddings: Dict[str, torch.Tensor],
        subgraph_edges: Dict[Tuple[str, str, str], torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply GAT layers and output projections to initial embeddings.

        Args:
            initial_embeddings: Dict of initial node embeddings
            subgraph_edges: Optional edge indices for message passing

        Returns:
            Final embeddings after GAT and projection
        """
        h_dict = initial_embeddings

        # If no edges provided, just apply output projection
        if subgraph_edges is None or self.model.num_layers == 0:
            for node_type in h_dict:
                if node_type in self.model.output_projection:
                    h_dict[node_type] = self.model.output_projection[node_type](h_dict[node_type])
            return h_dict

        # Apply GAT layers with subgraph
        for i, gat_layer in enumerate(self.model.gat_layers):
            h_dict_prev = {k: v.clone() for k, v in h_dict.items()}

            # Message passing on subgraph
            h_dict = gat_layer(h_dict, subgraph_edges)

            # Activation and dropout
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}
            h_dict = {
                k: F.dropout(v, p=self.model.dropout, training=self.model.training)
                for k, v in h_dict.items()
            }

            # Residual connection
            if i > 0:
                h_dict = {k: h_dict[k] + h_dict_prev[k] for k in h_dict.keys()}

            # Layer norm
            if self.model.use_layer_norm:
                for node_type in h_dict:
                    h_dict[node_type] = self.model.layer_norms[i](h_dict[node_type])

        # Output projection
        for node_type in h_dict:
            if node_type in self.model.output_projection:
                h_dict[node_type] = self.model.output_projection[node_type](h_dict[node_type])

        return h_dict


class EfficientBatchProcessor:
    """Handles efficient batch processing for training."""

    def __init__(self, graph, model):
        """Initialize with graph and model."""
        self.graph = graph
        self.model = model
        self.sparse_module = SparseEmbeddingModule(model)

        # Pre-extract features for efficiency
        self.playlist_features = graph["playlist"].x if hasattr(graph["playlist"], "x") else None
        self.track_features = graph["track"].x if hasattr(graph["track"], "x") else None

    def prepare_batch_data(
        self,
        batch_playlists: List[int],
        playlist_tracks: Dict[int, List[int]],
        num_neg_samples: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare all data needed for a batch.

        Args:
            batch_playlists: List of playlist indices in batch
            playlist_tracks: Dict mapping playlists to their tracks
            num_neg_samples: Number of negative samples per positive

        Returns:
            Tuple of (playlist_indices, positive_tracks, negative_tracks)
        """
        all_playlists = []
        all_pos_tracks = []
        all_neg_tracks = []

        # Collect all tracks in batch for negative sampling
        batch_tracks = set()
        for pid in batch_playlists:
            batch_tracks.update(playlist_tracks.get(pid, []))

        # All possible tracks for negative sampling
        all_track_ids = set(range(self.graph["track"].num_nodes))

        for pid in batch_playlists:
            pos_tracks = playlist_tracks.get(pid, [])
            if len(pos_tracks) < 2:
                continue

            # Sample some positive tracks
            num_pos = min(5, len(pos_tracks))
            sampled_pos = torch.tensor(torch.randperm(len(pos_tracks))[:num_pos].tolist())
            pos_track_ids = [pos_tracks[i] for i in sampled_pos]

            # Sample negative tracks
            neg_pool = list(all_track_ids - set(pos_tracks))
            neg_track_ids = torch.tensor(
                torch.randperm(len(neg_pool))[: num_neg_samples * num_pos].tolist()
            )
            neg_track_ids = [neg_pool[i] for i in neg_track_ids]

            # Add to batch
            all_playlists.extend([pid] * num_pos)
            all_pos_tracks.extend(pos_track_ids)
            all_neg_tracks.extend(neg_track_ids[: num_pos * num_neg_samples])

        return (
            torch.tensor(all_playlists),
            torch.tensor(all_pos_tracks),
            torch.tensor(all_neg_tracks),
        )

    def compute_batch_embeddings(
        self, playlist_indices: torch.Tensor, track_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute embeddings for batch playlists and tracks.

        Args:
            playlist_indices: Unique playlist indices
            track_indices: Unique track indices

        Returns:
            Tuple of (playlist_embeddings, track_embeddings)
        """
        # Get playlist embeddings
        playlist_feats = None
        if self.playlist_features is not None:
            playlist_feats = self.playlist_features[playlist_indices]
        playlist_emb = self.sparse_module.get_playlist_embeddings(playlist_indices, playlist_feats)

        # Get track embeddings
        track_feats = self.track_features[track_indices]
        track_emb = self.sparse_module.get_track_embeddings(track_indices, track_feats)

        # Apply output projections (skipping GAT for now)
        final_embs = self.sparse_module.compute_final_embeddings(
            {"playlist": playlist_emb, "track": track_emb}
        )

        return final_embs["playlist"], final_embs["track"]
