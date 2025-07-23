"""
Playlist-based GAT model for Kaggle data.

This model is designed for playlist completion/expansion tasks
rather than session-based next-song prediction.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, LayerNorm


class PlaylistGAT(nn.Module):
    """GAT model for playlist-based recommendations."""

    def __init__(
        self,
        num_playlists: int,
        num_tracks: int,
        num_artists: int,
        num_genres: int,
        playlist_feature_dim: int = 8,
        track_feature_dim: int = 7,
        artist_feature_dim: int = 1,
        genre_feature_dim: int = None,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # If genre_feature_dim not specified, assume one-hot encoding
        if genre_feature_dim is None:
            genre_feature_dim = num_genres

        # Node embeddings and feature projections
        # Playlists have both embeddings and features
        self.playlist_embedding = nn.Embedding(num_playlists, embedding_dim)
        self.playlist_feature_proj = nn.Linear(playlist_feature_dim, embedding_dim)

        # Tracks have features only
        self.track_feature_proj = nn.Linear(track_feature_dim, embedding_dim)

        # Artists have both embeddings and features
        self.artist_embedding = nn.Embedding(num_artists, embedding_dim)
        self.artist_feature_proj = nn.Linear(artist_feature_dim, embedding_dim)

        # Genres have features only (one-hot or learned)
        self.genre_feature_proj = nn.Linear(genre_feature_dim, embedding_dim)

        # Initial projection to hidden dimension
        self.input_projection = nn.ModuleDict(
            {
                "playlist": nn.Linear(embedding_dim * 2, hidden_dim),  # emb + features
                "track": nn.Linear(embedding_dim, hidden_dim),
                "artist": nn.Linear(embedding_dim * 2, hidden_dim),  # emb + features
                "genre": nn.Linear(embedding_dim, hidden_dim),
            }
        )

        # Multi-layer heterogeneous GAT
        self.gat_layers = nn.ModuleList()

        for _ in range(num_layers):
            convs = {}

            # Playlist-Track interactions (bidirectional)
            convs[("playlist", "contains", "track")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )
            convs[("track", "in_playlist", "playlist")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )

            # Track-Artist relationships (bidirectional)
            convs[("track", "by", "artist")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )
            convs[("artist", "created", "track")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )

            # Track-Genre associations (bidirectional)
            convs[("track", "has_genre", "genre")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )
            convs[("genre", "includes_track", "track")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )

            # Artist-Genre associations (bidirectional)
            convs[("artist", "performs_genre", "genre")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )
            convs[("genre", "performed_by", "artist")] = GATConv(
                hidden_dim,
                hidden_dim // heads,
                heads=heads,
                concat=True,
                dropout=dropout,
                add_self_loops=False,
            )

            # Create HeteroConv layer
            self.gat_layers.append(HeteroConv(convs, aggr="sum"))

        # Layer normalization
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleList(
                [LayerNorm(hidden_dim, mode="node") for _ in range(num_layers)]
            )

        # Final output projection
        self.output_projection = nn.ModuleDict(
            {
                "playlist": nn.Linear(hidden_dim, embedding_dim),
                "track": nn.Linear(hidden_dim, embedding_dim),
                "artist": nn.Linear(hidden_dim, embedding_dim),
                "genre": nn.Linear(hidden_dim, embedding_dim),
            }
        )

        # Initialize embeddings
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with Xavier uniform."""
        nn.init.xavier_uniform_(self.playlist_embedding.weight)
        nn.init.xavier_uniform_(self.artist_embedding.weight)

    def forward(
        self, x_dict: Dict[str, torch.Tensor], graph: HeteroData, return_attention: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """
        Forward pass through the heterogeneous graph.

        Args:
            x_dict: Dict containing node indices and features
            graph: HeteroData graph object
            return_attention: Whether to return attention weights

        Returns:
            Updated node embeddings and optionally attention weights
        """
        # Process initial embeddings and features
        h_dict = {}

        # Playlists: combine embeddings and features
        playlist_idx = x_dict["playlist"]
        playlist_emb = self.playlist_embedding(playlist_idx)
        playlist_feat = self.playlist_feature_proj(graph["playlist"].x)
        h_dict["playlist"] = torch.cat([playlist_emb, playlist_feat], dim=-1)

        # Tracks: features only
        h_dict["track"] = self.track_feature_proj(graph["track"].x)

        # Artists: combine embeddings and features
        artist_idx = x_dict["artist"]
        artist_emb = self.artist_embedding(artist_idx)
        artist_feat = self.artist_feature_proj(graph["artist"].x)
        h_dict["artist"] = torch.cat([artist_emb, artist_feat], dim=-1)

        # Genres: features only
        h_dict["genre"] = self.genre_feature_proj(graph["genre"].x)

        # Project to hidden dimension
        for node_type in h_dict:
            h_dict[node_type] = self.input_projection[node_type](h_dict[node_type])

        # Store attention weights if requested
        all_attention_weights = [] if return_attention else None

        # Apply GAT layers
        for i, gat_layer in enumerate(self.gat_layers):
            h_dict_prev = {k: v.clone() for k, v in h_dict.items()}  # For residual

            # Apply heterogeneous GAT
            h_dict = gat_layer(h_dict, graph.edge_index_dict)

            # Apply activation and dropout
            h_dict = {k: F.relu(v) for k, v in h_dict.items()}
            h_dict = {
                k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h_dict.items()
            }

            # Residual connection
            if i > 0:  # Skip residual for first layer
                h_dict = {k: h_dict[k] + h_dict_prev[k] for k in h_dict.keys()}

            # Layer normalization
            if self.use_layer_norm:
                for node_type in h_dict:
                    h_dict[node_type] = self.layer_norms[i](h_dict[node_type])

        # Final output projection
        for node_type in h_dict:
            h_dict[node_type] = self.output_projection[node_type](h_dict[node_type])

        return h_dict, all_attention_weights

    def compute_scores(
        self, playlist_embeds: torch.Tensor, track_embeds: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Compute recommendation scores between playlists and tracks.

        Args:
            playlist_embeds: Playlist embeddings [num_playlists, embedding_dim]
            track_embeds: Track embeddings [num_tracks, embedding_dim]
            temperature: Temperature for score scaling

        Returns:
            Scores matrix [num_playlists, num_tracks]
        """
        # Normalize embeddings
        playlist_embeds = F.normalize(playlist_embeds, p=2, dim=-1)
        track_embeds = F.normalize(track_embeds, p=2, dim=-1)

        # Compute scores (dot product)
        scores = torch.matmul(playlist_embeds, track_embeds.t()) / temperature

        return scores

    def get_playlist_recommendations(
        self,
        playlist_idx: int,
        x_dict: Dict[str, torch.Tensor],
        graph: HeteroData,
        k: int = 10,
        exclude_known: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get track recommendations for a playlist.

        Args:
            playlist_idx: Playlist index
            x_dict: Node features
            graph: HeteroData graph object
            k: Number of recommendations
            exclude_known: Whether to exclude tracks already in playlist

        Returns:
            Top-k track indices and scores
        """
        # Get embeddings
        h_dict, _ = self.forward(x_dict, graph)

        # Get playlist and track embeddings
        playlist_emb = h_dict["playlist"][playlist_idx].unsqueeze(0)
        track_embs = h_dict["track"]

        # Compute scores
        scores = self.compute_scores(playlist_emb, track_embs).squeeze(0)

        # Exclude known tracks if requested
        if exclude_known:
            # Get playlist's existing tracks from graph
            playlist_tracks_mask = torch.zeros_like(scores, dtype=torch.bool)
            if ("playlist", "contains", "track") in graph.edge_types:
                edge_index = graph[("playlist", "contains", "track")].edge_index
                playlist_edges = edge_index[0] == playlist_idx
                known_tracks = edge_index[1][playlist_edges]
                playlist_tracks_mask[known_tracks] = True

            # Set scores of known tracks to -inf
            scores[playlist_tracks_mask] = float("-inf")

        # Get top-k
        top_scores, top_indices = torch.topk(scores, min(k, scores.size(0)))

        return top_indices, top_scores

    def explain_recommendation(
        self,
        playlist_idx: int,
        track_idx: int,
        x_dict: Dict[str, torch.Tensor],
        graph: HeteroData,
    ) -> Dict[str, any]:
        """
        Explain why a track was recommended for a playlist.

        Args:
            playlist_idx: Playlist index
            track_idx: Track index
            x_dict: Node features
            graph: HeteroData graph object

        Returns:
            Dictionary with explanation details
        """
        # Get embeddings
        h_dict, _ = self.forward(x_dict, graph)

        explanation = {
            "playlist_idx": playlist_idx,
            "track_idx": track_idx,
            "score": None,
            "genre_influence": None,
            "artist_influence": None,
            "similar_tracks_in_playlist": None,
        }

        # Compute recommendation score
        playlist_emb = h_dict["playlist"][playlist_idx]
        track_emb = h_dict["track"][track_idx]
        score = F.cosine_similarity(playlist_emb, track_emb, dim=0)
        explanation["score"] = score.item()

        # Analyze genre influence
        track_genre_edges = graph[("track", "has_genre", "genre")].edge_index
        track_genres = track_genre_edges[1][track_genre_edges[0] == track_idx]

        if len(track_genres) > 0:
            # Check if playlist has tracks with similar genres
            playlist_tracks = graph[("playlist", "contains", "track")].edge_index[1][
                graph[("playlist", "contains", "track")].edge_index[0] == playlist_idx
            ]

            playlist_track_genres = set()
            for pt in playlist_tracks:
                pt_genres = track_genre_edges[1][track_genre_edges[0] == pt]
                playlist_track_genres.update(pt_genres.tolist())

            common_genres = set(track_genres.tolist()) & playlist_track_genres
            if common_genres:
                genre_scores = []
                for genre_idx in common_genres:
                    genre_emb = h_dict["genre"][genre_idx]
                    genre_sim = F.cosine_similarity(playlist_emb, genre_emb, dim=0)
                    genre_scores.append({"genre_idx": genre_idx, "similarity": genre_sim.item()})
                explanation["genre_influence"] = sorted(
                    genre_scores, key=lambda x: x["similarity"], reverse=True
                )

        # Analyze artist influence
        track_artist_edges = graph[("track", "by", "artist")].edge_index
        track_artists = track_artist_edges[1][track_artist_edges[0] == track_idx]

        if len(track_artists) > 0:
            artist_idx = track_artists[0].item()
            artist_emb = h_dict["artist"][artist_idx]
            artist_sim = F.cosine_similarity(playlist_emb, artist_emb, dim=0)

            # Check if playlist has other tracks by this artist
            artist_tracks = track_artist_edges[0][track_artist_edges[1] == artist_idx]
            playlist_artist_tracks = set(artist_tracks.tolist()) & set(playlist_tracks.tolist())

            explanation["artist_influence"] = {
                "artist_idx": artist_idx,
                "similarity": artist_sim.item(),
                "tracks_in_playlist": len(playlist_artist_tracks),
            }

        # Find similar tracks already in playlist
        if len(playlist_tracks) > 0:
            track_sims = []
            for pt in playlist_tracks[:10]:  # Check top 10 tracks
                pt_emb = h_dict["track"][pt]
                sim = F.cosine_similarity(track_emb, pt_emb, dim=0)
                track_sims.append({"track_idx": pt.item(), "similarity": sim.item()})
            explanation["similar_tracks_in_playlist"] = sorted(
                track_sims, key=lambda x: x["similarity"], reverse=True
            )[:3]

        return explanation
