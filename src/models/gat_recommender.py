"""
Graph Attention Network for music recommendations with explainability.

Simple 1-layer GAT that:
- Learns node embeddings for users and songs
- Uses attention to aggregate neighbor information
- Preserves attention weights for explainability
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv


class GATRecommender(nn.Module):
    """Single-layer GAT for music recommendations."""

    def __init__(
        self,
        num_users: int,
        num_songs: int,
        num_artists: int,
        embedding_dim: int = 32,
        heads: int = 4,
    ):
        """Initialize GAT recommender model.

        Args:
            num_users: Number of user nodes
            num_songs: Number of song nodes
            num_artists: Number of artist nodes
            embedding_dim: Dimension of node embeddings
            heads: Number of attention heads
        """
        super().__init__()

        # Node embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        self.artist_embedding = nn.Embedding(num_artists, embedding_dim)

        # GAT layer
        self.gat = GATConv(
            embedding_dim,
            embedding_dim // heads,  # Output dim per head
            heads=heads,
            concat=True,
            dropout=0.1,
            add_self_loops=False,
        )

        # Final projection
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.song_embedding.weight)
        nn.init.xavier_uniform_(self.artist_embedding.weight)

    def forward(self, x_dict, graph, return_attention=False):
        """
        Forward pass through GAT.

        Args:
            x_dict: Dict of node features by type
            graph: HeteroData graph object
            return_attention: Whether to return attention weights

        Returns:
            Updated node embeddings and optionally attention weights
        """
        # Ensure indices are Long type for embedding layers
        x_dict = {k: v.long() if torch.is_tensor(v) else v for k, v in x_dict.items()}

        # Get embeddings
        x_dict["user"] = self.user_embedding(x_dict["user"])
        x_dict["song"] = self.song_embedding(x_dict["song"])
        x_dict["artist"] = self.artist_embedding(x_dict["artist"])

        # Store original embeddings for skip connection
        x_orig = {k: v.clone() for k, v in x_dict.items()}

        # Initialize attention weights
        attention_weights = None

        # Apply GAT to user-song edges
        if ("user", "listens", "song") in graph.edge_types:
            edge_index = graph[("user", "listens", "song")].edge_index

            # Concatenate user and song features
            x_cat = torch.cat([x_dict["user"], x_dict["song"]], dim=0)

            # Apply GAT
            if return_attention:
                x_out, attention_weights = self.gat(
                    x_cat, edge_index, return_attention_weights=True
                )
            else:
                x_out = self.gat(x_cat, edge_index)
                attention_weights = None

            # Split back into user and song
            x_dict["user"] = x_out[: x_dict["user"].shape[0]]
            x_dict["song"] = x_out[x_dict["user"].shape[0] :]

        # Skip connection and projection
        for node_type in x_dict:
            x_dict[node_type] = self.output_projection(
                F.relu(x_dict[node_type]) + x_orig[node_type]
            )

        if return_attention:
            return x_dict, attention_weights
        return x_dict

    def recommend(self, user_idx: int, x_dict, graph, k: int = 10):
        """
        Get top-k recommendations for a user.

        Args:
            user_idx: User index
            x_dict: Node features
            graph: HeteroData graph object
            k: Number of recommendations

        Returns:
            Top-k song indices and scores
        """
        # Get embeddings
        x_dict, attention = self.forward(x_dict, graph, return_attention=True)

        # Get user and song embeddings
        user_emb = x_dict["user"][user_idx]
        song_embs = x_dict["song"]

        # Compute scores (dot product)
        scores = torch.matmul(song_embs, user_emb)

        # Get top-k
        top_scores, top_indices = torch.topk(scores, k)

        return top_indices, top_scores, attention
