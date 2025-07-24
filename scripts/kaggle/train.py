"""
Simplified training script for playlist recommendation.

Uses standard PyTorch Geometric patterns for heterogeneous graph learning.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, Linear
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlaylistTrackRecommender(nn.Module):
    """Track recommender for playlists using heterogeneous GNN."""

    def __init__(
        self,
        playlist_dim=1,  # Playlist features: normalized track count
        track_dim=11,  # Track features: 6 audio + 5 musical (tempo, key, etc.)
        artist_dim=1,  # Artist features: normalized track count
        album_dim=1,  # Album features: normalized track count
        hidden_channels=64,  # Hidden embedding dimension for all node types
        heads=4,  # Number of attention heads in GAT layers
    ):
        super().__init__()

        self.hidden_channels = hidden_channels  # Store for forward pass reference

        # Input projections - transform variable-sized features to fixed hidden dimension
        self.playlist_proj = Linear(playlist_dim, hidden_channels)  # 1D → 64D
        self.track_proj = Linear(track_dim, hidden_channels)  # 11D → 64D
        self.artist_proj = Linear(artist_dim, hidden_channels)  # 1D → 64D
        self.album_proj = Linear(album_dim, hidden_channels)  # 1D → 64D

        # GNN layers transform: raw 64D → diverse 256D → refined 64D
        # First GNN layer - uses 4 attention heads to learn diverse patterns (64D → 256D)
        self.conv1 = HeteroConv(
            {
                # Playlists aggregate info from their tracks to understand content
                ("playlist", "contains", "track"): GATConv(
                    hidden_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                # Tracks aggregate info from playlists they appear in for context
                ("track", "in_playlist", "playlist"): GATConv(
                    hidden_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                # Tracks aggregate info from their artists for style understanding
                ("track", "by", "artist"): GATConv(
                    hidden_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                # Artists aggregate info from their tracks for representation
                ("artist", "created", "track"): GATConv(
                    hidden_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                # Tracks aggregate info from their albums for cohesion
                ("track", "from_album", "album"): GATConv(
                    hidden_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
                # Albums aggregate info from their tracks for characteristics
                ("album", "contains", "track"): GATConv(
                    hidden_channels, hidden_channels, heads=heads, add_self_loops=False
                ),
            },
            aggr="sum",  # Sum messages from different edge types
        )

        # Second GNN layer - combines multi-head features into final representation (256D → 64D)
        self.conv2 = HeteroConv(
            {
                ("playlist", "contains", "track"): GATConv(
                    hidden_channels * heads, hidden_channels, heads=1, add_self_loops=False
                ),
                ("track", "in_playlist", "playlist"): GATConv(
                    hidden_channels * heads, hidden_channels, heads=1, add_self_loops=False
                ),
                ("track", "by", "artist"): GATConv(
                    hidden_channels * heads, hidden_channels, heads=1, add_self_loops=False
                ),
                ("artist", "created", "track"): GATConv(
                    hidden_channels * heads, hidden_channels, heads=1, add_self_loops=False
                ),
                ("track", "from_album", "album"): GATConv(
                    hidden_channels * heads, hidden_channels, heads=1, add_self_loops=False
                ),
                ("album", "contains", "track"): GATConv(
                    hidden_channels * heads, hidden_channels, heads=1, add_self_loops=False
                ),
            },
            aggr="sum",  # Combine refined messages
        )

        # Final projections for recommendation scoring (only playlist & track needed)
        self.playlist_out = Linear(hidden_channels, hidden_channels)  # Playlist embeddings for scoring
        self.track_out = Linear(hidden_channels, hidden_channels)  # Track embeddings for scoring

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass through the heterogeneous GNN.

        Args:
            x_dict: Dict of node features {node_type: tensor of shape [num_nodes, feature_dim]}
                - "playlist": [num_playlists, 1] 
                - "track": [num_tracks, 11]
                - "artist": [num_artists, 1]
                - "album": [num_albums, 1]
            edge_index_dict: Dict of edges {(src_type, edge_type, dst_type): [2, num_edges]}
                Each edge_index is a tensor where row 0 = source nodes, row 1 = destination nodes

        Returns:
            Dict of refined node embeddings {node_type: tensor of shape [num_nodes, 64]}
            All node types have 64-dimensional graph-aware representations
        """
        # Complete forward flow through the network:
        # 1. Input Projections: Raw features → 64D initial embeddings
        # 2. Conv1 (Layer 1): 64D → 256D using 4 attention heads (diverse patterns)
        # 3. Activation: ReLU + Dropout for non-linearity and regularization
        # 4. Conv2 (Layer 2): 256D → 64D using 1 attention head (consolidation)
        # 5. Output Projections: Final 64D embeddings for playlist-track scoring

        # Example with real data:
        # Input: playlist node 0 has feature [0.693] (5 tracks, log-normalized)
        # After projection: 64D vector with learned values
        # After conv1: 256D vector encoding patterns from connected tracks
        # After conv2: 64D refined embedding ready for similarity scoring
        # Result: Can compute dot product with any track embedding for recommendation

        # Project inputs
        x_dict = {
            "playlist": self.playlist_proj(x_dict["playlist"]),
            "track": self.track_proj(x_dict["track"]),
            "artist": self.artist_proj(x_dict["artist"]),
            "album": self.album_proj(x_dict["album"]),
        }

        # First conv layer
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.1, training=self.training) for key, x in x_dict.items()}

        # Second conv layer
        x_dict = self.conv2(x_dict, edge_index_dict)

        # Output projections
        x_dict["playlist"] = self.playlist_out(x_dict["playlist"])
        x_dict["track"] = self.track_out(x_dict["track"])

        return x_dict


def split_edges(graph, val_ratio=0.1, test_ratio=0.1):
    """
    Split playlist-track edges into train/val/test for link prediction evaluation.

    This is crucial for recommendation evaluation - we hide some playlist-track 
    connections and test if the model can predict them. All nodes remain in the graph;
    only edges are split. This simulates real-world scenarios where we know some tracks
    in a playlist and want to recommend others.

    Example: Playlist has tracks [A,B,C,D,E]
    - Train sees: [A,B,C] (learns patterns)
    - Val sees: [D] (picks best model)  
    - Test sees: [E] (final evaluation)
    - Model tries to predict D and E based on A,B,C

    Note: Only playlist-track edges are split. Artist/album edges remain intact
    as they represent fixed metadata relationships.
    """
    edge_index = graph["playlist", "contains", "track"].edge_index
    num_edges = edge_index.shape[1]

    # Random permutation
    perm = torch.randperm(num_edges)

    # Calculate split sizes
    val_size = int(num_edges * val_ratio)
    test_size = int(num_edges * test_ratio)
    train_size = num_edges - val_size - test_size

    # Split indices
    train_idx = perm[:train_size]
    val_idx = perm[train_size : train_size + val_size]
    test_idx = perm[train_size + val_size :]

    # Create edge sets
    train_edges = edge_index[:, train_idx]
    val_edges = edge_index[:, val_idx]
    test_edges = edge_index[:, test_idx]

    logger.info(f"Edge splits - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")

    return train_edges, val_edges, test_edges


def compute_bpr_loss(scores, num_neg_samples=5):
    """
    Compute Bayesian Personalized Ranking (BPR) loss for recommendation training.

    BPR is THE standard loss for implicit feedback recommendation. Core idea:
    For each playlist, tracks that belong should score higher than tracks that don't.
    This teaches the model to rank items, not just classify them.

    Args:
        scores: Tensor of shape [num_playlists, num_tracks] with similarity scores
        num_neg_samples: Number of negative tracks to sample per positive (default: 5)

    Returns:
        Scalar loss value - lower means better ranking

    Example:
        If playlist P1 contains track T1 but not T2,T3:
        BPR ensures score(P1,T1) > score(P1,T2) and score(P1,T1) > score(P1,T3)

    Note: This loss function is critical - it defines what "good recommendations" means.
    More negative samples = stronger training signal but slower computation.
    """
    batch_size = scores.shape[0]
    num_items = scores.shape[1]

    # If batch is smaller than requested, adjust
    if batch_size != num_items:
        # This means we don't have a square matrix - need to handle differently
        # For now, just use random negative sampling
        pos_scores = []
        neg_scores = []

        for i in range(min(batch_size, num_items)):
            pos_scores.append(scores[i, i])
            # Sample negative items
            neg_idx = torch.randint(0, num_items, (num_neg_samples,))
            neg_scores.append(scores[i, neg_idx])

        if len(pos_scores) == 0:
            return torch.tensor(0.0, requires_grad=True)

        pos_scores = torch.stack(pos_scores)
        neg_scores = torch.stack(neg_scores)
    else:
        # Square matrix case
        pos_scores = scores.diag()

        # Sample negative indices
        neg_indices = []
        for i in range(batch_size):
            # Sample without replacement, avoiding the positive index
            all_indices = torch.arange(num_items)
            mask = torch.ones(num_items, dtype=torch.bool)
            mask[i] = False
            valid_indices = all_indices[mask]

            if len(valid_indices) >= num_neg_samples:
                neg_idx = valid_indices[torch.randperm(len(valid_indices))[:num_neg_samples]]
            else:
                neg_idx = valid_indices

            neg_indices.append(neg_idx)

        # Pad if necessary
        max_len = max(len(idx) for idx in neg_indices)
        padded_indices = []
        for idx in neg_indices:
            if len(idx) < max_len:
                # Pad with the first index
                padded = torch.cat([idx, idx[:1].repeat(max_len - len(idx))])
                padded_indices.append(padded)
            else:
                padded_indices.append(idx)

        neg_indices = torch.stack(padded_indices)
        neg_scores = scores.gather(1, neg_indices)

    # BPR loss
    loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

    return loss


def evaluate_recall(model, graph, x_dict, edge_index_dict, eval_edges, k=10):
    """
    Evaluate recall@k metric on held-out playlist-track edges.

    Recall@k measures how many of the true tracks appear in the top-k recommendations.
    This is THE key metric for recommendation quality - it directly measures if we can
    recover hidden playlist tracks.

    Args:
        model: Trained PlaylistTrackRecommender model
        graph: Full graph (used for node counts)
        x_dict: Node features dictionary
        edge_index_dict: All edge indices (training edges only)
        eval_edges: Held-out edges to evaluate (validation or test)
        k: Number of top recommendations to consider (default: 10)

    Returns:
        Average recall@k across all playlists with held-out tracks

    Example:
        Playlist P1 has hidden tracks: [T1, T2, T3] (in eval_edges)
        Model recommends top-5: [T7, T1, T9, T3, T10]
        Hits: 2 (found T1 and T3)
        Recall@5 = 2/3 = 0.67

    Note: Perfect recall@k=1.0 means all hidden tracks were found in top-k.
    Typical good values: Recall@10 ≈ 0.3-0.5 for real datasets.
    """
    model.eval()

    with torch.no_grad():
        # Get embeddings
        out = model(x_dict, edge_index_dict)
        playlist_emb = out["playlist"]
        track_emb = out["track"]

        # Group edges by playlist
        playlist_tracks = {}
        for i in range(eval_edges.shape[1]):
            p, t = eval_edges[:, i].tolist()
            if p not in playlist_tracks:
                playlist_tracks[p] = []
            playlist_tracks[p].append(t)

        # Compute recall for each playlist
        recalls = []
        for playlist_idx, true_tracks in playlist_tracks.items():
            # Get playlist embedding
            p_emb = playlist_emb[playlist_idx]

            # Compute scores for all tracks
            scores = torch.matmul(track_emb, p_emb)

            # Get top-k predictions
            _, top_k_idx = torch.topk(scores, k=min(k, len(scores)))
            top_k_idx = top_k_idx.cpu().tolist()

            # Calculate recall
            hits = len(set(true_tracks) & set(top_k_idx))
            recall = hits / min(len(true_tracks), k)
            recalls.append(recall)

        return sum(recalls) / len(recalls) if recalls else 0.0


def train_one_epoch(model, optimizer, train_edges, x_dict, edge_index_dict,
                    batch_size, num_track_nodes, epoch, num_epochs):
    """
    Train model for one epoch using mini-batch BPR loss.

    Args:
        model: PlaylistTrackRecommender model
        optimizer: PyTorch optimizer
        train_edges: Training edges tensor
        x_dict: Node features dictionary
        edge_index_dict: Edge indices dictionary
        batch_size: Number of edges per batch
        num_track_nodes: Total number of track nodes (for negative sampling)
        epoch: Current epoch number (for progress bar)
        num_epochs: Total epochs (for progress bar)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_edges = train_edges.shape[1]
    num_batches = max(1, num_edges // batch_size)

    # Mini-batch training on edges
    for _ in tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
        # Sample batch of edges randomly
        batch_idx = torch.randint(0, num_edges, (batch_size,))
        batch_edges = train_edges[:, batch_idx]

        # Forward pass
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict)

        # Get embeddings for edges in batch
        playlist_emb = out["playlist"][batch_edges[0]]
        track_emb = out["track"][batch_edges[1]]

        # Positive scores (from actual edges)
        pos_scores = (playlist_emb * track_emb).sum(dim=1)

        # Negative sampling - 5 random tracks per positive
        num_neg = 5
        neg_track_idx = torch.randint(0, num_track_nodes, (batch_size, num_neg))
        neg_track_emb = out["track"][neg_track_idx]

        # Negative scores using batch matrix multiplication
        neg_scores = torch.bmm(neg_track_emb, playlist_emb.unsqueeze(-1)).squeeze(-1)

        # BPR loss - positives should score higher than negatives
        loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


def train(graph, model, train_edges, val_edges, num_epochs=30, lr=0.01, batch_size=128,
          patience=3, val_frequency=3):
    """
    Train the model with early stopping and learning rate scheduling.

    Training flow:
    1. Setup: Create optimizer, scheduler, and training graph (without val/test edges)
    2. For each epoch:
       a) Mini-batch training:
          - Sample batch_size edges randomly
          - Compute embeddings via forward pass
          - Calculate positive scores for true playlist-track pairs
          - Sample negative tracks and compute negative scores
          - Compute BPR loss (positives should score > negatives)
          - Backpropagate and update weights
       b) Validation (every val_frequency epochs):
          - Evaluate recall@10 on held-out edges
          - Adjust learning rate if performance plateaus
          - Save model if best so far
          - Check early stopping condition
    3. Restore best model before returning

    Key improvements over basic training:
    - Early stopping: Prevents overfitting by stopping when val performance plateaus
    - LR scheduling: Reduces learning rate when stuck for better convergence
    - Best model tracking: Returns best model, not just final one
    - Frequent validation: Catches best model more precisely

    Args:
        graph: Full graph with all nodes and edges
        model: PlaylistTrackRecommender to train
        train_edges: Training playlist-track edges
        val_edges: Validation edges for evaluation
        num_epochs: Maximum epochs to train (default: 30)
        lr: Initial learning rate (default: 0.01)
        batch_size: Edges per mini-batch (default: 128)
        patience: Epochs to wait before early stopping (default: 3)
        val_frequency: Validate every N epochs (default: 3)

    Returns:
        history: Dict with train_loss and val_recall lists
        best_val_recall: Best validation recall@10 achieved
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-5
    )

    # Prepare training graph (remove val/test edges)
    train_graph = graph.clone()
    train_graph["playlist", "contains", "track"].edge_index = train_edges
    train_graph["track", "in_playlist", "playlist"].edge_index = train_edges.flip(0)

    # Get node features and edge indices
    x_dict = {node_type: train_graph[node_type].x for node_type in train_graph.node_types}
    edge_index_dict = {
        edge_type: train_graph[edge_type].edge_index for edge_type in train_graph.edge_types
    }

    history = {"train_loss": [], "val_recall": []}
    best_val_recall = 0
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Train for one epoch
        avg_loss = train_one_epoch(
            model, optimizer, train_edges, x_dict, edge_index_dict,
            batch_size, graph["track"].num_nodes, epoch, num_epochs
        )
        history["train_loss"].append(avg_loss)

        # Validation
        if (epoch + 1) % val_frequency == 0:
            val_recall = evaluate_recall(model, train_graph, x_dict, edge_index_dict, val_edges)
            history["val_recall"].append(val_recall)

            # Learning rate scheduling
            scheduler.step(val_recall)
            current_lr = optimizer.param_groups[0]['lr']

            if val_recall > best_val_recall:
                best_val_recall = val_recall
                best_model_state = model.state_dict().copy()
                epochs_without_improvement = 0
                logger.info(f"📈 New best model! Recall@10={val_recall:.4f}")
            else:
                epochs_without_improvement += 1

            logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Val Recall@10={val_recall:.4f}, LR={current_lr:.6f}")

            # Early stopping check
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model with recall@10={best_val_recall:.4f}")

    return history, best_val_recall


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train playlist recommender")
    parser.add_argument(
        "--graph",
        type=str,
        default="data/kaggle/playlist_graph.pt",
        help="Path to graph file",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/kaggle",
        help="Output directory",
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)

    # Load graph
    logger.info(f"Loading graph from {args.graph}")
    graph = torch.load(args.graph, weights_only=False)

    # Split edges
    train_edges, val_edges, test_edges = split_edges(graph)

    # Create model
    model = PlaylistTrackRecommender(
        playlist_dim=graph["playlist"].x.shape[1],
        track_dim=graph["track"].x.shape[1],
        artist_dim=graph["artist"].x.shape[1],
        album_dim=graph["album"].x.shape[1],
        hidden_channels=args.hidden_dim,
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,}")

    # Train model
    logger.info("Starting training...")
    history, best_val_recall = train(
        graph,
        model,
        train_edges,
        val_edges,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_graph = graph.clone()
    test_graph["playlist", "contains", "track"].edge_index = train_edges
    test_graph["track", "in_playlist", "playlist"].edge_index = train_edges.flip(0)

    x_dict = {node_type: test_graph[node_type].x for node_type in test_graph.node_types}
    edge_index_dict = {
        edge_type: test_graph[edge_type].edge_index for edge_type in test_graph.edge_types
    }

    test_recall = evaluate_recall(model, test_graph, x_dict, edge_index_dict, test_edges)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save(model.state_dict(), output_dir / "model.pt")

    # Save metrics
    metrics = {
        "best_val_recall": float(best_val_recall),
        "test_recall": float(test_recall),
        "history": history,
        "config": vars(args),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print summary
    print("\nTraining Complete!")
    print(f"Best Val Recall@10: {best_val_recall:.4f}")
    print(f"Test Recall@10: {test_recall:.4f}")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
