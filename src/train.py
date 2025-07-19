"""
Train GAT model for music recommendations.

Uses:
- BPR (Bayesian Personalized Ranking) loss
- Negative sampling for implicit feedback
- Recall@10 for evaluation
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from src.models.gat_recommender import GATRecommender


def bpr_loss(pos_scores, neg_scores):
    """Bayesian Personalized Ranking loss."""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def compute_recall_at_k(model, graph, k=10, num_eval_users=100):
    """Compute Recall@K on a sample of users."""
    model.eval()
    recalls = []

    # Sample users for evaluation
    user_indices = torch.randperm(graph["user"].num_nodes)[:num_eval_users]

    # Create node indices dict
    x_dict = {
        "user": torch.arange(graph["user"].num_nodes, dtype=torch.long),
        "song": torch.arange(graph["song"].num_nodes, dtype=torch.long),
        "artist": torch.arange(graph["artist"].num_nodes, dtype=torch.long),
    }

    with torch.no_grad():
        # Get embeddings
        embeddings = model(x_dict, graph)

        for user_idx in user_indices:
            # Get user's true interactions
            edge_index = graph["user", "listens", "song"].edge_index
            user_songs = edge_index[1][edge_index[0] == user_idx]

            if len(user_songs) < 5:  # Skip users with too few interactions
                continue

            # Get recommendations
            user_emb = embeddings["user"][user_idx]
            song_embs = embeddings["song"]
            scores = torch.matmul(song_embs, user_emb)

            # Get top-k
            _, top_k_songs = torch.topk(scores, k)

            # Compute recall
            hits = len(set(top_k_songs.tolist()) & set(user_songs.tolist()))
            recall = hits / min(k, len(user_songs))
            recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0


def train_epoch(model, graph, optimizer, batch_size=512):
    """Train one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    # Get edge data
    edge_index = graph["user", "listens", "song"].edge_index
    num_edges = edge_index.shape[1]

    # Shuffle edges
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm]

    # Create node indices
    x_dict = {
        "user": torch.arange(graph["user"].num_nodes, dtype=torch.long),
        "song": torch.arange(graph["song"].num_nodes, dtype=torch.long),
        "artist": torch.arange(graph["artist"].num_nodes, dtype=torch.long),
    }

    # Process in batches
    for i in range(0, num_edges, batch_size):
        batch_edges = edge_index[:, i : i + batch_size]

        # Get embeddings
        embeddings = model(x_dict, graph)

        # Positive samples
        user_indices = batch_edges[0]
        pos_song_indices = batch_edges[1]

        user_embs = embeddings["user"][user_indices]
        pos_song_embs = embeddings["song"][pos_song_indices]
        pos_scores = (user_embs * pos_song_embs).sum(dim=1)

        # Negative sampling
        neg_song_indices = torch.randint(0, graph["song"].num_nodes, (len(user_indices),))
        neg_song_embs = embeddings["song"][neg_song_indices]
        neg_scores = (user_embs * neg_song_embs).sum(dim=1)

        # BPR loss
        loss = bpr_loss(pos_scores, neg_scores)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def main():
    """Train GAT recommender model."""
    parser = argparse.ArgumentParser(description="Train GAT recommender")
    parser.add_argument("--graph", type=str, default="data/graph.pt", help="Input graph file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--output", type=str, default="models/model.ckpt", help="Output model checkpoint"
    )

    args = parser.parse_args()

    # Load graph
    print(f"Loading graph from: {args.graph}")
    graph = torch.load(args.graph)

    # Create model
    model = GATRecommender(
        num_users=graph["user"].num_nodes,
        num_songs=graph["song"].num_nodes,
        num_artists=graph["artist"].num_nodes,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training
    print(f"\nTraining for {args.epochs} epochs...")
    metrics: Dict[str, List[float]] = {"train_loss": [], "recall@10": []}

    for epoch in range(args.epochs):
        # Train
        loss = train_epoch(model, graph, optimizer, args.batch_size)

        # Evaluate
        recall = compute_recall_at_k(model, graph)

        metrics["train_loss"].append(loss)
        metrics["recall@10"].append(recall)

        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f}, Recall@10: {recall:.4f}")

    # Save model
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_users": graph["user"].num_nodes,
            "num_songs": graph["song"].num_nodes,
            "num_artists": graph["artist"].num_nodes,
            "metrics": metrics,
        },
        output_path,
    )

    print(f"\nSaved model to: {args.output}")

    # Save metrics
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
