"""
Train GAT model for music recommendations with improved pipeline.

Features:
- Train/validation/test splits
- Early stopping with best model checkpointing
- Learning rate scheduling
- BPR (Bayesian Personalized Ranking) loss
- Negative sampling for implicit feedback
- Recall@K and NDCG@K evaluation metrics
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.models.gat_recommender import GATRecommender
from src.utils import create_node_indices


def split_edges(
    edge_index: torch.Tensor, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split edges into train/val/test sets ensuring each user has edges in all splits."""
    num_edges = edge_index.shape[1]

    # Group edges by user
    user_edges: Dict[int, List[int]] = {}
    for i in range(num_edges):
        user = int(edge_index[0, i].item())
        if user not in user_edges:
            user_edges[user] = []
        user_edges[user].append(i)

    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)

    # Split each user's edges
    for _, edges in user_edges.items():
        if len(edges) < 3:  # Skip users with too few edges
            train_mask[edges] = True
            continue

        # Shuffle user's edges
        np.random.shuffle(edges)

        # Calculate split sizes
        n_val = max(1, int(len(edges) * val_ratio))
        n_test = max(1, int(len(edges) * test_ratio))
        n_train = len(edges) - n_val - n_test

        # Assign to splits
        train_mask[edges[:n_train]] = True
        val_mask[edges[n_train : n_train + n_val]] = True
        test_mask[edges[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """Bayesian Personalized Ranking loss."""
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()


def compute_metrics(
    model: torch.nn.Module,
    graph,
    edge_mask: torch.Tensor,
    k: int = 10,
    num_eval_users: int = 100,
) -> Dict[str, float]:
    """Compute Recall@K and NDCG@K on a subset of edges."""
    model.eval()

    # Get edges for evaluation
    edge_index = graph["user", "listens", "song"].edge_index[:, edge_mask]

    # Sample users for evaluation
    unique_users = edge_index[0].unique()
    if len(unique_users) > num_eval_users:
        sample_idx = torch.randperm(len(unique_users))[:num_eval_users]
        eval_users = unique_users[sample_idx]
    else:
        eval_users = unique_users

    # Debug info
    if len(eval_users) == 0:
        print("Warning: No users to evaluate!")
        return {"recall@10": 0.0, "ndcg@10": 0.0}

    # Create node indices dict
    x_dict = create_node_indices(graph)

    recalls = []
    ndcgs = []

    with torch.no_grad():
        # Get embeddings
        embeddings = model(x_dict, graph)

        for user_idx in eval_users:
            # Get user's true interactions from evaluation set
            user_mask = edge_index[0] == user_idx
            user_songs = edge_index[1][user_mask].unique()

            if len(user_songs) < 2:  # Skip users with too few interactions
                continue

            # Get recommendations
            user_emb = embeddings["user"][user_idx]
            song_embs = embeddings["song"]
            scores = torch.matmul(song_embs, user_emb)

            # # For validation/test, we should exclude training items
            # # But for now, let's skip this to debug
            # all_edge_index = graph["user", "listens", "song"].edge_index
            # train_edge_index = all_edge_index[:, ~edge_mask]  # Get training edges
            # train_user_mask = train_edge_index[0] == user_idx
            # if train_user_mask.any():
            #     train_songs = train_edge_index[1][train_user_mask].unique()
            #     scores[train_songs] = -float('inf')

            # Get top-k
            _, top_k_songs = torch.topk(scores, k)

            # Compute recall
            hits = torch.isin(top_k_songs, user_songs).sum().item()
            recall = hits / min(k, len(user_songs))
            recalls.append(recall)

            # Compute NDCG
            dcg = 0.0
            for i, song in enumerate(top_k_songs):
                if song in user_songs:
                    dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

            # Ideal DCG
            ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(user_songs))))

            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
            ndcgs.append(ndcg)

    return {
        "recall@10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg@10": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }


def train_epoch(
    model: torch.nn.Module,
    graph,
    optimizer: torch.optim.Optimizer,
    train_mask: torch.Tensor,
    batch_size: int = 512,
) -> float:
    """Train one epoch on training edges."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Get training edges
    edge_index = graph["user", "listens", "song"].edge_index[:, train_mask]
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
    """Train GAT recommender model with improved pipeline."""
    parser = argparse.ArgumentParser(description="Train GAT recommender with improved pipeline")

    # Data arguments
    parser.add_argument("--graph", type=str, default="data/graph.pt", help="Input graph file")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.15, help="Test split ratio")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min-lr", type=float, default=0.0001, help="Minimum learning rate")

    # Output arguments
    parser.add_argument(
        "--checkpoint-dir", type=str, default="models/checkpoints", help="Checkpoint directory"
    )
    parser.add_argument(
        "--output", type=str, default="models/model_improved.ckpt", help="Final model path"
    )

    args = parser.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    print(f"Loading graph from: {args.graph}")
    graph = torch.load(args.graph)

    # Split edges
    print(
        f"\nSplitting edges: train={1 - args.val_split - args.test_split:.0%}, "
        f"val={args.val_split:.0%}, test={args.test_split:.0%}"
    )

    edge_index = graph["user", "listens", "song"].edge_index
    train_mask, val_mask, test_mask = split_edges(edge_index, args.val_split, args.test_split)

    print(f"Train edges: {train_mask.sum():,}")
    print(f"Val edges: {val_mask.sum():,}")
    print(f"Test edges: {test_mask.sum():,}")

    # Create model
    model = GATRecommender(
        num_users=graph["user"].num_nodes,
        num_songs=graph["song"].num_nodes,
        num_artists=graph["artist"].num_nodes,
    )
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=args.min_lr)

    # Training state
    best_val_recall = -1.0  # Start with -1 to ensure first epoch saves
    best_epoch = 0
    patience_counter = 0

    # Metrics tracking
    metrics: Dict[str, List[float]] = {
        "train_loss": [],
        "val_recall@10": [],
        "val_ndcg@10": [],
        "lr": [],
    }

    print(f"\nTraining for up to {args.epochs} epochs with early stopping...")
    print("=" * 80)

    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(model, graph, optimizer, train_mask, args.batch_size)

        # Validate
        val_metrics = compute_metrics(model, graph, val_mask)
        val_recall = val_metrics["recall@10"]
        val_ndcg = val_metrics["ndcg@10"]

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Update metrics
        metrics["train_loss"].append(train_loss)
        metrics["val_recall@10"].append(val_recall)
        metrics["val_ndcg@10"].append(val_ndcg)
        metrics["lr"].append(current_lr)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Loss: {train_loss:.4f}, "
            f"Val Recall@10: {val_recall:.4f}, "
            f"Val NDCG@10: {val_ndcg:.4f}, "
            f"LR: {current_lr:.6f}"
        )

        # Learning rate scheduling
        scheduler.step(val_recall)

        # Check for improvement
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            checkpoint_path = checkpoint_dir / "best_model.ckpt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_recall@10": val_recall,
                    "val_ndcg@10": val_ndcg,
                    "train_loss": train_loss,
                    "num_users": graph["user"].num_nodes,
                    "num_songs": graph["song"].num_nodes,
                    "num_artists": graph["artist"].num_nodes,
                },
                checkpoint_path,
            )
            print(f"  → New best model saved! (Val Recall@10: {val_recall:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered! No improvement for {args.patience} epochs.")
                print(
                    f"Best model was from epoch {best_epoch} with "
                    f"Val Recall@10: {best_val_recall:.4f}"
                )
                break

    print("=" * 80)

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    best_model_path = checkpoint_dir / "best_model.ckpt"
    if not best_model_path.exists():
        print("Warning: No best model checkpoint found. Using current model.")
        # Save current model as best
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_recall@10": metrics["val_recall@10"][-1],
                "val_ndcg@10": metrics["val_ndcg@10"][-1],
            },
            best_model_path,
        )
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Test evaluation
    test_metrics = compute_metrics(model, graph, test_mask)
    print("\nTest Set Performance:")
    print(f"  - Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  - NDCG@10: {test_metrics['ndcg@10']:.4f}")

    # Save final model and metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)

    final_checkpoint = {
        "model_state_dict": model.state_dict(),
        "num_users": graph["user"].num_nodes,
        "num_songs": graph["song"].num_nodes,
        "num_artists": graph["artist"].num_nodes,
        "best_epoch": best_epoch,
        "val_recall@10": best_val_recall,
        "test_recall@10": test_metrics["recall@10"],
        "test_ndcg@10": test_metrics["ndcg@10"],
        "metrics_history": metrics,
    }

    torch.save(final_checkpoint, output_path)
    print(f"\nSaved final model to: {args.output}")

    # Save metrics
    metrics_path = output_path.with_suffix(".json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_recall@10": best_val_recall,
                "test_metrics": test_metrics,
                "training_history": metrics,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics to: {metrics_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Best validation performance at epoch {best_epoch}:")
    print(f"  - Val Recall@10: {best_val_recall:.4f}")
    print(f"  - Val NDCG@10: {best_checkpoint['val_ndcg@10']:.4f}")
    print("\nFinal test performance:")
    print(f"  - Test Recall@10: {test_metrics['recall@10']:.4f}")
    print(f"  - Test NDCG@10: {test_metrics['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()
