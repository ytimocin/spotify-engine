"""
Fast training implementation for Kaggle playlist GAT model.

This module provides an optimized training loop that uses sparse
operations to dramatically speed up training.
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.kaggle.models import PlaylistGAT
from src.kaggle.sparse_ops import EfficientBatchProcessor
from src.kaggle.train import (
    create_training_graph,
    evaluate_playlist_completion,
    split_playlist_tracks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_vectorized_bpr_loss(
    playlist_embs: torch.Tensor,
    pos_track_embs: torch.Tensor,
    neg_track_embs: torch.Tensor,
    num_negatives_per_positive: int = 5,
) -> torch.Tensor:
    """
    Compute BPR loss in a vectorized manner.

    Args:
        playlist_embs: [B, D] playlist embeddings
        pos_track_embs: [B, D] positive track embeddings
        neg_track_embs: [B*N, D] negative track embeddings
        num_negatives_per_positive: N negatives per positive

    Returns:
        Scalar loss value
    """
    # Normalize embeddings
    playlist_embs = F.normalize(playlist_embs, p=2, dim=-1)
    pos_track_embs = F.normalize(pos_track_embs, p=2, dim=-1)
    neg_track_embs = F.normalize(neg_track_embs, p=2, dim=-1)

    # Compute positive scores
    pos_scores = (playlist_embs * pos_track_embs).sum(dim=-1)  # [B]

    # Reshape negatives for broadcasting
    batch_size = playlist_embs.size(0)
    neg_track_embs = neg_track_embs.view(batch_size, num_negatives_per_positive, -1)  # [B, N, D]

    # Compute negative scores
    neg_scores = torch.bmm(neg_track_embs, playlist_embs.unsqueeze(-1)).squeeze(-1)  # [B, N]

    # BPR loss: -log(sigmoid(pos - neg))
    # Broadcast positive scores
    pos_scores_expanded = pos_scores.unsqueeze(1).expand_as(neg_scores)  # [B, N]

    # Compute losses
    losses = -F.logsigmoid(pos_scores_expanded - neg_scores)  # [B, N]

    # Average over all pairs
    return losses.mean()


def train_fast(
    graph,
    train_tracks: Dict[int, List[int]],
    val_tracks: Dict[int, List[int]],
    num_epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 256,
    patience: int = 5,
    output_dir: str = "models/kaggle",
    device: str = "cpu",
) -> Dict:
    """
    Fast training loop using sparse operations.

    Args:
        graph: Full playlist graph
        train_tracks: Training playlist-track mappings
        val_tracks: Validation playlist-track mappings
        num_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        patience: Early stopping patience
        output_dir: Output directory
        device: Device to train on

    Returns:
        Training results dict
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Move graph to device
    graph = graph.to(device)

    # Get dimensions
    playlist_feature_dim = graph["playlist"].x.size(1) if hasattr(graph["playlist"], "x") else 0
    track_feature_dim = graph["track"].x.size(1) if hasattr(graph["track"], "x") else 0
    artist_feature_dim = graph["artist"].x.size(1) if hasattr(graph["artist"], "x") else 0

    # Initialize model
    model = PlaylistGAT(
        num_playlists=graph["playlist"].num_nodes,
        num_tracks=graph["track"].num_nodes,
        num_artists=graph["artist"].num_nodes,
        num_genres=graph["genre"].num_nodes,
        playlist_feature_dim=playlist_feature_dim,
        track_feature_dim=track_feature_dim,
        artist_feature_dim=artist_feature_dim,
        embedding_dim=32,  # Reduced for speed
        hidden_dim=32,
        num_layers=1,  # Less layers for speed
        heads=4,
        dropout=0.1,
    ).to(device)

    # Initialize batch processor
    batch_processor = EfficientBatchProcessor(graph, model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")

    # Create training graph
    train_graph = create_training_graph(graph, train_tracks)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    # Training state
    best_val_recall = 0.0
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_recall@10": [], "time_per_epoch": []}

    # Get playlists with training tracks
    train_playlists = [p for p, tracks in train_tracks.items() if len(tracks) > 0]
    logger.info(f"Training on {len(train_playlists)} playlists")

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Shuffle playlists
        random.shuffle(train_playlists)

        # Process in batches
        progress_bar = tqdm(range(0, len(train_playlists), batch_size), desc=f"Epoch {epoch + 1}")

        for i in progress_bar:
            batch_playlists = train_playlists[i : i + batch_size]

            # Prepare batch data
            playlist_indices, pos_tracks, neg_tracks = batch_processor.prepare_batch_data(
                batch_playlists, train_tracks, num_neg_samples=5
            )

            if len(playlist_indices) == 0:
                continue

            # Get unique indices for embeddings
            unique_playlists = torch.unique(playlist_indices)
            unique_tracks = torch.unique(torch.cat([pos_tracks, neg_tracks]))

            # Create index mappings
            playlist_map = {p.item(): i for i, p in enumerate(unique_playlists)}
            track_map = {t.item(): i for i, t in enumerate(unique_tracks)}

            # Map batch indices to unique indices
            batch_playlist_idx = torch.tensor([playlist_map[p.item()] for p in playlist_indices])
            batch_pos_idx = torch.tensor([track_map[t.item()] for t in pos_tracks])
            batch_neg_idx = torch.tensor([track_map[t.item()] for t in neg_tracks])

            # Compute embeddings only for needed nodes
            playlist_embs, track_embs = batch_processor.compute_batch_embeddings(
                unique_playlists, unique_tracks
            )

            # Get embeddings for batch
            batch_playlist_embs = playlist_embs[batch_playlist_idx]
            batch_pos_embs = track_embs[batch_pos_idx]
            batch_neg_embs = track_embs[batch_neg_idx]

            # Compute loss
            optimizer.zero_grad()
            loss = compute_vectorized_bpr_loss(
                batch_playlist_embs, batch_pos_embs, batch_neg_embs, num_negatives_per_positive=5
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            if num_batches > 0:
                progress_bar.set_postfix({"loss": total_loss / num_batches})

        # Average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        history["train_loss"].append(float(avg_loss))

        # Validation
        val_metrics = evaluate_playlist_completion(
            model, graph, train_graph, val_tracks, k_values=[10]
        )
        val_recall = val_metrics["recall@10"]
        history["val_recall@10"].append(float(val_recall))

        # Track time
        epoch_time = time.time() - epoch_start
        history["time_per_epoch"].append(epoch_time)

        # Learning rate scheduling
        scheduler.step(val_recall)

        # Logging
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, "
            f"Val Recall@10: {val_recall:.4f}, "
            f"Time: {epoch_time:.1f}s, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Early stopping
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch + 1
            patience_counter = 0

            # Save best model
            torch.save(model.state_dict(), output_path / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(torch.load(output_path / "best_model.pt"))

    # Report speedup
    avg_time = sum(history["time_per_epoch"]) / len(history["time_per_epoch"])
    logger.info(f"Average time per epoch: {avg_time:.1f}s")

    return {
        "best_epoch": best_epoch,
        "best_val_recall": float(best_val_recall),
        "history": history,
        "model": model,
        "avg_epoch_time": avg_time,
    }


def main():
    """Main function for fast training."""
    parser = argparse.ArgumentParser(description="Fast training for playlist GAT")
    parser.add_argument(
        "--graph", type=str, default="data/kaggle/playlist_graph.pt", help="Path to playlist graph"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--holdout-size", type=int, default=5, help="Hold-out size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--max-playlists", type=int, default=None, help="Max playlists for training"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/kaggle_fast", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    # Load graph
    logger.info(f"Loading graph from {args.graph}")
    graph = torch.load(args.graph, weights_only=False)

    # Split data
    logger.info("Splitting playlist tracks for train/val/test...")
    train_tracks, val_tracks, test_tracks = split_playlist_tracks(
        graph, holdout_size=args.holdout_size, max_playlists=args.max_playlists
    )

    # Train model
    logger.info("Starting fast training...")
    results = train_fast(
        graph,
        train_tracks,
        val_tracks,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=args.output_dir,
        device=args.device,
    )

    logger.info(f"Training completed! Best epoch: {results['best_epoch']}")
    logger.info(f"Best validation Recall@10: {results['best_val_recall']:.4f}")
    logger.info(f"Average epoch time: {results['avg_epoch_time']:.1f}s")

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_playlist_completion(
        results["model"], graph, create_training_graph(graph, train_tracks), test_tracks
    )

    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Save metrics
    metrics_path = Path(args.output_dir) / "metrics.json"
    all_metrics = {
        "best_epoch": results["best_epoch"],
        "best_val_recall@10": results["best_val_recall"],
        "test_metrics": test_metrics,
        "history": results["history"],
        "avg_epoch_time": results["avg_epoch_time"],
    }

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
