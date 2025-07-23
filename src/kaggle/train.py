"""
Training script for playlist-based GAT model.

This implements training for playlist completion tasks where we hold out
the last N tracks from each playlist for validation/testing.
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.kaggle.models import PlaylistGAT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_playlist_tracks(
    graph, holdout_size: int = 5, val_ratio: float = 0.2, seed: int = 42
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Split playlist tracks into train/val/test sets.
    
    For each playlist, we hold out the last N tracks for testing,
    and split the remaining tracks for training/validation.
    
    Args:
        graph: PyTorch Geometric graph
        holdout_size: Number of tracks to hold out per playlist for testing
        val_ratio: Ratio of remaining tracks to use for validation
        seed: Random seed
        
    Returns:
        train_tracks, val_tracks, test_tracks: Dicts mapping playlist_idx to track indices
    """
    random.seed(seed)
    torch.manual_seed(seed)
    
    train_tracks = {}
    val_tracks = {}
    test_tracks = {}
    
    # Get playlist-track edges
    edge_index = graph[("playlist", "contains", "track")].edge_index
    
    # Group tracks by playlist
    playlist_to_tracks = {}
    for i in range(edge_index.shape[1]):
        playlist_idx = edge_index[0, i].item()
        track_idx = edge_index[1, i].item()
        
        if playlist_idx not in playlist_to_tracks:
            playlist_to_tracks[playlist_idx] = []
        playlist_to_tracks[playlist_idx].append(track_idx)
    
    # Split tracks for each playlist
    for playlist_idx, tracks in playlist_to_tracks.items():
        if len(tracks) <= holdout_size + 1:
            # Too few tracks, use all for training
            train_tracks[playlist_idx] = tracks
            val_tracks[playlist_idx] = []
            test_tracks[playlist_idx] = []
        else:
            # Shuffle tracks to simulate playlist order
            shuffled_tracks = tracks.copy()
            random.shuffle(shuffled_tracks)
            
            # Hold out last N for test
            test_tracks[playlist_idx] = shuffled_tracks[-holdout_size:]
            remaining = shuffled_tracks[:-holdout_size]
            
            # Split remaining into train/val
            val_size = max(1, int(len(remaining) * val_ratio))
            val_tracks[playlist_idx] = remaining[-val_size:]
            train_tracks[playlist_idx] = remaining[:-val_size]
    
    # Log statistics
    total_train = sum(len(tracks) for tracks in train_tracks.values())
    total_val = sum(len(tracks) for tracks in val_tracks.values())
    total_test = sum(len(tracks) for tracks in test_tracks.values())
    
    logger.info(f"Track splits - Train: {total_train:,}, Val: {total_val:,}, Test: {total_test:,}")
    logger.info(f"Playlists with test tracks: {sum(1 for t in test_tracks.values() if t)}")
    
    return train_tracks, val_tracks, test_tracks


def create_training_graph(graph, train_tracks: Dict[int, List[int]]):
    """Create a subgraph containing only training edges."""
    # Clone the graph
    train_graph = graph.clone()
    
    # Filter playlist-track edges to include only training tracks
    edge_index = graph[("playlist", "contains", "track")].edge_index
    train_mask = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    
    for i in range(edge_index.shape[1]):
        playlist_idx = edge_index[0, i].item()
        track_idx = edge_index[1, i].item()
        
        if playlist_idx in train_tracks and track_idx in train_tracks[playlist_idx]:
            train_mask[i] = True
    
    # Update edges in training graph
    train_graph[("playlist", "contains", "track")].edge_index = edge_index[:, train_mask]
    train_graph[("track", "in_playlist", "playlist")].edge_index = edge_index[[1, 0], train_mask]
    
    return train_graph


def evaluate_playlist_completion(
    model: PlaylistGAT,
    graph,
    train_graph,
    eval_tracks: Dict[int, List[int]],
    k_values: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Evaluate playlist completion performance.
    
    Args:
        model: Trained model
        graph: Full graph (for getting all tracks)
        train_graph: Training graph (for making predictions)
        eval_tracks: Dict mapping playlist_idx to held-out track indices
        k_values: List of k values for metrics
        
    Returns:
        Dict of metrics
    """
    model.eval()
    metrics = {f"recall@{k}": [] for k in k_values}
    metrics.update({f"ndcg@{k}": [] for k in k_values})
    
    # Create node indices
    x_dict = {
        "playlist": torch.arange(graph["playlist"].num_nodes),
        "track": torch.arange(graph["track"].num_nodes),
        "artist": torch.arange(graph["artist"].num_nodes),
        "genre": torch.arange(graph["genre"].num_nodes),
    }
    
    with torch.no_grad():
        # Process playlists with held-out tracks
        for playlist_idx, held_out_tracks in eval_tracks.items():
            if not held_out_tracks:
                continue
                
            # Get recommendations
            rec_tracks, rec_scores = model.get_playlist_recommendations(
                playlist_idx, x_dict, train_graph, k=max(k_values), exclude_known=True
            )
            
            # Convert to sets for evaluation
            held_out_set = set(held_out_tracks)
            
            # Calculate metrics for each k
            for k in k_values:
                top_k_recs = set(rec_tracks[:k].tolist())
                
                # Recall@k
                if held_out_set:
                    recall = len(top_k_recs & held_out_set) / len(held_out_set)
                    metrics[f"recall@{k}"].append(recall)
                
                # NDCG@k
                dcg = 0.0
                for i, track_idx in enumerate(rec_tracks[:k].tolist()):
                    if track_idx in held_out_set:
                        dcg += 1.0 / torch.log2(torch.tensor(i + 2.0))
                
                # Ideal DCG (all held-out tracks at top positions)
                idcg = sum(1.0 / torch.log2(torch.tensor(i + 2.0)) 
                          for i in range(min(k, len(held_out_set))))
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                metrics[f"ndcg@{k}"].append(ndcg)
    
    # Average metrics
    avg_metrics = {}
    for metric_name, values in metrics.items():
        if values:
            avg_metrics[metric_name] = sum(values) / len(values)
        else:
            avg_metrics[metric_name] = 0.0
    
    return avg_metrics


def train_playlist_gat(
    graph,
    train_tracks: Dict[int, List[int]],
    val_tracks: Dict[int, List[int]],
    num_epochs: int = 30,
    lr: float = 0.01,
    batch_size: int = 32,
    patience: int = 5,
    output_dir: str = "models/kaggle",
) -> Dict[str, any]:
    """
    Train the playlist GAT model.
    
    Args:
        graph: Full graph
        train_tracks: Training tracks per playlist
        val_tracks: Validation tracks per playlist
        num_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for playlist sampling
        patience: Early stopping patience
        output_dir: Output directory for model
        
    Returns:
        Training results
    """
    # Create training graph
    train_graph = create_training_graph(graph, train_tracks)
    
    # Get feature dimensions from graph
    playlist_feature_dim = graph["playlist"].x.shape[1]
    track_feature_dim = graph["track"].x.shape[1]
    artist_feature_dim = graph["artist"].x.shape[1]
    
    # Initialize model
    model = PlaylistGAT(
        num_playlists=graph["playlist"].num_nodes,
        num_tracks=graph["track"].num_nodes,
        num_artists=graph["artist"].num_nodes,
        num_genres=graph["genre"].num_nodes,
        playlist_feature_dim=playlist_feature_dim,
        track_feature_dim=track_feature_dim,
        artist_feature_dim=artist_feature_dim,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.1,
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-4)
    
    # Create node indices
    x_dict = {
        "playlist": torch.arange(graph["playlist"].num_nodes),
        "track": torch.arange(graph["track"].num_nodes),
        "artist": torch.arange(graph["artist"].num_nodes),
        "genre": torch.arange(graph["genre"].num_nodes),
    }
    
    # Training loop
    best_val_recall = -float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_recall@10": []}
    
    # Get playlists with training tracks
    train_playlists = [p for p, tracks in train_tracks.items() if len(tracks) > 0]
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Shuffle playlists
        random.shuffle(train_playlists)
        
        # Process in batches
        for i in tqdm(range(0, len(train_playlists), batch_size), desc=f"Epoch {epoch+1}"):
            batch_playlists = train_playlists[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            # Get embeddings
            h_dict, _ = model(x_dict, train_graph)
            
            for playlist_idx in batch_playlists:
                playlist_tracks_set = set(train_tracks[playlist_idx])
                
                if len(playlist_tracks_set) < 2:
                    continue
                
                # Sample positive and negative tracks
                pos_tracks = random.sample(list(playlist_tracks_set), 
                                         min(5, len(playlist_tracks_set)))
                
                # Get negative tracks (not in playlist)
                all_tracks = set(range(graph["track"].num_nodes))
                neg_candidates = list(all_tracks - playlist_tracks_set)
                neg_tracks = random.sample(neg_candidates, min(5, len(neg_candidates)))
                
                # Compute scores
                playlist_emb = h_dict["playlist"][playlist_idx]
                
                for pos_track in pos_tracks:
                    pos_track_emb = h_dict["track"][pos_track]
                    pos_score = F.cosine_similarity(playlist_emb, pos_track_emb, dim=0)
                    
                    for neg_track in neg_tracks:
                        neg_track_emb = h_dict["track"][neg_track]
                        neg_score = F.cosine_similarity(playlist_emb, neg_track_emb, dim=0)
                        
                        # BPR loss
                        loss = -F.logsigmoid(pos_score - neg_score)
                        batch_loss += loss
            
            if batch_loss > 0:
                batch_loss = batch_loss / len(batch_playlists)
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        history["train_loss"].append(avg_loss)
        
        # Validation
        val_metrics = evaluate_playlist_completion(model, graph, train_graph, val_tracks)
        val_recall = val_metrics["recall@10"]
        history["val_recall@10"].append(val_recall)
        
        # Learning rate scheduling
        scheduler.step(val_recall)
        
        # Logging
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f}, "
            f"Val Recall@10: {val_recall:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # Early stopping
        if val_recall > best_val_recall:
            best_val_recall = val_recall
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load(Path(output_dir) / "best_model.pt"))
    
    return {
        "best_epoch": best_epoch,
        "best_val_recall": best_val_recall,
        "history": history,
        "model": model,
    }


def main():
    """Main function for training playlist-based GAT model."""
    parser = argparse.ArgumentParser(description="Train playlist-based GAT model")
    parser.add_argument(
        "--graph", type=str, default="data/kaggle/playlist_graph.pt", help="Path to playlist graph"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--holdout-size", type=int, default=5, help="Tracks to hold out per playlist")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/kaggle",
        help="Output directory for trained models",
    )
    
    args = parser.parse_args()
    
    # Load graph
    logger.info(f"Loading graph from {args.graph}")
    graph = torch.load(args.graph, weights_only=False)
    
    # Split data
    logger.info("Splitting playlist tracks for train/val/test...")
    train_tracks, val_tracks, test_tracks = split_playlist_tracks(
        graph, holdout_size=args.holdout_size
    )
    
    # Train model
    logger.info("Starting training...")
    results = train_playlist_gat(
        graph,
        train_tracks,
        val_tracks,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        output_dir=args.output_dir,
    )
    
    logger.info(f"Training completed! Best epoch: {results['best_epoch']}")
    logger.info(f"Best validation Recall@10: {results['best_val_recall']:.4f}")
    
    # Test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_playlist_completion(
        results["model"], graph, create_training_graph(graph, train_tracks), test_tracks
    )
    
    logger.info("Test metrics:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Save metrics
    output_path = Path(args.output_dir)
    metrics_path = output_path / "metrics.json"
    
    all_metrics = {
        "best_epoch": results["best_epoch"],
        "best_val_recall@10": results["best_val_recall"],
        "test_metrics": test_metrics,
        "history": results["history"],
    }
    
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
