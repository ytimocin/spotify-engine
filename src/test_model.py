"""
Test and evaluate trained GAT models.

This script:
- Loads a trained model checkpoint
- Evaluates on test set (if available)
- Shows sample recommendations with explanations
- Compares models if multiple are available
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.gat_recommender import GATRecommender
from src.utils import create_node_indices


def load_model_and_data(checkpoint_path: str, graph_path: str) -> Tuple[GATRecommender, dict, dict]:
    """Load model from checkpoint and graph data."""
    # Load graph
    print(f"Loading graph from: {graph_path}")
    graph = torch.load(graph_path)

    # Load checkpoint
    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Create model
    model = GATRecommender(
        num_users=checkpoint.get("num_users", graph["user"].num_nodes),
        num_songs=checkpoint.get("num_songs", graph["song"].num_nodes),
        num_artists=checkpoint.get("num_artists", graph["artist"].num_nodes),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, graph, checkpoint


def evaluate_on_test(model: GATRecommender, graph, checkpoint: dict) -> Dict[str, float]:
    """Evaluate model on test set if available."""
    # Unused arguments kept for API consistency
    _ = model  # Will be used in future for re-evaluation
    _ = graph  # Will be used in future for re-evaluation

    # Check if this is an improved model with test metrics
    if "test_recall@10" in checkpoint:
        print("\nTest metrics from training:")
        print(f"  Test Recall@10: {checkpoint['test_recall@10']:.4f}")
        print(f"  Test NDCG@10: {checkpoint.get('test_ndcg@10', 'N/A')}")
        return {
            "test_recall@10": checkpoint["test_recall@10"],
            "test_ndcg@10": checkpoint.get("test_ndcg@10", 0.0),
        }

    print("\nNo test metrics found in checkpoint (model trained on all data)")
    return {}


def show_sample_recommendations(
    model: GATRecommender,
    graph,
    num_users: int = 5,
    num_recs: int = 10,
    songs_df_path: str = "data/synthetic_songs.csv",
) -> None:
    """Show sample recommendations with explanations."""
    print(f"\n{'=' * 80}")
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 80)

    # Load song metadata if available
    try:
        songs_df = pd.read_csv(songs_df_path)
        has_metadata = True
    except FileNotFoundError:
        print("Warning: Could not load song metadata")
        has_metadata = False

    # Sample random users
    user_indices = torch.randperm(graph["user"].num_nodes)[:num_users]

    # Create node indices
    x_dict = create_node_indices(graph)

    with torch.no_grad():
        for _, user_idx in enumerate(user_indices):
            print(f"\n{'-' * 60}")
            print(f"User {user_idx} Recommendations:")
            print("-" * 60)

            # Get user's listening history
            edge_index = graph["user", "listens", "song"].edge_index
            user_songs = edge_index[1][edge_index[0] == user_idx]
            print(f"Listened to {len(user_songs)} songs")

            # Get recommendations
            try:
                top_songs, scores, attention = model.recommend(
                    user_idx.item(), x_dict, graph, k=num_recs
                )

                print(f"\nTop {num_recs} Recommendations:")
                for j, (song_idx, score) in enumerate(zip(top_songs, scores)):
                    song_idx = song_idx.item()
                    if has_metadata and song_idx < len(songs_df):
                        song_info = songs_df.iloc[song_idx]
                        print(
                            f"{j + 1:2d}. {song_info['track_name'][:40]:<40} "
                            f"(Score: {score:.3f})"
                        )
                    else:
                        print(f"{j + 1:2d}. Song_{song_idx:04d} (Score: {score:.3f})")

                # Show attention-based explanations if available
                if attention is not None and len(user_songs) > 0:
                    print("\nInfluenced by your listening history:")
                    # Get attention weights for this user
                    edge_idx, attn_weights = attention

                    # Find edges from this user
                    user_edges = (edge_idx[0] == user_idx).nonzero(as_tuple=True)[0]
                    if len(user_edges) > 0:
                        # Get top influential songs
                        user_attn = attn_weights[user_edges].mean(dim=1)  # Average over heads
                        top_attn_idx = torch.topk(user_attn, min(3, len(user_attn)))[1]

                        for idx in top_attn_idx:
                            edge_pos = user_edges[idx]
                            influenced_song = edge_idx[1][edge_pos].item()
                            attn_val = user_attn[idx].item()

                            if has_metadata and influenced_song < len(songs_df):
                                song_info = songs_df.iloc[influenced_song]
                                print(
                                    f"  - {song_info['track_name'][:35]:<35} "
                                    f"(Attention: {attn_val:.3f})"
                                )
                            else:
                                print(
                                    f"  - Song_{influenced_song:04d} "
                                    f"(Attention: {attn_val:.3f})"
                                )

            except (RuntimeError, ValueError) as e:
                print(f"Error getting recommendations: {e}")


def compare_models(model_paths: List[str], graph_path: str, num_users: int = 100) -> None:
    """Compare multiple models side by side."""
    # num_users parameter reserved for future sampling
    _ = num_users

    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print("=" * 80)

    results = {}

    for model_path in model_paths:
        if not Path(model_path).exists():
            print(f"\nSkipping {model_path} (not found)")
            continue

        print(f"\nEvaluating: {model_path}")
        model, graph, checkpoint = load_model_and_data(model_path, graph_path)

        # Get metrics
        metrics = evaluate_on_test(model, graph, checkpoint)

        # Store results
        results[model_path] = {
            "metrics": metrics,
            "best_epoch": checkpoint.get("best_epoch", "N/A"),
            "final_loss": (
                checkpoint.get("metrics_history", {}).get("train_loss", [])[-1]
                if "metrics_history" in checkpoint
                else "N/A"
            ),
        }

    # Print comparison table
    if len(results) > 1:
        print("\n" + "-" * 80)
        print("Comparison Summary:")
        print("-" * 80)
        print(f"{'Model':<30} {'Test Recall@10':<15} {'Best Epoch':<12} {'Final Loss':<12}")
        print("-" * 80)

        for model_path, res in results.items():
            model_name = Path(model_path).name
            recall = res["metrics"].get("test_recall@10", "N/A")
            recall_str = f"{recall:.4f}" if isinstance(recall, float) else recall
            loss = res["final_loss"]
            loss_str = f"{loss:.4f}" if isinstance(loss, float) else loss

            print(f"{model_name:<30} {recall_str:<15} {res['best_epoch']:<12} {loss_str:<12}")


def main():
    """Test model performance and show recommendations."""
    parser = argparse.ArgumentParser(description="Test trained GAT recommender models")

    parser.add_argument(
        "--model", type=str, default="models/model_improved.ckpt", help="Path to model checkpoint"
    )
    parser.add_argument("--graph", type=str, default="data/graph.pt", help="Path to graph file")
    parser.add_argument(
        "--num-users",
        type=int,
        default=5,
        help="Number of sample users to show recommendations for",
    )
    parser.add_argument(
        "--num-recs", type=int, default=10, help="Number of recommendations per user"
    )
    parser.add_argument("--compare", action="store_true", help="Compare all available models")
    parser.add_argument(
        "--songs-metadata",
        type=str,
        default="data/synthetic_songs.csv",
        help="Path to songs metadata CSV",
    )

    args = parser.parse_args()

    if args.compare:
        # Compare all available models
        model_paths = [
            "models/model.ckpt",
            "models/model_improved.ckpt",
            "models/checkpoints/best_model.ckpt",
        ]
        compare_models(model_paths, args.graph, args.num_users)
    else:
        # Test single model
        model, graph, checkpoint = load_model_and_data(args.model, args.graph)

        # Show test metrics
        evaluate_on_test(model, graph, checkpoint)

        # Show sample recommendations
        show_sample_recommendations(
            model, graph, args.num_users, args.num_recs, args.songs_metadata
        )

        # Show training history if available
        if "metrics_history" in checkpoint:
            print(f"\n{'=' * 80}")
            print("TRAINING HISTORY")
            print("=" * 80)
            history = checkpoint["metrics_history"]

            if "val_recall@10" in history:
                val_recalls = history["val_recall@10"]
                best_epoch = np.argmax(val_recalls) + 1
                print("Best validation performance:")
                print(f"  Epoch {best_epoch}: Recall@10 = {max(val_recalls):.4f}")

                # Show last few epochs
                print("\nLast 5 epochs:")
                for i in range(max(0, len(val_recalls) - 5), len(val_recalls)):
                    print(f"  Epoch {i + 1}: Recall@10 = {val_recalls[i]:.4f}")


if __name__ == "__main__":
    main()
