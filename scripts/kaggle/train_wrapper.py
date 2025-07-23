"""
Wrapper script for Kaggle training that allows choosing implementation.

This script provides a unified interface to run either the original
or fast training implementation.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch  # noqa: E402

from src.kaggle.fast_train import train_fast  # noqa: E402
from src.kaggle.train import (  # noqa: E402
    create_training_graph,
    evaluate_playlist_completion,
    split_playlist_tracks,
    train_playlist_gat,
)


def main():
    """Main function for training wrapper."""
    parser = argparse.ArgumentParser(description="Train Kaggle playlist model")

    # Common arguments
    parser.add_argument(
        "--graph", type=str, default="data/kaggle/playlist_graph.pt", help="Path to playlist graph"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--holdout-size", type=int, default=5, help="Hold-out size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument(
        "--max-playlists", type=int, default=None, help="Max playlists for training"
    )
    parser.add_argument("--output-dir", type=str, default="models/kaggle", help="Output directory")

    # Implementation choice
    parser.add_argument(
        "--fast", action="store_true", help="Use fast sparse implementation (10-100x speedup)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device for training (cpu/cuda)")

    args = parser.parse_args()

    # Load graph
    print(f"Loading graph from {args.graph}...")
    graph = torch.load(args.graph, weights_only=False)

    # Split data
    print("Splitting playlist tracks for train/val/test...")
    train_tracks, val_tracks, test_tracks = split_playlist_tracks(
        graph, holdout_size=args.holdout_size, max_playlists=args.max_playlists
    )

    # Choose training implementation
    if args.fast:
        print("\n🚀 Using FAST sparse training implementation")
        print("Expected speedup: 10-100x for large batches\n")

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

        # Test evaluation
        print("\nEvaluating on test set...")
        test_metrics = evaluate_playlist_completion(
            results["model"], graph, create_training_graph(graph, train_tracks), test_tracks
        )

    else:
        print("\n📊 Using original full-graph training implementation")
        print("Note: Use --fast flag for 10-100x speedup\n")

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

        # Test evaluation
        print("\nEvaluating on test set...")
        test_metrics = evaluate_playlist_completion(
            results["model"], graph, create_training_graph(graph, train_tracks), test_tracks
        )

    # Report results
    print(f"\nTraining completed! Best epoch: {results['best_epoch']}")
    print(f"Best validation Recall@10: {results['best_val_recall']:.4f}")

    print("\nTest metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save final metrics
    import json

    metrics_path = Path(args.output_dir) / "metrics.json"

    all_metrics = {
        "best_epoch": results["best_epoch"],
        "best_val_recall@10": float(results["best_val_recall"]),
        "test_metrics": test_metrics,
        "history": results["history"],
        "implementation": "fast" if args.fast else "original",
    }

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
