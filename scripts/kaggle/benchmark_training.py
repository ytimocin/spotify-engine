"""
Benchmark script to compare original vs fast training.

This script runs both training methods and reports speedup metrics.
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch

from src.kaggle.fast_train import train_fast
from src.kaggle.train import split_playlist_tracks, train_playlist_gat


def benchmark_training(graph, max_playlists=500, epochs=3, batch_size=256):
    """Run benchmarks comparing original vs fast training."""
    print("=" * 60)
    print("TRAINING SPEED BENCHMARK")
    print("=" * 60)
    print("Configuration:")
    print(f"  Playlists: {max_playlists}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Graph nodes: {sum(graph[n].num_nodes for n in graph.node_types)}")
    print("=" * 60)

    # Split data
    print("\nPreparing data splits...")
    train_tracks, val_tracks, test_tracks = split_playlist_tracks(
        graph, holdout_size=5, max_playlists=max_playlists
    )

    num_train = len([p for p, t in train_tracks.items() if len(t) > 0])
    print(f"Training playlists: {num_train}")

    # Benchmark original training
    print("\n1. ORIGINAL TRAINING (full graph forward pass)")
    print("-" * 40)
    start_time = time.time()

    try:
        original_results = train_playlist_gat(
            graph,
            train_tracks,
            val_tracks,
            num_epochs=epochs,
            batch_size=batch_size,
            output_dir="models/kaggle_benchmark_orig",
        )
        original_time = time.time() - start_time
        original_recall = original_results["best_val_recall"]

        print(f"✓ Completed in {original_time:.1f}s")
        print(f"  Best validation recall: {original_recall:.4f}")
        print(f"  Avg time per epoch: {original_time/epochs:.1f}s")
    except Exception as e:
        print(f"✗ Failed: {e}")
        original_time = float("inf")
        original_recall = 0.0

    # Benchmark fast training
    print("\n2. FAST TRAINING (sparse operations)")
    print("-" * 40)
    start_time = time.time()

    try:
        fast_results = train_fast(
            graph,
            train_tracks,
            val_tracks,
            num_epochs=epochs,
            batch_size=batch_size,
            output_dir="models/kaggle_benchmark_fast",
        )
        fast_time = time.time() - start_time
        fast_recall = fast_results["best_val_recall"]
        fast_avg_epoch = fast_results["avg_epoch_time"]

        print(f"✓ Completed in {fast_time:.1f}s")
        print(f"  Best validation recall: {fast_recall:.4f}")
        print(f"  Avg time per epoch: {fast_avg_epoch:.1f}s")
    except Exception as e:
        print(f"✗ Failed: {e}")
        fast_time = float("inf")
        fast_recall = 0.0
        fast_avg_epoch = 0.0

    # Report speedup
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    if original_time < float("inf") and fast_time < float("inf"):
        speedup = original_time / fast_time
        print(f"Speedup: {speedup:.1f}x faster")
        print(f"Time saved: {original_time - fast_time:.1f}s")
        print(f"Per-epoch speedup: {(original_time/epochs) / fast_avg_epoch:.1f}x")

        # Compare quality
        print("\nModel quality comparison:")
        print(f"  Original recall@10: {original_recall:.4f}")
        print(f"  Fast recall@10: {fast_recall:.4f}")
        print(f"  Difference: {fast_recall - original_recall:+.4f}")

        # Extrapolate to larger scales
        print("\nProjected times for larger datasets:")
        for scale in [1000, 5000, 10000, 50000]:
            orig_projected = (original_time / max_playlists) * scale
            fast_projected = (fast_time / max_playlists) * scale
            print(f"  {scale:,} playlists: {orig_projected/60:.1f}min → {fast_projected/60:.1f}min")
    else:
        print("Benchmark failed - cannot compute speedup")


def main():
    """Main benchmark function."""
    parser = argparse.ArgumentParser(description="Benchmark training speed")
    parser.add_argument(
        "--graph",
        type=str,
        default="data/kaggle/playlist_graph.pt",
        help="Path to graph",
    )
    parser.add_argument(
        "--max-playlists",
        type=int,
        default=500,
        help="Number of playlists to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size",
    )

    args = parser.parse_args()

    # Load graph
    print(f"Loading graph from {args.graph}...")
    graph = torch.load(args.graph, weights_only=False)

    # Run benchmark
    benchmark_training(
        graph,
        max_playlists=args.max_playlists,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
