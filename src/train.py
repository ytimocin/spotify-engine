"""
Train GAT model for music recommendations using SimpleTrainer.

This script uses the SimpleTrainer for basic training without validation splits.
For more advanced training with validation and early stopping, use train_improved.py.
"""

import argparse
import logging

import torch

from src.trainers import SimpleTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train GAT recommender model using SimpleTrainer."""
    parser = argparse.ArgumentParser(description="Train GAT recommender with SimpleTrainer")

    # Data arguments
    parser.add_argument("--graph", type=str, default="data/graph.pt", help="Input graph file")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--eval-k", type=int, default=10, help="K for Recall@K evaluation")
    parser.add_argument(
        "--num-eval-users", type=int, default=100, help="Number of users for evaluation"
    )

    # Model arguments
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default="models/simple", help="Output directory")

    args = parser.parse_args()

    # Load graph
    logger.info("Loading graph from: %s", args.graph)
    graph = torch.load(args.graph)

    # Model configuration
    model_config = {
        "num_users": graph["user"].num_nodes,
        "num_songs": graph["song"].num_nodes,
        "num_artists": graph["artist"].num_nodes,
        "embedding_dim": args.embedding_dim,
        "heads": args.heads,
    }

    # Training configuration
    training_config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_k": args.eval_k,
        "num_eval_users": args.num_eval_users,
    }

    # Create trainer
    trainer = SimpleTrainer(
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output_dir,
    )

    # Train model
    results = trainer.train(graph, args.epochs)

    # Print final results
    logger.info("Training complete!")
    logger.info("Final epoch: %d", results["final_epoch"])
    final_metrics = results["metrics_history"]
    if "train_loss" in final_metrics:
        logger.info("Final loss: %.4f", final_metrics["train_loss"][-1])
    if "recall@10" in final_metrics:
        logger.info("Final Recall@10: %.4f", final_metrics["recall@10"][-1])


if __name__ == "__main__":
    main()
