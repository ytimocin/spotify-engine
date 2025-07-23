"""
Train GAT model for music recommendations with improved pipeline using AdvancedTrainer.

Features:
- Train/validation/test splits
- Early stopping with best model checkpointing
- Learning rate scheduling
- Comprehensive metrics (Recall@K and NDCG@K)
"""

import argparse
import logging

import torch

from src.common.trainers import AdvancedTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Train GAT recommender model with AdvancedTrainer."""
    parser = argparse.ArgumentParser(description="Train GAT recommender with AdvancedTrainer")

    # Data arguments
    parser.add_argument(
        "--graph", type=str, default="data/synthetic/graph.pt", help="Input graph file"
    )
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test split ratio")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min-lr", type=float, default=0.0001, help="Minimum learning rate")
    parser.add_argument("--lr-factor", type=float, default=0.5, help="LR reduction factor")
    parser.add_argument("--lr-patience", type=int, default=3, help="LR scheduler patience")

    # Evaluation arguments
    parser.add_argument("--eval-k", type=int, default=10, help="K for Recall@K and NDCG@K")
    parser.add_argument(
        "--num-eval-users", type=int, default=100, help="Number of users for evaluation"
    )

    # Model arguments
    parser.add_argument("--embedding-dim", type=int, default=32, help="Embedding dimension")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")

    # Output arguments
    parser.add_argument(
        "--output-dir", type=str, default="models/synthetic/advanced", help="Output directory"
    )
    parser.add_argument(
        "--tensorboard-dir", type=str, default=None, help="Tensorboard log directory"
    )

    # Other arguments
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--use-scheduler", action="store_true", help="Use learning rate scheduler")

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.random_state)

    # Load graph
    logger.info("Loading graph from: %s", args.graph)
    graph = torch.load(args.graph, weights_only=False)

    # Model configuration
    model_config = {
        "num_users": graph["user"].num_nodes,
        "num_songs": graph["song"].num_nodes,
        "num_artists": graph["artist"].num_nodes,
        "embedding_dim": args.embedding_dim,
        "heads": args.heads,
    }

    # Add genre information if available
    if "genre" in graph.node_types:
        model_config["num_genres"] = graph["genre"].num_nodes
        model_config["use_enhanced"] = True
        logger.info("Detected %d genres in graph, using enhanced model", graph["genre"].num_nodes)

    # Training configuration
    training_config = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "min_lr": args.min_lr,
        "lr_factor": args.lr_factor,
        "lr_patience": args.lr_patience,
        "eval_k": args.eval_k,
        "num_eval_users": args.num_eval_users,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "random_state": args.random_state,
        "use_scheduler": args.use_scheduler,
    }

    # Create trainer
    trainer = AdvancedTrainer(
        model_config=model_config,
        training_config=training_config,
        output_dir=args.output_dir,
        tensorboard_dir=args.tensorboard_dir,
    )

    # Train model
    results = trainer.train(graph, args.epochs)

    # Print final results
    logger.info("\n%s", "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)

    # Training summary
    logger.info("Final epoch: %d", results["final_epoch"])
    logger.info("Best epoch: %d", trainer.best_epoch)

    # Validation performance
    if "val_recall@10" in results["metrics_history"]:
        best_val_recall = max(results["metrics_history"]["val_recall@10"])
        logger.info("Best Val Recall@10: %.4f", best_val_recall)

    # Test performance
    if "test_metrics" in results:
        logger.info("\nTest Set Performance:")
        for metric, value in results["test_metrics"].items():
            logger.info("  %s: %.4f", metric, value)

    logger.info("\nModel saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()
