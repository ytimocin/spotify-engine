"""Base trainer class for recommendation models."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None  # type: ignore

from src.models.gat_recommender import GATRecommender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base trainer class for recommendation models.

    Provides common functionality for training, evaluation, and checkpointing.
    Subclasses should implement specific training strategies.
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        output_dir: str = "models",
        tensorboard_dir: Optional[str] = None,
    ):
        """
        Initialize base trainer.

        Args:
            model_config: Model configuration (num_users, num_songs, etc.)
            training_config: Training configuration (lr, batch_size, etc.)
            output_dir: Directory for saving models and checkpoints
            tensorboard_dir: Directory for tensorboard logs (optional)
        """
        self.model_config = model_config
        self.training_config = training_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize model, optimizer, and other components
        self.model = self._create_model()
        self.optimizer = self._create_optimizer()
        self.writer = None
        if tensorboard_dir and HAS_TENSORBOARD:
            self.writer = SummaryWriter(tensorboard_dir)
        elif tensorboard_dir and not HAS_TENSORBOARD:
            logger.warning(
                "TensorBoard requested but not installed. Install with: pip install tensorboard"
            )

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.metrics_history: Dict[str, list] = {}

    def _create_model(self) -> nn.Module:
        """Create and initialize the model."""
        model = GATRecommender(**self.model_config)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        return torch.optim.Adam(self.model.parameters(), lr=self.training_config.get("lr", 0.01))

    def train(self, graph, num_epochs: int) -> Dict[str, Any]:
        """
        Main training loop.

        Args:
            graph: PyTorch Geometric HeteroData graph
            num_epochs: Number of epochs to train

        Returns:
            Dictionary containing final metrics and model state
        """
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            # Epoch start hook
            self.on_epoch_start()

            # Train one epoch
            train_metrics = self.train_epoch(graph)

            # Evaluate
            eval_metrics = self.evaluate(graph)

            # Update metrics history
            self._update_metrics_history(train_metrics, eval_metrics)

            # Log progress
            self._log_epoch_metrics(train_metrics, eval_metrics)

            # Epoch end hook
            should_stop = self.on_epoch_end(train_metrics, eval_metrics)

            # Check early stopping
            if should_stop:
                logger.info(f"Early stopping triggered at epoch {self.current_epoch}")
                break

        # Final model save
        self.save_final_model()

        return {
            "final_epoch": self.current_epoch,
            "metrics_history": self.metrics_history,
            "model_state": self.model.state_dict(),
        }

    @abstractmethod
    def train_epoch(self, graph) -> Dict[str, float]:
        """
        Train one epoch.

        Args:
            graph: Training graph

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def evaluate(self, graph) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            graph: Evaluation graph

        Returns:
            Dictionary of evaluation metrics
        """
        pass

    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics or {},
            "model_config": self.model_config,
            "training_config": self.training_config,
        }

        # Regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best model checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model checkpoint at epoch {self.current_epoch}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def save_final_model(self):
        """Save final model and metrics."""
        # Save model
        final_path = self.output_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "metrics_history": self.metrics_history,
                "num_users": self.model_config["num_users"],
                "num_songs": self.model_config["num_songs"],
                "num_artists": self.model_config["num_artists"],
            },
            final_path,
        )

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        logger.info(f"Saved final model to {final_path}")

    def _update_metrics_history(
        self, train_metrics: Dict[str, float], eval_metrics: Dict[str, float]
    ):
        """Update metrics history."""
        for key, value in train_metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

        for key, value in eval_metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append(value)

    def _log_epoch_metrics(self, train_metrics: Dict[str, float], eval_metrics: Dict[str, float]):
        """Log metrics for current epoch."""
        log_str = f"Epoch {self.current_epoch}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
        if eval_metrics:
            log_str += " | " + ", ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
        logger.info(log_str)

        # Tensorboard logging
        if self.writer:
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, self.current_epoch)
            for key, value in eval_metrics.items():
                self.writer.add_scalar(f"eval/{key}", value, self.current_epoch)

    # Hooks for subclasses
    def on_epoch_start(self):
        """Hook called at the start of each epoch."""
        pass

    def on_epoch_end(self, train_metrics: Dict[str, float], eval_metrics: Dict[str, float]) -> bool:
        """
        Hook called at the end of each epoch.

        Returns:
            True if training should stop early
        """
        return False
