"""Advanced trainer with validation splits and early stopping."""

import logging
from typing import Any, Dict, Optional

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_utils import batch_edge_iterator, split_edges_by_user
from src.losses import bpr_loss
from src.metrics import evaluate_batch
from src.utils import create_node_indices

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class AdvancedTrainer(BaseTrainer):
    """
    Advanced trainer with validation splits and sophisticated training strategies.

    Features:
    - Train/validation/test splits
    - Learning rate scheduling
    - Early stopping
    - Best model tracking
    - Comprehensive metrics (Recall@K, NDCG@K)
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
        output_dir: str = "models",
        tensorboard_dir: Optional[str] = None,
    ):
        super().__init__(model_config, training_config, output_dir, tensorboard_dir)

        # Advanced components
        self.scheduler = self._create_scheduler()
        self.train_mask: Optional[torch.Tensor] = None
        self.val_mask: Optional[torch.Tensor] = None
        self.test_mask: Optional[torch.Tensor] = None

        # Early stopping
        self.patience = training_config.get("patience", 5)
        self.patience_counter = 0
        self.best_val_metric = -float("inf")
        self.best_epoch = 0

    def _create_scheduler(self) -> Optional[ReduceLROnPlateau]:
        """Create learning rate scheduler."""
        if self.training_config.get("use_scheduler", True):
            return ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.training_config.get("lr_factor", 0.5),
                patience=self.training_config.get("lr_patience", 3),
                min_lr=self.training_config.get("min_lr", 1e-4),
            )
        return None

    def prepare_data(self, graph) -> None:
        """Prepare data splits for training."""
        edge_index = graph["user", "listens", "song"].edge_index

        # Split edges
        val_ratio = self.training_config.get("val_ratio", 0.15)
        test_ratio = self.training_config.get("test_ratio", 0.15)

        self.train_mask, self.val_mask, self.test_mask = split_edges_by_user(
            edge_index,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_state=self.training_config.get("random_state", 42),
        )

        if self.train_mask is not None and self.val_mask is not None and self.test_mask is not None:
            logger.info(
                "Data splits - Train: %s, Val: %s, Test: %s",
                f"{self.train_mask.sum():,}",
                f"{self.val_mask.sum():,}",
                f"{self.test_mask.sum():,}",
            )

    def train(self, graph, num_epochs: int) -> Dict[str, Any]:
        """Override train to prepare data splits first."""
        self.prepare_data(graph)
        result = super().train(graph, num_epochs)

        # Add test evaluation
        # Add test evaluation
        if self.test_mask is not None:
            test_metrics = self._evaluate_on_split(graph, self.test_mask, "test")
            result["test_metrics"] = test_metrics
            logger.info("Final test metrics: %s", test_metrics)

        return result

    def train_epoch(self, graph) -> Dict[str, float]:
        """Train one epoch on training split."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Get training edges
        if self.train_mask is None:
            raise ValueError("Train mask not initialized. Call prepare_data() first.")
        edge_index = graph["user", "listens", "song"].edge_index[:, self.train_mask]
        batch_size = self.training_config.get("batch_size", 512)

        # Create node indices
        x_dict = create_node_indices(graph)

        # Process in batches
        for batch_edges in batch_edge_iterator(edge_index, batch_size):
            # Get embeddings
            embeddings = self.model(x_dict, graph)

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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

        return {"train_loss": total_loss / num_batches}

    def evaluate(self, graph) -> Dict[str, float]:
        """Evaluate on validation split."""
        if self.val_mask is not None:
            return self._evaluate_on_split(graph, self.val_mask, "val")
        return {}

    def _evaluate_on_split(
        self, graph, edge_mask: torch.Tensor, split_name: str
    ) -> Dict[str, float]:
        """Evaluate model on a specific data split."""
        self.model.eval()

        # Get edges for evaluation
        edge_index = graph["user", "listens", "song"].edge_index[:, edge_mask]

        # Sample users for evaluation
        unique_users = edge_index[0].unique()
        num_eval_users = self.training_config.get("num_eval_users", 100)

        if len(unique_users) > num_eval_users:
            sample_idx = torch.randperm(len(unique_users))[:num_eval_users]
            eval_users = unique_users[sample_idx]
        else:
            eval_users = unique_users

        if len(eval_users) == 0:
            return {f"{split_name}_recall@10": 0.0, f"{split_name}_ndcg@10": 0.0}

        # Create node indices dict
        x_dict = create_node_indices(graph)

        # Get embeddings once
        with torch.no_grad():
            embeddings = self.model(x_dict, graph)

        # Build interactions dict
        interactions = {}
        for user_idx in eval_users:
            user_mask = edge_index[0] == user_idx
            user_songs = edge_index[1][user_mask].unique()
            if len(user_songs) >= 2:
                interactions[user_idx.item()] = set(user_songs.tolist())

        # Evaluate
        k = self.training_config.get("eval_k", 10)
        metrics = evaluate_batch(
            embeddings["user"], embeddings["song"], interactions, k=k, metrics=["recall", "ndcg"]
        )

        # Rename metrics with split prefix
        return {
            f"{split_name}_recall@{k}": metrics[f"recall@{k}"],
            f"{split_name}_ndcg@{k}": metrics[f"ndcg@{k}"],
        }

    def on_epoch_end(self, train_metrics: Dict[str, float], eval_metrics: Dict[str, float]) -> bool:
        """Handle end of epoch: scheduling, early stopping, checkpointing."""
        # Get validation metric for tracking
        val_metric = eval_metrics.get("val_recall@10", 0.0)

        # Learning rate scheduling
        if self.scheduler:
            self.scheduler.step(val_metric)
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info("Current learning rate: %.6f", current_lr)

        # Check for improvement
        if val_metric > self.best_val_metric:
            self.best_val_metric = val_metric
            self.best_epoch = self.current_epoch
            self.patience_counter = 0

            # Save best model
            self.save_checkpoint(eval_metrics, is_best=True)
            logger.info("New best model! Val Recall@10: %.4f", val_metric)
        else:
            self.patience_counter += 1

        # Early stopping check
        if self.patience_counter >= self.patience:
            logger.info("Early stopping triggered! No improvement for %d epochs.", self.patience)
            logger.info(
                "Best model was from epoch %d with Val Recall@10: %.4f",
                self.best_epoch,
                self.best_val_metric,
            )
            return True

        return False
