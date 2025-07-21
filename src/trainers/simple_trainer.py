"""Simple trainer for basic training without validation splits."""

import logging
from typing import Dict

import numpy as np
import torch

from src.data_utils import batch_edge_iterator
from src.losses import bpr_loss
from src.metrics import recall_at_k
from src.utils import create_node_indices

from .base_trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SimpleTrainer(BaseTrainer):
    """
    Simple trainer that trains on all data without validation splits.

    Features:
    - Fixed learning rate
    - Basic metric tracking (loss and recall@k)
    - No early stopping
    - Minimal configuration
    """

    def train_epoch(self, graph) -> Dict[str, float]:
        """Train one epoch on all available data."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Get all edges
        edge_index = graph["user", "listens", "song"].edge_index
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
        """Evaluate model using Recall@K on a sample of users."""
        self.model.eval()
        recalls = []

        # Sample users for evaluation
        num_eval_users = self.training_config.get("num_eval_users", 100)
        user_indices = torch.randperm(graph["user"].num_nodes)[:num_eval_users]

        # Create node indices dict
        x_dict = create_node_indices(graph)

        with torch.no_grad():
            # Get embeddings
            embeddings = self.model(x_dict, graph)

            for user_idx in user_indices:
                # Get user's true interactions
                edge_index = graph["user", "listens", "song"].edge_index
                user_songs = edge_index[1][edge_index[0] == user_idx]

                if len(user_songs) < 5:  # Skip users with too few interactions
                    continue

                # Get recommendations
                user_emb = embeddings["user"][user_idx]
                song_embs = embeddings["song"]
                scores = torch.matmul(song_embs, user_emb)

                # Get top-k
                k = self.training_config.get("eval_k", 10)
                _, top_k_songs = torch.topk(scores, k)

                # Compute recall
                relevant = set(user_songs.tolist())
                recall = recall_at_k(top_k_songs, relevant, k)
                recalls.append(recall)

        return {"recall@10": float(np.mean(recalls)) if recalls else 0.0}
