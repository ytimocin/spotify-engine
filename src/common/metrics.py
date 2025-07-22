"""Evaluation metrics for recommendation systems."""

from typing import Dict, List, Optional, Set

import numpy as np
import torch


def recall_at_k(predicted: torch.Tensor, relevant: Set[int], k: int) -> float:
    """
    Calculate Recall@K for a single user.

    Args:
        predicted: Tensor of predicted item indices (sorted by score)
        relevant: Set of relevant item indices
        k: Number of top predictions to consider

    Returns:
        Recall@K score
    """
    if not relevant:
        return 0.0

    top_k = set(predicted[:k].tolist())
    hits = len(top_k & relevant)
    return hits / min(k, len(relevant))


def ndcg_at_k(predicted: torch.Tensor, relevant: Set[int], k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain) for a single user.

    Args:
        predicted: Tensor of predicted item indices (sorted by score)
        relevant: Set of relevant item indices
        k: Number of top predictions to consider

    Returns:
        NDCG@K score
    """
    if not relevant:
        return 0.0

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(predicted[:k].tolist()):
        if item in relevant:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

    # Calculate IDCG (ideal DCG)
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_batch(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    interactions: Dict[int, Set[int]],
    k: int = 10,
    metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Evaluate recommendation performance for a batch of users.

    Args:
        user_embeddings: User embedding matrix
        item_embeddings: Item embedding matrix
        interactions: Dictionary mapping user indices to sets of interacted items
        k: Number of top predictions to consider
        metrics: List of metrics to compute (default: ["recall", "ndcg"])

    Returns:
        Dictionary of metric values
    """
    if metrics is None:
        metrics = ["recall", "ndcg"]

    results: Dict[str, List[float]] = {f"{metric}@{k}": [] for metric in metrics}

    for user_idx, relevant_items in interactions.items():
        if not relevant_items:
            continue

        # Get scores for all items
        user_emb = user_embeddings[user_idx]
        scores = torch.matmul(item_embeddings, user_emb)

        # Get top-k predictions
        _, top_k_indices = torch.topk(scores, k)

        # Calculate metrics
        if "recall" in metrics:
            recall = recall_at_k(top_k_indices, relevant_items, k)
            results[f"recall@{k}"].append(recall)

        if "ndcg" in metrics:
            ndcg = ndcg_at_k(top_k_indices, relevant_items, k)
            results[f"ndcg@{k}"].append(ndcg)

    # Return mean values
    return {metric: float(np.mean(values)) if values else 0.0 for metric, values in results.items()}
