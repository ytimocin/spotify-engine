"""Loss functions for recommendation models."""

import torch


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Bayesian Personalized Ranking (BPR) loss.

    BPR loss optimizes the relative ordering between positive and negative items
    for implicit feedback recommendation systems.

    Args:
        pos_scores: Scores for positive (observed) user-item pairs
        neg_scores: Scores for negative (unobserved) user-item pairs
        eps: Small epsilon value for numerical stability

    Returns:
        Mean BPR loss across all pairs
    """
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + eps).mean()
