"""Data processing utilities for recommendation systems."""

from typing import Dict, Generator, List, Optional, Set, Tuple

import numpy as np
import torch


def split_edges_by_user(
    edge_index: torch.Tensor,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_edges_per_user: int = 3,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split edges into train/val/test sets ensuring each user has edges in all splits.

    Args:
        edge_index: Edge index tensor of shape (2, num_edges)
        val_ratio: Ratio of edges for validation
        test_ratio: Ratio of edges for test
        min_edges_per_user: Minimum edges required per user to perform split
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_mask, val_mask, test_mask) boolean tensors
    """
    np.random.seed(random_state)
    num_edges = edge_index.shape[1]

    # Group edges by user
    user_edges: Dict[int, List[int]] = {}
    for i in range(num_edges):
        user = int(edge_index[0, i].item())
        if user not in user_edges:
            user_edges[user] = []
        user_edges[user].append(i)

    # Initialize masks
    train_mask = torch.zeros(num_edges, dtype=torch.bool)
    val_mask = torch.zeros(num_edges, dtype=torch.bool)
    test_mask = torch.zeros(num_edges, dtype=torch.bool)

    # Split each user's edges
    for _, edges in user_edges.items():
        if len(edges) < min_edges_per_user:
            # Not enough edges to split - put all in training
            train_mask[edges] = True
            continue

        # Shuffle user's edges
        np.random.shuffle(edges)

        # Calculate split sizes
        n_val = max(1, int(len(edges) * val_ratio))
        n_test = max(1, int(len(edges) * test_ratio))
        n_train = len(edges) - n_val - n_test

        # Assign to splits
        train_mask[edges[:n_train]] = True
        val_mask[edges[n_train : n_train + n_val]] = True
        test_mask[edges[n_train + n_val :]] = True

    return train_mask, val_mask, test_mask


def create_negative_samples(
    positive_edges: torch.Tensor,
    num_items: int,
    num_neg_per_pos: int = 1,
    exclude_observed: bool = True,
    user_item_dict: Optional[Dict[int, Set[int]]] = None,
) -> torch.Tensor:
    """
    Create negative samples for training.

    Args:
        positive_edges: Positive edge indices of shape (2, num_edges)
        num_items: Total number of items
        num_neg_per_pos: Number of negative samples per positive sample
        exclude_observed: Whether to exclude observed items when sampling
        user_item_dict: Optional dict of user->items for excluding observed

    Returns:
        Negative item indices of shape (num_edges * num_neg_per_pos,)
    """
    num_edges = positive_edges.shape[1]
    negative_items = []

    for i in range(num_edges):
        user = int(positive_edges[0, i].item())

        # Sample negative items
        for _ in range(num_neg_per_pos):
            if exclude_observed and user_item_dict and user in user_item_dict:
                # Sample until we get an unobserved item
                while True:
                    neg_item = np.random.randint(0, num_items)
                    if neg_item not in user_item_dict[user]:
                        break
            else:
                neg_item = np.random.randint(0, num_items)

            negative_items.append(neg_item)

    return torch.tensor(negative_items, dtype=torch.long)


def batch_edge_iterator(
    edge_index: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Generator[torch.Tensor, None, None]:
    """
    Create batches of edges for training.

    Args:
        edge_index: Edge index tensor
        batch_size: Size of each batch
        shuffle: Whether to shuffle edges
        random_state: Optional random seed

    Yields:
        Batches of edge indices
    """
    num_edges = edge_index.shape[1]

    if shuffle:
        if random_state is not None:
            torch.manual_seed(random_state)
        perm = torch.randperm(num_edges)
        edge_index = edge_index[:, perm]

    for i in range(0, num_edges, batch_size):
        yield edge_index[:, i : i + batch_size]
