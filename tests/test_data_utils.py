"""Unit tests for data utilities."""

import pytest
import torch
import numpy as np

from src.data_utils import (
    split_edges_by_user,
    create_negative_samples,
    batch_edge_iterator
)


class TestSplitEdgesByUser:
    """Test cases for edge splitting."""
    
    def test_split_basic(self):
        """Test basic edge splitting."""
        # Create edges: user 0 -> items 0,1,2,3,4
        #               user 1 -> items 5,6,7,8,9
        edge_index = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ])
        
        train_mask, val_mask, test_mask = split_edges_by_user(
            edge_index, val_ratio=0.2, test_ratio=0.2
        )
        
        # Check masks are mutually exclusive
        assert not (train_mask & val_mask).any()
        assert not (train_mask & test_mask).any()
        assert not (val_mask & test_mask).any()
        
        # Check all edges are assigned
        assert (train_mask | val_mask | test_mask).all()
        
        # Check each user has edges in each split
        for user in [0, 1]:
            user_edges = edge_index[0] == user
            assert (train_mask & user_edges).any()
            assert (val_mask & user_edges).any()
            assert (test_mask & user_edges).any()
            
    def test_split_insufficient_edges(self):
        """Test splitting when users have too few edges."""
        # User 0 has 2 edges (< min_edges_per_user=3)
        # User 1 has 5 edges
        edge_index = torch.tensor([
            [0, 0, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6]
        ])
        
        train_mask, val_mask, test_mask = split_edges_by_user(
            edge_index, min_edges_per_user=3
        )
        
        # User 0's edges should all be in training
        user0_edges = edge_index[0] == 0
        user0_train = train_mask[user0_edges]
        assert user0_train.all()
        assert not val_mask[user0_edges].any()
        assert not test_mask[user0_edges].any()
        
        # User 1 should have edges in all splits
        user1_edges = edge_index[0] == 1
        assert train_mask[user1_edges].any()
        assert val_mask[user1_edges].any()
        assert test_mask[user1_edges].any()
        
    def test_split_reproducibility(self):
        """Test that splitting is reproducible with same seed."""
        edge_index = torch.tensor([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        ])
        
        # First split
        train1, val1, test1 = split_edges_by_user(
            edge_index, random_state=42
        )
        
        # Second split with same seed
        train2, val2, test2 = split_edges_by_user(
            edge_index, random_state=42
        )
        
        assert torch.equal(train1, train2)
        assert torch.equal(val1, val2)
        assert torch.equal(test1, test2)
        
    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        # Create many edges for better ratio testing
        users = torch.repeat_interleave(torch.arange(10), 20)
        items = torch.arange(200)
        edge_index = torch.stack([users, items])
        
        train_mask, val_mask, test_mask = split_edges_by_user(
            edge_index, val_ratio=0.15, test_ratio=0.15
        )
        
        # Check overall ratios (should be close to specified)
        total = edge_index.shape[1]
        train_ratio = train_mask.sum().item() / total
        val_ratio = val_mask.sum().item() / total
        test_ratio = test_mask.sum().item() / total
        
        assert abs(train_ratio - 0.70) < 0.05
        assert abs(val_ratio - 0.15) < 0.05
        assert abs(test_ratio - 0.15) < 0.05


class TestCreateNegativeSamples:
    """Test cases for negative sampling."""
    
    def test_negative_sampling_basic(self):
        """Test basic negative sampling."""
        positive_edges = torch.tensor([[0, 0, 1], [1, 2, 3]])
        num_items = 10
        
        neg_samples = create_negative_samples(
            positive_edges, num_items, num_neg_per_pos=1
        )
        
        assert len(neg_samples) == 3  # One per positive edge
        assert (neg_samples >= 0).all()
        assert (neg_samples < num_items).all()
        
    def test_negative_sampling_multiple(self):
        """Test multiple negative samples per positive."""
        positive_edges = torch.tensor([[0, 1], [1, 2]])
        num_items = 10
        
        neg_samples = create_negative_samples(
            positive_edges, num_items, num_neg_per_pos=3
        )
        
        assert len(neg_samples) == 6  # 3 per positive edge
        
    def test_negative_sampling_exclude_observed(self):
        """Test negative sampling with exclusion."""
        positive_edges = torch.tensor([[0, 0, 0], [0, 1, 2]])
        user_item_dict = {0: {0, 1, 2}}  # User 0 has seen items 0, 1, 2
        num_items = 5
        
        neg_samples = create_negative_samples(
            positive_edges, num_items, 
            exclude_observed=True,
            user_item_dict=user_item_dict
        )
        
        # All negative samples should be 3 or 4
        assert ((neg_samples == 3) | (neg_samples == 4)).all()


class TestBatchEdgeIterator:
    """Test cases for batch edge iterator."""
    
    def test_batch_iterator_basic(self):
        """Test basic batch iteration."""
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        
        batches = list(batch_edge_iterator(edge_index, batch_size=2, shuffle=False))
        
        assert len(batches) == 3  # 5 edges / batch_size 2 = 3 batches
        assert torch.equal(batches[0], torch.tensor([[0, 1], [5, 6]]))
        assert torch.equal(batches[1], torch.tensor([[2, 3], [7, 8]]))
        assert torch.equal(batches[2], torch.tensor([[4], [9]]))
        
    def test_batch_iterator_shuffle(self):
        """Test batch iteration with shuffling."""
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        
        # Get batches with shuffling
        batches1 = list(batch_edge_iterator(
            edge_index, batch_size=2, shuffle=True, random_state=42
        ))
        
        # Without shuffling
        batches2 = list(batch_edge_iterator(
            edge_index, batch_size=2, shuffle=False
        ))
        
        # Should be different (with high probability)
        all_equal = all(torch.equal(b1, b2) for b1, b2 in zip(batches1, batches2))
        assert not all_equal
        
    def test_batch_iterator_full_batch(self):
        """Test when batch size equals number of edges."""
        edge_index = torch.tensor([[0, 1, 2], [3, 4, 5]])
        
        batches = list(batch_edge_iterator(edge_index, batch_size=3, shuffle=False))
        
        assert len(batches) == 1
        assert torch.equal(batches[0], edge_index)