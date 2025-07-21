"""Unit tests for evaluation metrics."""

import pytest
import torch

from src.metrics import recall_at_k, ndcg_at_k, evaluate_batch


class TestRecallAtK:
    """Test cases for Recall@K metric."""
    
    def test_recall_perfect(self):
        """Test recall when all predictions are correct."""
        predicted = torch.tensor([1, 2, 3, 4, 5])
        relevant = {1, 2, 3, 4, 5}
        
        assert recall_at_k(predicted, relevant, k=5) == 1.0
        
    def test_recall_partial(self):
        """Test recall with partial matches."""
        predicted = torch.tensor([1, 2, 6, 7, 8])
        relevant = {1, 2, 3, 4, 5}
        
        # 2 hits out of 5 relevant items
        assert recall_at_k(predicted, relevant, k=5) == 2/5
        
    def test_recall_no_match(self):
        """Test recall with no matches."""
        predicted = torch.tensor([6, 7, 8, 9, 10])
        relevant = {1, 2, 3, 4, 5}
        
        assert recall_at_k(predicted, relevant, k=5) == 0.0
        
    def test_recall_empty_relevant(self):
        """Test recall with empty relevant set."""
        predicted = torch.tensor([1, 2, 3])
        relevant = set()
        
        assert recall_at_k(predicted, relevant, k=3) == 0.0
        
    def test_recall_k_larger_than_relevant(self):
        """Test recall when k > number of relevant items."""
        predicted = torch.tensor([1, 2, 6, 7, 8])
        relevant = {1, 2}
        
        # 2 hits out of 2 relevant items (not out of k=5)
        assert recall_at_k(predicted, relevant, k=5) == 1.0


class TestNDCGAtK:
    """Test cases for NDCG@K metric."""
    
    def test_ndcg_perfect_order(self):
        """Test NDCG with perfect ordering."""
        predicted = torch.tensor([1, 2, 3, 4, 5])
        relevant = {1, 2, 3, 4, 5}
        
        assert ndcg_at_k(predicted, relevant, k=5) == 1.0
        
    def test_ndcg_reversed_order(self):
        """Test NDCG with different orderings."""
        # Best case: relevant items first
        predicted_best = torch.tensor([1, 2, 6, 7, 8])
        relevant = {1, 2}
        ndcg_best = ndcg_at_k(predicted_best, relevant, k=5)
        
        # Worst case: relevant items last
        predicted_worst = torch.tensor([6, 7, 8, 1, 2])
        ndcg_worst = ndcg_at_k(predicted_worst, relevant, k=5)
        
        # Best should be better than worst
        assert ndcg_best > ndcg_worst
        assert ndcg_best == 1.0  # Perfect ordering
        assert 0 < ndcg_worst < 1.0  # Imperfect but still has items
        
    def test_ndcg_partial_match(self):
        """Test NDCG with partial matches."""
        predicted = torch.tensor([1, 6, 2, 7, 3])
        relevant = {1, 2, 3}
        
        ndcg = ndcg_at_k(predicted, relevant, k=5)
        assert 0 < ndcg < 1.0
        
    def test_ndcg_no_match(self):
        """Test NDCG with no matches."""
        predicted = torch.tensor([6, 7, 8, 9, 10])
        relevant = {1, 2, 3, 4, 5}
        
        assert ndcg_at_k(predicted, relevant, k=5) == 0.0
        
    def test_ndcg_empty_relevant(self):
        """Test NDCG with empty relevant set."""
        predicted = torch.tensor([1, 2, 3])
        relevant = set()
        
        assert ndcg_at_k(predicted, relevant, k=3) == 0.0
        
    def test_ndcg_single_relevant(self):
        """Test NDCG with single relevant item."""
        # Item at position 0 (best possible)
        predicted = torch.tensor([1, 2, 3, 4, 5])
        relevant = {1}
        assert ndcg_at_k(predicted, relevant, k=5) == 1.0
        
        # Item at position 2
        predicted = torch.tensor([2, 3, 1, 4, 5])
        relevant = {1}
        ndcg = ndcg_at_k(predicted, relevant, k=5)
        expected = 1.0 / torch.log2(torch.tensor(4.0)).item()  # 1/log2(3+1)
        assert abs(ndcg - expected) < 0.01


class TestEvaluateBatch:
    """Test cases for batch evaluation."""
    
    def test_evaluate_batch_basic(self):
        """Test batch evaluation with simple data."""
        # Create simple embeddings with matching dimensions
        embedding_dim = 8
        user_embeddings = torch.randn(3, embedding_dim)  # 3 users
        item_embeddings = torch.randn(5, embedding_dim)  # 5 items
        
        # User 0 interacted with items 0, 1
        # User 1 interacted with items 2, 3
        # User 2 interacted with items 1, 4
        interactions = {
            0: {0, 1},
            1: {2, 3},
            2: {1, 4}
        }
        
        results = evaluate_batch(
            user_embeddings, item_embeddings, interactions, k=2
        )
        
        assert "recall@2" in results
        assert "ndcg@2" in results
        assert 0 <= results["recall@2"] <= 1.0
        assert 0 <= results["ndcg@2"] <= 1.0
        
    def test_evaluate_batch_custom_metrics(self):
        """Test batch evaluation with specific metrics."""
        user_embeddings = torch.randn(10, 8)
        item_embeddings = torch.randn(20, 8)
        interactions = {i: {i, i+1, i+2} for i in range(5)}
        
        # Test with only recall
        results = evaluate_batch(
            user_embeddings, item_embeddings, interactions, 
            k=5, metrics=["recall"]
        )
        assert "recall@5" in results
        assert "ndcg@5" not in results
        
        # Test with only ndcg
        results = evaluate_batch(
            user_embeddings, item_embeddings, interactions,
            k=5, metrics=["ndcg"]
        )
        assert "ndcg@5" in results
        assert "recall@5" not in results
        
    def test_evaluate_batch_empty_interactions(self):
        """Test batch evaluation with no interactions."""
        user_embeddings = torch.randn(5, 8)
        item_embeddings = torch.randn(10, 8)
        interactions = {}
        
        results = evaluate_batch(
            user_embeddings, item_embeddings, interactions, k=5
        )
        
        assert results["recall@5"] == 0.0
        assert results["ndcg@5"] == 0.0