"""Unit tests for loss functions."""

import pytest
import torch

from src.losses import bpr_loss


class TestBPRLoss:
    """Test cases for BPR loss function."""
    
    def test_bpr_loss_basic(self):
        """Test basic BPR loss calculation."""
        # When positive scores > negative scores, loss should be small
        pos_scores = torch.tensor([5.0, 4.0, 3.0])
        neg_scores = torch.tensor([1.0, 0.0, -1.0])
        
        loss = bpr_loss(pos_scores, neg_scores)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() < 0.1  # Should be close to 0
        
    def test_bpr_loss_reversed(self):
        """Test BPR loss when rankings are reversed."""
        # When negative scores > positive scores, loss should be large
        pos_scores = torch.tensor([1.0, 0.0, -1.0])
        neg_scores = torch.tensor([5.0, 4.0, 3.0])
        
        loss = bpr_loss(pos_scores, neg_scores)
        
        assert loss.item() > 1.0  # Should be large
        
    def test_bpr_loss_equal_scores(self):
        """Test BPR loss when scores are equal."""
        # When scores are equal, loss should be -log(0.5) ≈ 0.693
        pos_scores = torch.tensor([2.0, 2.0, 2.0])
        neg_scores = torch.tensor([2.0, 2.0, 2.0])
        
        loss = bpr_loss(pos_scores, neg_scores)
        
        assert abs(loss.item() - 0.693) < 0.01
        
    def test_bpr_loss_gradient(self):
        """Test that BPR loss is differentiable."""
        pos_scores = torch.tensor([3.0, 2.0, 1.0], requires_grad=True)
        neg_scores = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        
        assert pos_scores.grad is not None
        assert neg_scores.grad is not None
        
    def test_bpr_loss_shape(self):
        """Test BPR loss with different input shapes."""
        # 2D tensors
        pos_scores = torch.randn(10, 5)
        neg_scores = torch.randn(10, 5)
        
        loss = bpr_loss(pos_scores, neg_scores)
        
        assert loss.shape == torch.Size([])  # Scalar output
        
    def test_bpr_loss_numerical_stability(self):
        """Test BPR loss with extreme values."""
        # Very large score differences
        pos_scores = torch.tensor([100.0])
        neg_scores = torch.tensor([-100.0])
        
        loss = bpr_loss(pos_scores, neg_scores)
        
        assert torch.isfinite(loss)
        assert loss.item() < 1e-5  # Should be very close to 0