"""Unit tests for trainer classes."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import HeteroData

from src.trainers import AdvancedTrainer, BaseTrainer, SimpleTrainer, create_trainer


def create_mock_graph(num_users=10, num_songs=20, num_artists=5, num_edges=50):
    """Create a mock graph for testing."""
    graph = HeteroData()
    
    # Add nodes
    graph["user"].num_nodes = num_users
    graph["song"].num_nodes = num_songs
    graph["artist"].num_nodes = num_artists
    
    # Add edges (user -> song)
    user_indices = torch.randint(0, num_users, (num_edges,))
    song_indices = torch.randint(0, num_songs, (num_edges,))
    graph["user", "listens", "song"].edge_index = torch.stack([user_indices, song_indices])
    
    # Add edges (song -> artist)
    song_artist_indices = torch.randint(0, num_artists, (num_songs,))
    graph["song", "performed_by", "artist"].edge_index = torch.stack([
        torch.arange(num_songs),
        song_artist_indices
    ])
    
    return graph


class TestCreateTrainer:
    """Test trainer factory function."""
    
    def test_create_simple_trainer(self):
        """Test creating SimpleTrainer."""
        model_config = {"num_users": 10, "num_songs": 20, "num_artists": 5}
        training_config = {"lr": 0.01}
        
        trainer = create_trainer(
            "simple",
            model_config=model_config,
            training_config=training_config,
        )
        
        assert isinstance(trainer, SimpleTrainer)
        
    def test_create_advanced_trainer(self):
        """Test creating AdvancedTrainer."""
        model_config = {"num_users": 10, "num_songs": 20, "num_artists": 5}
        training_config = {"lr": 0.01}
        
        trainer = create_trainer(
            "advanced",
            model_config=model_config,
            training_config=training_config,
        )
        
        assert isinstance(trainer, AdvancedTrainer)
        
    def test_invalid_trainer_type(self):
        """Test creating trainer with invalid type."""
        with pytest.raises(ValueError, match="Unknown trainer type"):
            create_trainer("invalid")


class TestSimpleTrainer:
    """Test SimpleTrainer functionality."""
    
    def test_initialization(self):
        """Test SimpleTrainer initialization."""
        model_config = {"num_users": 10, "num_songs": 20, "num_artists": 5}
        training_config = {"lr": 0.01, "batch_size": 32}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SimpleTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.current_epoch == 0
            assert trainer.training_config["batch_size"] == 32
    
    def test_train_epoch(self):
        """Test training one epoch."""
        model_config = {"num_users": 5, "num_songs": 10, "num_artists": 3}
        training_config = {"lr": 0.01, "batch_size": 8}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SimpleTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            graph = create_mock_graph(5, 10, 3, 20)
            
            # Train one epoch
            metrics = trainer.train_epoch(graph)
            
            assert "train_loss" in metrics
            assert metrics["train_loss"] > 0
            assert trainer.global_step > 0
    
    def test_evaluate(self):
        """Test evaluation."""
        model_config = {"num_users": 5, "num_songs": 10, "num_artists": 3}
        training_config = {"lr": 0.01, "num_eval_users": 3, "eval_k": 5}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SimpleTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            graph = create_mock_graph(5, 10, 3, 30)
            
            # Evaluate
            metrics = trainer.evaluate(graph)
            
            assert "recall@10" in metrics
            assert 0 <= metrics["recall@10"] <= 1.0
    
    def test_full_training(self):
        """Test complete training loop."""
        model_config = {"num_users": 5, "num_songs": 10, "num_artists": 3}
        training_config = {"lr": 0.01, "batch_size": 8}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SimpleTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            graph = create_mock_graph(5, 10, 3, 20)
            
            # Train for 2 epochs
            results = trainer.train(graph, num_epochs=2)
            
            assert results["final_epoch"] == 2
            assert "train_loss" in results["metrics_history"]
            assert len(results["metrics_history"]["train_loss"]) == 2
            
            # Check that model was saved
            final_model_path = Path(tmpdir) / "final_model.pt"
            assert final_model_path.exists()


class TestAdvancedTrainer:
    """Test AdvancedTrainer functionality."""
    
    def test_initialization(self):
        """Test AdvancedTrainer initialization."""
        model_config = {"num_users": 10, "num_songs": 20, "num_artists": 5}
        training_config = {
            "lr": 0.01,
            "patience": 3,
            "use_scheduler": True,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdvancedTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.scheduler is not None
            assert trainer.patience == 3
            assert trainer.best_val_metric == -float("inf")
    
    def test_prepare_data(self):
        """Test data splitting."""
        model_config = {"num_users": 10, "num_songs": 20, "num_artists": 5}
        training_config = {
            "lr": 0.01,
            "val_ratio": 0.2,
            "test_ratio": 0.2,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdvancedTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            graph = create_mock_graph(10, 20, 5, 100)
            trainer.prepare_data(graph)
            
            assert trainer.train_mask is not None
            assert trainer.val_mask is not None
            assert trainer.test_mask is not None
            
            # Check splits sum to total
            total_edges = graph["user", "listens", "song"].edge_index.shape[1]
            assert trainer.train_mask.sum() + trainer.val_mask.sum() + trainer.test_mask.sum() == total_edges
    
    def test_early_stopping(self):
        """Test early stopping behavior."""
        model_config = {"num_users": 5, "num_songs": 10, "num_artists": 3}
        training_config = {
            "lr": 0.01,
            "patience": 2,
            "batch_size": 8,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdvancedTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            # Simulate epochs with no improvement
            train_metrics = {"train_loss": 0.5}
            
            # First epoch - improvement
            eval_metrics = {"val_recall@10": 0.1}
            should_stop = trainer.on_epoch_end(train_metrics, eval_metrics)
            assert not should_stop
            assert trainer.patience_counter == 0
            
            # Second epoch - no improvement
            eval_metrics = {"val_recall@10": 0.09}
            should_stop = trainer.on_epoch_end(train_metrics, eval_metrics)
            assert not should_stop
            assert trainer.patience_counter == 1
            
            # Third epoch - no improvement (patience exceeded)
            eval_metrics = {"val_recall@10": 0.08}
            should_stop = trainer.on_epoch_end(train_metrics, eval_metrics)
            assert should_stop
            assert trainer.patience_counter == 2
    
    def test_full_training_with_early_stopping(self):
        """Test complete training with early stopping."""
        model_config = {"num_users": 5, "num_songs": 10, "num_artists": 3}
        training_config = {
            "lr": 0.01,
            "patience": 2,
            "batch_size": 8,
            "num_eval_users": 3,
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = AdvancedTrainer(
                model_config=model_config,
                training_config=training_config,
                output_dir=tmpdir,
            )
            
            graph = create_mock_graph(5, 10, 3, 50)
            
            # Train (should stop early due to small data)
            results = trainer.train(graph, num_epochs=10)
            
            # Should stop before 10 epochs
            assert results["final_epoch"] <= 10
            assert "test_metrics" in results
            
            # Check that best model was saved
            best_model_path = Path(tmpdir) / "best_model.pt"
            assert best_model_path.exists()