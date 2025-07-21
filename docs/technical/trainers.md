# Trainer Architecture

## Overview

The Spotify Engine uses a modular trainer architecture that separates training logic from model implementation. This allows for different training strategies while reusing the same model.

## Architecture

```text
BaseTrainer (Abstract)
    ├── SimpleTrainer
    └── AdvancedTrainer
```

### BaseTrainer

The abstract base class provides:

- Model initialization
- Optimizer creation
- Training loop orchestration
- Checkpoint management
- Metrics tracking
- TensorBoard logging (optional)

### SimpleTrainer

Basic training implementation:

- Trains on all data
- No validation splits
- Fixed learning rate
- Minimal configuration

### AdvancedTrainer

Production-ready training:

- Automatic train/val/test splits
- Early stopping
- Learning rate scheduling
- Best model tracking
- Comprehensive metrics

## Using Trainers

### Via Command Line

```bash
# SimpleTrainer
python -m src.train --epochs 20

# AdvancedTrainer
python -m src.train_improved --epochs 50 --patience 5
```

### Programmatically

```python
from src.trainers import SimpleTrainer, AdvancedTrainer
import torch

# Load graph
graph = torch.load('data/graph.pt')

# Configure model
model_config = {
    "num_users": graph["user"].num_nodes,
    "num_songs": graph["song"].num_nodes,
    "num_artists": graph["artist"].num_nodes,
    "embedding_dim": 32,
    "heads": 4,
}

# Configure training
training_config = {
    "lr": 0.01,
    "batch_size": 512,
    "eval_k": 10,
    "num_eval_users": 100,
}

# Create trainer
trainer = AdvancedTrainer(
    model_config=model_config,
    training_config=training_config,
    output_dir="models/output"
)

# Train
results = trainer.train(graph, num_epochs=50)
```

### Factory Pattern

```python
from src.trainers import create_trainer

trainer = create_trainer(
    trainer_type="advanced",  # or "simple"
    model_config=model_config,
    training_config=training_config
)
```

## Creating Custom Trainers

### Step 1: Inherit from BaseTrainer

```python
from src.trainers import BaseTrainer
from typing import Dict

class CustomTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Add custom initialization
```

### Step 2: Implement Required Methods

```python
def train_epoch(self, graph) -> Dict[str, float]:
    """Train one epoch."""
    self.model.train()
    
    # Your training logic here
    # Must return metrics dict
    return {"loss": loss_value}

def evaluate(self, graph) -> Dict[str, float]:
    """Evaluate model."""
    self.model.eval()
    
    # Your evaluation logic
    # Must return metrics dict
    return {"metric": value}
```

### Step 3: Override Optional Hooks

```python
def on_epoch_start(self):
    """Called at epoch start."""
    # Custom logic (e.g., curriculum learning)

def on_epoch_end(self, train_metrics, eval_metrics) -> bool:
    """Called at epoch end."""
    # Custom logic (e.g., custom early stopping)
    # Return True to stop training
    return should_stop
```

## Example: Custom Trainer with Curriculum Learning

```python
class CurriculumTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.difficulty_schedule = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    def train_epoch(self, graph) -> Dict[str, float]:
        self.model.train()
        
        # Get current difficulty
        difficulty_idx = min(self.current_epoch // 10, 4)
        difficulty = self.difficulty_schedule[difficulty_idx]
        
        # Filter edges by difficulty
        edges = self._get_edges_by_difficulty(graph, difficulty)
        
        # Training loop...
        return {"loss": loss, "difficulty": difficulty}
```

## Configuration Reference

### Model Config

```python
model_config = {
    "num_users": int,          # Number of users
    "num_songs": int,          # Number of songs  
    "num_artists": int,        # Number of artists
    "embedding_dim": int,      # Embedding dimension (default: 32)
    "heads": int,              # GAT attention heads (default: 4)
    "dropout": float,          # Dropout rate (default: 0.1)
}
```

### Training Config

```python
training_config = {
    # Common
    "lr": float,               # Learning rate
    "batch_size": int,         # Batch size
    "eval_k": int,             # K for metrics (default: 10)
    "num_eval_users": int,     # Users to evaluate
    
    # AdvancedTrainer only
    "patience": int,           # Early stopping patience
    "val_ratio": float,        # Validation split ratio
    "test_ratio": float,       # Test split ratio
    "min_lr": float,           # Minimum learning rate
    "lr_factor": float,        # LR reduction factor
    "lr_patience": int,        # LR scheduler patience
    "use_scheduler": bool,     # Enable LR scheduling
}
```

## Best Practices

1. **Start Simple**: Use SimpleTrainer for initial experiments
2. **Validate Properly**: Use AdvancedTrainer for final models
3. **Monitor Metrics**: Log comprehensive metrics for debugging
4. **Save Checkpoints**: Enable recovery from interruptions
5. **Use Hooks**: Implement custom behavior via hooks, not by modifying core logic

## Trainer Comparison

| Feature        | SimpleTrainer | AdvancedTrainer | Custom   |
| -------------- | ------------- | --------------- | -------- |
| Complexity     | Low           | Medium          | Variable |
| Setup Time     | Minimal       | Moderate        | High     |
| Flexibility    | Low           | Medium          | High     |
| Best For       | Experiments   | Production      | Research |
| Data Splits    | No            | Yes             | Custom   |
| Early Stopping | No            | Yes             | Custom   |
| LR Scheduling  | No            | Yes             | Custom   |

## Future Extensions

Potential custom trainers:

- **MultiTaskTrainer**: Train on multiple objectives
- **FederatedTrainer**: Privacy-preserving training
- **OnlineTrainer**: Incremental learning
- **AdversarialTrainer**: Robust recommendations
- **MetaTrainer**: Few-shot learning

See the existing implementations in `src/trainers/` for examples.
