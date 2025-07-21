# Training Guide

## Overview

The Spotify Engine uses two training strategies through modular trainer classes:

- **SimpleTrainer**: Quick experiments with all data
- **AdvancedTrainer**: Production-ready with validation and sophisticated features

## Training Theory

### BPR (Bayesian Personalized Ranking) Loss

We use implicit feedback learning:

- **Positive samples**: Songs the user listened to
- **Negative samples**: Random songs they haven't heard
- **Objective**: `score(user, listened_song) > score(user, random_song)`

```python
loss = -log(sigmoid(pos_score - neg_score))
```

## Basic Training (SimpleTrainer)

### Quick Start

```bash
make train
# or
python -m src.train --epochs 20
```

### Features

- Trains on all data (no validation split)
- Fixed learning rate
- Basic metrics (Loss, Recall@10)
- Fast iteration (~2 minutes)

### Options

| Parameter         | Default | Description         |
| ----------------- | ------- | ------------------- |
| `--epochs`        | 10      | Training iterations |
| `--lr`            | 0.01    | Learning rate       |
| `--batch-size`    | 512     | Batch size          |
| `--embedding-dim` | 32      | Embedding dimension |
| `--heads`         | 4       | GAT attention heads |

### When to Use

- Quick experiments
- Hyperparameter exploration
- Baseline models
- Limited data scenarios

## Advanced Training (AdvancedTrainer)

### Quick Start

```bash
make train-improved
# or
python -m src.train_improved --epochs 50 --patience 5
```

### Features

#### 1. Data Splits

- **70% Training**: Model updates
- **15% Validation**: Hyperparameter tuning
- **15% Test**: Final evaluation

#### 2. Early Stopping

- Monitors validation Recall@10
- Stops when no improvement for `patience` epochs
- Saves best model automatically

#### 3. Learning Rate Scheduling

- ReduceLROnPlateau strategy
- Halves LR when validation plateaus
- Minimum LR threshold

#### 4. Enhanced Metrics

- **Recall@K**: Fraction of relevant items in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- Separate train/val/test metrics

### Options

| Parameter         | Default | Description             |
| ----------------- | ------- | ----------------------- |
| `--epochs`        | 50      | Maximum epochs          |
| `--patience`      | 5       | Early stopping patience |
| `--lr`            | 0.01    | Initial learning rate   |
| `--min-lr`        | 0.0001  | Minimum LR              |
| `--lr-factor`     | 0.5     | LR reduction factor     |
| `--lr-patience`   | 3       | LR scheduler patience   |
| `--val-ratio`     | 0.15    | Validation split        |
| `--test-ratio`    | 0.15    | Test split              |
| `--use-scheduler` | flag    | Enable LR scheduling    |

### Output Files

```text
models/advanced/
├── best_model.pt       # Best validation checkpoint
├── final_model.pt      # Final model state
├── metrics.json        # Training history
└── checkpoint_epoch_*.pt  # Regular checkpoints
```

## Monitoring Training

### Basic Training Output

```text
Epoch 5/20 - Loss: 0.3142, Recall@10: 0.2874
```

### Advanced Training Output

```text
Epoch 16/50 - train_loss: 0.2757, val_recall@10: 0.4260, val_ndcg@10: 0.3142
Current learning rate: 0.010000
New best model! Val Recall@10: 0.4260
```

## Choosing a Trainer

| Use Case              | Recommended Trainer | Why               |
| --------------------- | ------------------- | ----------------- |
| First experiment      | SimpleTrainer       | Fast feedback     |
| Hyperparameter search | SimpleTrainer       | Quick iterations  |
| Final model           | AdvancedTrainer     | Best performance  |
| Production deployment | AdvancedTrainer     | Reliable metrics  |
| Limited time          | SimpleTrainer       | 2-3x faster       |
| Research paper        | AdvancedTrainer     | Proper evaluation |

## Custom Training

Create your own trainer by extending BaseTrainer:

```python
from src.trainers import BaseTrainer

class CustomTrainer(BaseTrainer):
    def train_epoch(self, graph):
        # Your training logic
        pass
    
    def evaluate(self, graph):
        # Your evaluation logic
        pass
```

See [Trainer Architecture](trainers.md) for details.

## Tips for Better Results

1. **Start Simple**: Use SimpleTrainer for initial experiments
2. **Monitor Metrics**: Stop if validation metrics plateau
3. **Adjust Learning Rate**: Try 0.1, 0.01, 0.001, 0.0001
4. **Batch Size**: Larger = faster but needs more memory
5. **Patience**: Increase for noisy data (10-15)
6. **Validation Size**: 20% for small datasets, 10% for large

## Common Issues

### Overfitting

- High train metrics, low validation
- **Solution**: Reduce epochs, add dropout, smaller model

### Underfitting

- Low metrics across the board
- **Solution**: More epochs, higher LR, larger model

### Memory Errors

- Out of memory during training
- **Solution**: Reduce batch size (256, 128, 64)

### Slow Convergence

- Metrics improve very slowly
- **Solution**: Increase learning rate, check data quality
