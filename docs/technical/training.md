# Training Guide

## Overview

The Spotify Engine supports two distinct pipelines with different training approaches:

### Synthetic Pipeline (Session-based)
Uses modular trainer classes:
- **SimpleTrainer**: Quick experiments with all data
- **AdvancedTrainer**: Production-ready with validation and sophisticated features

Supports two model architectures:
- **Basic GAT Model**: Collaborative filtering only (~206K parameters)
- **Enhanced GAT Model**: Genre-aware with explainability (~500K+ parameters)

### Kaggle Pipeline (Playlist-based)
Uses a custom training loop for playlist completion:
- **PlaylistGAT Model**: Heterogeneous graph with playlists, tracks, artists, genres (~16M parameters)
- **Playlist Completion Objective**: Hold-out last N tracks from each playlist
- **Multiple Training Modes**: Mini (5min), Quick (15min), Balanced (45min), Full (3-4hrs)

## Training Theory

### BPR (Bayesian Personalized Ranking) Loss

Both pipelines use implicit feedback learning with different contexts:

**Synthetic (Session-based)**:
- **Positive samples**: Songs the user listened to
- **Negative samples**: Random songs they haven't heard
- **Objective**: `score(user, listened_song) > score(user, random_song)`

**Kaggle (Playlist-based)**:
- **Positive samples**: Tracks in the playlist
- **Negative samples**: Random tracks not in playlist
- **Objective**: `score(playlist, playlist_track) > score(playlist, random_track)`

```python
loss = -log(sigmoid(pos_score - neg_score))
```

## Synthetic Pipeline Training

### Basic Training (SimpleTrainer)

#### Quick Start

```bash
python -m src.synthetic.train --epochs 20
```

#### Features

- Trains on all data (no validation split)
- Fixed learning rate
- Basic metrics (Loss, Recall@10)
- Fast iteration (~2 minutes)

#### Options

| Parameter         | Default | Description         |
| ----------------- | ------- | ------------------- |
| `--epochs`        | 10      | Training iterations |
| `--lr`            | 0.01    | Learning rate       |
| `--batch-size`    | 512     | Batch size          |
| `--embedding-dim` | 32      | Embedding dimension |
| `--heads`         | 4       | GAT attention heads |
| `--use-enhanced`  | flag    | Use enhanced GAT model |

### Advanced Training (AdvancedTrainer)

#### Quick Start

```bash
python -m src.synthetic.train_improved --epochs 50 --patience 5
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
| `--use-enhanced`  | flag    | Use enhanced GAT model  |

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

## Enhanced Model Training

### Training with Genre Support

```bash
# Train enhanced model
python -m src.train_improved --epochs 50 --use-enhanced

# With custom genre weight
python -m src.train_improved --epochs 50 --use-enhanced --genre-weight 0.2
```

### Performance Comparison

| Model           | Parameters | Train Time | Val Recall@10 | Test Recall@10 | Features            |
| --------------- | ---------- | ---------- | ------------- | -------------- | ------------------- |
| Basic GAT       | ~206K      | ~2 min     | ~42%          | ~38%           | Collaborative only  |
| Enhanced GAT    | ~500K+     | ~5-10 min  | ~45%*         | ~40%*          | + Genres, Explain   |

*Performance varies based on genre weight and data quality

### Enhanced Model Benefits

1. **Better Cold Start**: Genre information helps recommend to new users
2. **Explainability**: Understand why songs were recommended
3. **Genre Diversity**: More varied recommendations
4. **User Type Awareness**: Adapts to casual/regular/power users

## Tips for Better Results

1. **Start Simple**: Use SimpleTrainer for initial experiments
2. **Monitor Metrics**: Stop if validation metrics plateau
3. **Adjust Learning Rate**: Try 0.1, 0.01, 0.001, 0.0001
4. **Batch Size**: Larger = faster but needs more memory
5. **Patience**: Increase for noisy data (10-15)
6. **Validation Size**: 20% for small datasets, 10% for large
7. **Genre Weight**: Start with 0.1-0.2 for enhanced model
8. **Model Selection**: Use basic for speed, enhanced for quality

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

## Kaggle Pipeline Training

### Overview

The Kaggle pipeline uses a custom training loop designed for playlist completion tasks:

```bash
python -m src.kaggle.train --epochs 10 --max-playlists 1000 --batch-size 128
```

### Training Modes

Configure training speed vs quality in the Makefile:

| Mode | Playlists | Epochs | Batch Size | Time | Use Case |
|------|-----------|---------|------------|------|----------|
| Mini | 500 | 3 | 256 | ~5 min | Quick testing |
| Quick | 1,000 | 5 | 128 | ~15 min | Demo quality |
| Balanced | 5,000 | 8 | 96 | ~45 min | Better results |
| Full | 50,000 | 20 | 64 | ~3-4 hrs | Best quality |

### Key Differences from Synthetic

1. **Data Splits**: Hold out last N tracks per playlist (not random)
2. **Objective**: Playlist completion (not next-song prediction)
3. **Evaluation**: Can the model predict held-out tracks?
4. **Scale**: Much larger graphs (200K+ tracks vs 5K songs)

### Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 30 | Training iterations |
| `--lr` | 0.01 | Learning rate |
| `--batch-size` | 32 | Playlists per batch |
| `--max-playlists` | None | Limit training data |
| `--holdout-size` | 5 | Tracks to hold out |
| `--patience` | 5 | Early stopping patience |

### Training Process

1. **Split Data**: Hold out last 5 tracks from each playlist for testing
2. **Create Training Graph**: Remove held-out edges
3. **Train Model**: Learn playlist-track associations
4. **Evaluate**: Can model predict the held-out tracks?

### Performance Tips

1. **Start Small**: Use `--max-playlists 1000` for initial tests
2. **Monitor Loss**: Should decrease steadily
3. **Check Recall**: Even 0.1-0.2 is decent for playlist completion
4. **Batch Size**: Larger = faster but may hurt convergence
