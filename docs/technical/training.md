# Training Process

## Overview

The model is trained using implicit feedback - we only know what users listened to, not what they explicitly liked or disliked.

## Training Approach

### BPR (Bayesian Personalized Ranking) Loss

BPR assumes that users prefer songs they've listened to over songs they haven't:

- **Positive samples**: User-song pairs from listening history
- **Negative samples**: Randomly sampled songs the user hasn't listened to
- **Objective**: Make positive songs score higher than negative ones

### Why This Works

For each user-song pair (u, i) and a random song (j):

- We want: score(u, i) > score(u, j)
- BPR loss: -log(sigmoid(score(u, i) - score(u, j)))

This pushes the model to rank listened songs higher than unlistened ones.

## Current Implementation

### Data Usage

- **Training**: All edges in the graph
- **Evaluation**: Same data (checking reconstruction ability)
- **No train/test split** in v0 (keeping it simple)

### Training Loop

1. Shuffle all user-song edges
2. For each batch:
   - Get positive user-song pairs
   - Sample negative songs randomly
   - Compute BPR loss
   - Update model

### Evaluation Metric

**Recall@10**: What fraction of a user's actual listened songs appear in their top-10 recommendations?

- Evaluated on 100 random users each epoch
- Higher is better (1.0 = perfect)

## Running Training

### Basic Training

```bash
python -m src.train
```

Model checkpoints saved to: `models/model.ckpt`

### Training Options

#### Epochs

Number of times to go through all the data:

```bash
python -m src.train --epochs 20  # Default: 10
```

- More epochs = more learning, but can overfit
- Watch if Recall@10 stops improving

#### Learning Rate

How big steps the model takes when learning:

```bash
python -m src.train --lr 0.001  # Default: 0.01
```

- Higher = faster learning but might overshoot
- Lower = more stable but slower
- Try 0.1, 0.01, 0.001

#### Batch Size

How many examples to process at once:

```bash
python -m src.train --batch-size 256  # Default: 512
```

- Larger = faster training, more memory
- Smaller = more stable updates
- Limited by your RAM/GPU memory

#### Combine Options

```bash
python -m src.train --epochs 20 --lr 0.001 --batch-size 256
```

### What to Look For

Good training shows:

- Loss decreasing (lower is better)
- Recall@10 increasing (higher is better, max 1.0)
- Both stabilizing after a few epochs
