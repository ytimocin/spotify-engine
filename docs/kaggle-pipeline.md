# Kaggle Playlist-Based Recommendation Pipeline

This document describes the playlist-based recommendation system implemented for Kaggle data, which is fundamentally different from the session-based synthetic data pipeline.

## Overview

The Kaggle pipeline implements a **playlist completion** task where:
- We predict which tracks should be added to a playlist based on its existing tracks
- The model learns from playlist-track relationships, genre associations, and artist connections
- Recommendations are explained through genre influence and artist similarity

## Key Differences from Synthetic Pipeline

| Aspect | Synthetic (Session-based) | Kaggle (Playlist-based) |
|--------|---------------------------|-------------------------|
| **Primary Node Type** | Users | Playlists |
| **Task** | Next-song prediction | Playlist completion |
| **Edge Types** | user→song, song→artist | playlist→track, track→artist/genre |
| **Training Objective** | Session coherence | Playlist coherence |
| **Evaluation** | Recall@K on next song | Recall@K on held-out tracks |

## Architecture

### PlaylistGAT Model (`src/kaggle/models.py`)

The model uses a heterogeneous Graph Attention Network with:
- **Node types**: playlist, track, artist, genre
- **Edge types**: 
  - `playlist ↔ track` (contains/in_playlist)
  - `track ↔ artist` (by/created)
  - `track ↔ genre` (has_genre/includes_track)
  - `artist ↔ genre` (performs_genre/performed_by)

Key features:
- Combines learned embeddings with audio features
- Multi-layer GAT with residual connections
- Attention-based explainability

### Training Strategy (`src/kaggle/train.py`)

The training implements a **playlist completion** objective:
1. For each playlist, hold out the last N tracks for testing
2. Train using BPR (Bayesian Personalized Ranking) loss
3. Evaluate by predicting held-out tracks

## Usage

### Running the Complete Pipeline

```bash
# Run entire Kaggle pipeline
make kaggle-all

# Or run individual steps:
python scripts/kaggle/prepare_data.py    # Prepare data
python scripts/kaggle/build_graph.py     # Build graph
python -m src.kaggle.train               # Train model
python scripts/kaggle/test_model.py      # Test model
```

### Training Options

```bash
python -m src.kaggle.train \
    --epochs 30 \
    --lr 0.01 \
    --batch-size 32 \
    --holdout-size 5 \
    --patience 5
```

### Testing the Model

```bash
# Basic testing
python scripts/kaggle/test_model.py

# Test specific playlist with explanations
python scripts/kaggle/test_model.py \
    --playlist 100 \
    --top-k 20 \
    --explain
```

## Model Evaluation

The model is evaluated using:
- **Recall@K**: Fraction of held-out tracks that appear in top-K recommendations
- **NDCG@K**: Normalized Discounted Cumulative Gain for ranking quality

Typical performance (on sample data):
- Recall@10: ~0.15-0.25
- NDCG@10: ~0.20-0.30

## Explainability

The model provides three types of explanations:

1. **Genre Influence**: How shared genres between playlist and track contribute
2. **Artist Influence**: Whether the playlist contains other tracks by the same artist
3. **Track Similarity**: Which existing playlist tracks are most similar to the recommendation

Example output:
```
Explaining why Track 1234 was recommended for Playlist 0...
----------------------------------------------------------------------
Recommendation Score: 0.823

Genre Influence:
  - Genre 5: similarity = 0.712
  - Genre 12: similarity = 0.654

Artist Influence:
  - Artist 567: similarity = 0.789
    (Playlist already contains 2 tracks by this artist)

Similar Tracks Already in Playlist:
  - Track 890: similarity = 0.901
  - Track 345: similarity = 0.867
```

## Implementation Details

### Data Splits

```python
# For each playlist:
# - Hold out last 5 tracks for testing
# - Split remaining: 80% train, 20% validation
train_tracks, val_tracks, test_tracks = split_playlist_tracks(
    graph, holdout_size=5, val_ratio=0.2
)
```

### Loss Function

The model uses BPR loss to learn that tracks in the playlist should score higher than random tracks:

```python
# For each playlist-track pair (positive)
# Sample negative tracks not in playlist
loss = -log(sigmoid(pos_score - neg_score))
```

### Feature Processing

- **Playlists**: Aggregate audio features from member tracks
- **Tracks**: 7 audio features (energy, valence, etc.)
- **Artists**: Track count (log-scaled)
- **Genres**: One-hot encoding

## Next Steps

1. **Hyperparameter Tuning**: Optimize embedding dimensions, learning rate, etc.
2. **Advanced Features**: Add temporal features, user behavior patterns
3. **Scalability**: Implement mini-batch training for larger datasets
4. **Production Features**: Model versioning, A/B testing, online learning