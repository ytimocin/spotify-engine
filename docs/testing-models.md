# Testing Trained Models

This guide explains how to test and evaluate your trained models for both pipelines.

## Quick Test

### Synthetic Pipeline (Session-based)

After training a synthetic model:

```bash
# Test the model
python -m src.synthetic.test_model

# Or use the shortcut
make test-model
```

This will:
- Load the model (prioritizes advanced trainer output)
- Show test set metrics (if model was trained with validation splits)
- Display sample recommendations for users
- Show attention-based explanations

### Kaggle Pipeline (Playlist-based)

After training a Kaggle model:

```bash
# Test the model
python scripts/kaggle/test_model.py

# Test with explanations
python scripts/kaggle/test_model.py --explain

# Test specific playlist
python scripts/kaggle/test_model.py --playlist 100 --top-k 20
```

This will:
- Load the trained playlist model
- Show playlist recommendations
- Display genre and artist influence explanations

## Detailed Testing

### Synthetic Model Testing

```bash
# Test specific model checkpoints
python -m src.synthetic.test_model --model models/synthetic/advanced/final_model.pt
python -m src.synthetic.test_model --model models/synthetic/simple/final_model.pt

# Test with custom parameters
python -m src.synthetic.test_model --user 42 --top-k 10 --verbose
```

### Kaggle Model Testing

```bash
# Test specific model
python scripts/kaggle/test_model.py --model models/kaggle/best_model.pt

# Test multiple playlists
python scripts/kaggle/test_model.py --playlist 0 --top-k 15 --explain
```

### Customize Output

For synthetic models:
```bash
# Show recommendations for more users
python -m src.synthetic.test_model --num-users 10

# Show more recommendations per user
python -m src.synthetic.test_model --num-recs 20
```

For Kaggle models:
```bash
# Test multiple playlists
for i in {0..10}; do
    python scripts/kaggle/test_model.py --playlist $i --top-k 5
done
```

### Compare Models

Compare all available models:

```bash
make compare-models
```

This shows a comparison table with:

- Test Recall@10
- Best training epoch
- Final loss
- Model file size

## Understanding the Output

### Synthetic Model Output

For session-based recommendations:

```text
Test metrics from training:
  Test Recall@10: 0.3842
  Test NDCG@10: 0.2913

User 42 Recommendations:
Listened to 87 songs

Top 5 Recommendations:
 1. Sunny Day Blues                      (Score: 0.892)
 2. Electric Dreams                      (Score: 0.831)
 3. Midnight Jazz                        (Score: 0.798)

Influenced by your listening history:
  - Rainy Night Jazz                    (Attention: 0.342)
  - Blues Collection                    (Attention: 0.281)
```

### Kaggle Model Output

For playlist-based recommendations:

```text
Playlist 100 Information:
Features: track_count=0.25, danceability=0.65, energy=0.72
Number of tracks: 25

Top 10 Recommended Tracks:
1. Track 4521 (score: 0.921)
   Features: energy=0.71, valence=0.68, danceability=0.64
   Artist: 1823
   Genres: [2, 5]

Explaining why Track 4521 was recommended:
Recommendation Score: 0.921

Genre Influence:
  - Genre 2: similarity = 0.812
  - Genre 5: similarity = 0.754

Artist Influence:
  - Artist 1823: similarity = 0.689
    (Playlist already contains 2 tracks by this artist)
```

### Model Comparison

```text
Model                             Test Recall@10  Best Epoch    Final Loss
--------------------------------------------------------------------------------
models/simple/final_model.pt      N/A             N/A           0.2531
models/advanced/final_model.pt    0.3842          16            0.2757
models/advanced/best_model.pt     0.3842          16            0.2757
```

## Interpreting Results

### Good Performance Indicators

- **Test Recall@10 > 0.35**: Model generalizes well
- **Consistent attention patterns**: Similar songs influence recommendations
- **Diverse recommendations**: Not just popular songs

### Warning Signs

- **Test Recall@10 < 0.20**: Model may be underfitting
- **All users get same recommendations**: Model not personalizing
- **Very high training recall, low test**: Overfitting

## Advanced Testing

### Test on Specific Users/Playlists

For synthetic models:
```python
from src.synthetic.test_model import load_model_and_data

# Load model
model, graph, checkpoint = load_model_and_data(
    "models/synthetic/model_improved.ckpt", 
    "data/synthetic/graph.pt"
)

# Test specific user
user_id = 123
top_songs, scores, attention = model.recommend(
    user_id, x_dict, graph, k=20
)
```

For Kaggle models:
```python
from src.kaggle.models import PlaylistGAT
import torch

# Load graph and model
graph = torch.load("data/kaggle/playlist_graph.pt")
model = PlaylistGAT(
    num_playlists=graph["playlist"].num_nodes,
    num_tracks=graph["track"].num_nodes,
    num_artists=graph["artist"].num_nodes,
    num_genres=graph["genre"].num_nodes,
)
model.load_state_dict(torch.load("models/kaggle/best_model.pt"))

# Test specific playlist
playlist_id = 123
tracks, scores = model.get_playlist_recommendations(
    playlist_id, x_dict, graph, k=20
)
```

### Export Recommendations

Save recommendations to file:

```bash
# Synthetic recommendations
python -m src.synthetic.test_model > synthetic_recommendations.txt

# Kaggle recommendations
python scripts/kaggle/test_model.py --playlist 0 --top-k 50 > playlist_recommendations.txt
```

### Visualize Attention

The attention weights can be visualized to understand why certain songs were recommended. Higher attention means stronger influence on the recommendation.

## Troubleshooting

### "No test metrics found"

This means the model was trained with SimpleTrainer (all data, no splits). Use AdvancedTrainer for proper test metrics.

### Low Test Performance

- Model might need more epochs
- Try adjusting hyperparameters
- Check if data splits are balanced

### FileNotFoundError

Make sure you've trained a model first:

```bash
# For synthetic pipeline
make synthetic-all

# For Kaggle pipeline
make kaggle-all
```

## Next Steps

After testing:

1. If performance is good → Deploy model
2. If performance is poor → Tune hyperparameters
3. To improve → Add features or try deeper architecture
