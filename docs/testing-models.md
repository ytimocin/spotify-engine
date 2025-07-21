# Testing Trained Models

This guide explains how to test and evaluate your trained GAT models.

## Quick Test

After training a model, test it with:

```bash
make test-model
```

This will:

- Load the model (prioritizes advanced trainer output)
- Show test set metrics (if model was trained with validation splits)
- Display sample recommendations for 3 users
- Show attention-based explanations

## Detailed Testing

### Test Specific Model

```bash
# Test models from different trainers
python -m src.test_model --model models/advanced/final_model.pt  # AdvancedTrainer
python -m src.test_model --model models/simple/final_model.pt    # SimpleTrainer

# Test best checkpoint
python -m src.test_model --model models/advanced/best_model.pt
```

### Customize Output

```bash
# Show recommendations for more users
python -m src.test_model --num-users 10

# Show more recommendations per user
python -m src.test_model --num-recs 20

# Full example
python -m src.test_model \
    --model models/model_improved.ckpt \
    --num-users 5 \
    --num-recs 15
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

### Test Metrics

For models trained with validation splits:

```text
Test metrics from training:
  Test Recall@10: 0.3842
  Test NDCG@10: 0.2913
```

### Sample Recommendations

Shows recommendations for random users:

```text
User 42 Recommendations:
Listened to 87 songs

Top 5 Recommendations:
 1. Sunny Day Blues                      (Score: 0.892)
 2. Electric Dreams                      (Score: 0.831)
 3. Midnight Jazz                        (Score: 0.798)
 4. Summer Vibes                         (Score: 0.765)
 5. Acoustic Morning                     (Score: 0.724)

Influenced by your listening history:
  - Rainy Night Jazz                    (Attention: 0.342)
  - Blues Collection                    (Attention: 0.281)
  - Morning Coffee Playlist             (Attention: 0.198)
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

### Test on Specific Users

Create a custom script:

```python
from src.test_model import load_model_and_data

# Load model
model, graph, checkpoint = load_model_and_data(
    "models/model_improved.ckpt", 
    "data/graph.pt"
)

# Test specific user
user_id = 123
top_songs, scores, attention = model.recommend(
    user_id, x_dict, graph, k=20
)
```

### Export Recommendations

Save recommendations to CSV:

```bash
python -m src.test_model --model models/model_improved.ckpt > recommendations.txt
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
make train           # SimpleTrainer
make train-improved  # AdvancedTrainer
```

## Next Steps

After testing:

1. If performance is good → Deploy model
2. If performance is poor → Tune hyperparameters
3. To improve → Add features or try deeper architecture
