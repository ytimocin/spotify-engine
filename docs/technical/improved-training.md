# Improved Training Pipeline

This document explains the enhanced training pipeline implemented in `src/train_improved.py`.

## Key Improvements

### 1. Train/Validation/Test Splits

The improved pipeline properly splits the edge data:
- **70% Training**: Used for model updates
- **15% Validation**: Used for hyperparameter tuning and early stopping
- **15% Test**: Held out for final evaluation

The split ensures each user has edges in all three sets (when possible), preventing cold-start issues during evaluation.

### 2. Early Stopping

Training automatically stops when validation performance stops improving:
- **Patience**: 5 epochs (configurable)
- **Monitoring**: Validation Recall@10
- **Best Model**: Automatically saved when validation improves

### 3. Learning Rate Scheduling

The learning rate adapts during training:
- **Strategy**: ReduceLROnPlateau
- **Reduction Factor**: 0.5 (halves LR)
- **Patience**: 3 epochs
- **Minimum LR**: 0.0001

### 4. Enhanced Metrics

In addition to Recall@10, we now compute:
- **NDCG@10**: Normalized Discounted Cumulative Gain
  - Considers ranking position (higher ranks = more important)
  - Better metric for recommendation quality

### 5. Fair Evaluation

During validation/test evaluation:
- Training items are excluded from recommendations
- Prevents overly optimistic metrics
- More realistic performance estimates

## Usage

### Basic Usage

```bash
make train-improved
```

### Advanced Options

```bash
python -m src.train_improved \
    --epochs 100 \
    --patience 10 \
    --lr 0.01 \
    --batch-size 256 \
    --val-split 0.15 \
    --test-split 0.15
```

### Parameters

- `--epochs`: Maximum training epochs (default: 50)
- `--patience`: Early stopping patience (default: 5)
- `--lr`: Initial learning rate (default: 0.01)
- `--min-lr`: Minimum learning rate (default: 0.0001)
- `--batch-size`: Training batch size (default: 512)
- `--val-split`: Validation split ratio (default: 0.15)
- `--test-split`: Test split ratio (default: 0.15)
- `--checkpoint-dir`: Directory for saving checkpoints

## Output Files

The training creates several output files:

```
models/
├── checkpoints/
│   └── best_model.ckpt      # Best model based on validation
├── model_improved.ckpt      # Final model with metadata
└── model_improved.json      # Training history and metrics
```

## Monitoring Training

The training output shows:
- **Loss**: BPR training loss (should decrease)
- **Val Recall@10**: Validation recall (should increase)
- **Val NDCG@10**: Validation NDCG (should increase)
- **LR**: Current learning rate

Example output:
```
Epoch 16/50 - Loss: 0.2757, Val Recall@10: 0.4260, Val NDCG@10: 0.3142, LR: 0.010000
  → New best model saved! (Val Recall@10: 0.4260)
```

## Expected Performance

With the improved pipeline on synthetic data:
- **Validation Recall@10**: 40-45%
- **Test Recall@10**: 35-40%
- **Test NDCG@10**: 25-35%

The test performance is typically slightly lower than validation due to:
1. Model selection based on validation set
2. Natural variance in data distribution

## Comparison with Basic Training

| Feature | Basic (`train.py`) | Improved (`train_improved.py`) |
|---------|-------------------|--------------------------------|
| Data Splits | No (uses all data) | Yes (70/15/15) |
| Early Stopping | No | Yes |
| LR Scheduling | No | Yes |
| Best Model Save | No | Yes |
| NDCG Metric | No | Yes |
| Fair Evaluation | No | Yes |

## Tips for Better Results

1. **Increase Patience**: For noisy data, use `--patience 10`
2. **Adjust Learning Rate**: Try `--lr 0.001` for stability
3. **Larger Validation Set**: Use `--val-split 0.2` for better estimates
4. **More Epochs**: Use `--epochs 100` if not converging

## Next Steps

After training with the improved pipeline:
1. Analyze the training curves in `model_improved.json`
2. Use the best model for inference
3. Try hyperparameter tuning
4. Implement cross-validation for robust estimates