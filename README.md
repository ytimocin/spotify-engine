# Spotify Engine

Graph-based music recommendations with explainable AI using Graph Attention Networks.

**Version**: 0.0 (Proof of Concept)  
**Status**: ✅ Working implementation with synthetic data

## What It Does

Recommends music by learning from listening patterns and explains WHY each song was suggested using attention weights from Graph Attention Networks.

## Prerequisites

- Python 3.8-3.12 (tested with 3.12)
- 2GB+ RAM for training
- macOS, Linux, or Windows

## Quick Start

```bash
# 1. Setup environment
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install PyTorch (REQUIRED FIRST)
# macOS/Linux:
pip install torch torchvision torchaudio

# Windows:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install other dependencies
pip install -r requirements.txt
pip install torch-geometric

# 4. Generate synthetic data
python scripts/generate_synthetic_data.py

# 5. Build graph and train model
python scripts/prepare_mssd.py
python -m src.build_graph
python -m src.train_improved  # Recommended: includes validation & early stopping

# 6. View recommendations
jupyter notebook notebooks/quick_demo.ipynb
```

## How It Works

1. **Data Processing**: Aggregate listening sessions into user-song interaction edges
2. **Graph Construction**: Build heterogeneous graph with users, songs, and artists
3. **Model Training**: Train GAT to predict user preferences using attention mechanism
4. **Explainable Recommendations**: Use attention weights to explain why songs were recommended

## Architecture

### Data Pipeline

```text
raw_sessions.csv
      ↓
prepare_mssd.py (ETL: aggregate sessions)
      ↓
edge_list.parquet (user-song interactions with play metrics)
      ↓
build_graph.py (construct PyTorch Geometric graph)
      ↓
graph.pt (heterogeneous graph with 3 node types)
```

### Training Architecture

```text
graph.pt
      ↓
┌─────────────┬───────────────┐
│SimpleTrainer│AdvancedTrainer│
└──────┬──────┴───────┬───────┘
       │              │
       ↓              ↓
  model.ckpt    model_improved.ckpt
  (all data)    (with val/test splits)
```

Both trainers inherit from `BaseTrainer` and provide different training strategies:

- **SimpleTrainer**: Fast training on all data, good for experiments
- **AdvancedTrainer**: Production-ready with validation, early stopping, and LR scheduling

## Model Training

### Training Options

```bash
# Basic training (fast, no validation)
make train

# Advanced training (recommended for production)
make train-improved
```

| Feature         | Basic (`make train`) | Advanced (`make train-improved`) |
| --------------- | -------------------- | -------------------------------- |
| Data splits     | None                 | 70/15/15 train/val/test          |
| Early stopping  | No                   | Yes (patience=5)                 |
| LR scheduling   | No                   | Yes (ReduceLROnPlateau)          |
| Best model save | No                   | Yes                              |
| Metrics         | Loss, Recall@10      | + NDCG@10, validation metrics    |
| Training time   | ~2 min               | ~5 min                           |

### Performance (Synthetic Data)

With advanced training:

- **Validation Recall@10**: ~42%
- **Test Recall@10**: ~38%
- **Parameters**: 206,688 (< 1MB model)

## Model Evaluation

```bash
# Test the trained model
make test-model

# Compare all models
make compare-models
```

The comparison shows metrics, training history, and sample recommendations.

**Note**: Basic model shows N/A for test metrics (trains on all data).

## Project Structure

```text
spotify-engine/
├── data/               # Data files (gitignored)
├── docs/               # Documentation
│   ├── technical/      # Architecture details
│   └── future/         # Enhancement ideas
├── models/             # Trained model checkpoints
├── notebooks/          # Jupyter notebooks
├── scripts/            # Data processing scripts
└── src/                # Core implementation
    ├── models/         # GAT model
    └── trainers/       # Training strategies
```

## Development

### Code Quality Tools

```bash
# Install development dependencies
make dev-install

# Format code automatically
make format

# Run linters
make lint

# Run all quality checks
make quality
```

### Code Style

- **Formatter**: Black (100 char line length)
- **Import sorting**: isort
- **Linting**: flake8 + pylint
- **Type checking**: mypy

## Troubleshooting

**Import errors**: Make sure to install PyTorch before torch-geometric

**Out of memory**: Reduce batch size with `--batch-size 256` during training

**Jupyter not starting**: Run `pip install notebook` if needed

## Documentation

- [Getting Started Guide](docs/getting-started.md)
- [Technical Architecture](docs/technical/architecture.md)
- [Training Guide](docs/technical/training.md)
- [Trainer Architecture](docs/technical/trainers.md)
- [Evaluation Metrics](docs/evaluation.md)
- [Future Enhancements](docs/future/)

## Next Steps

1. **Add Genre Features** - Improve cold-start handling ([details](docs/future/genre-features.md))
2. **Model Versioning** - Track experiments systematically
3. **Create API Endpoint** - REST/GraphQL for serving recommendations
4. **Add Real Music Data** - Replace synthetic data with actual datasets

See [Future Enhancements](docs/future/) for the complete roadmap.
