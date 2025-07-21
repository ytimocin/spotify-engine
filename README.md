# Spotify Engine

Graph-based music recommendations with explainable AI using Graph Attention Networks.

**Version**: 0.0 (Proof of Concept)  
**Status**: ✅ Working implementation with synthetic data

## What It Does

Recommends music by learning from listening patterns and explains WHY each song was suggested using attention weights from Graph Attention Networks. Features genre-aware recommendations with comprehensive explainability and realistic synthetic data generation.

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

1. **Enhanced Data Generation**: Create realistic synthetic data with 35 genres, user behavioral patterns, and temporal listening preferences
2. **Data Processing**: Aggregate listening sessions into user-song interaction edges with play metrics
3. **Graph Construction**: Build heterogeneous graph with users, songs, artists, and genres
4. **Model Training**: Train enhanced GAT with genre awareness and multi-layer attention mechanisms
5. **Explainable Recommendations**: Use attention weights and genre influence to explain why songs were recommended
6. **Data Validation**: Comprehensive quality checks ensuring realistic behavioral patterns and genre distributions

## Architecture

### Data Pipeline

```text
generate_synthetic_data.py (35 genres, user types, realistic patterns)
      ↓
raw_sessions.csv + genre/user metadata
      ↓
prepare_mssd.py (ETL: aggregate sessions)
      ↓
edge_list.parquet (user-song interactions with play metrics)
      ↓
build_graph.py (construct PyTorch Geometric graph with genres)
      ↓
graph.pt (heterogeneous graph with 4 node types: user, song, artist, genre)
```

### Training Architecture

```text
graph.pt (with genres)
      ↓
┌─────────────┬───────────────┐
│SimpleTrainer│AdvancedTrainer│
└──────┬──────┴───────┬───────┘
       │              │
       ↓              ↓
  model.ckpt    model_improved.ckpt
  (all data)    (with val/test splits)
```

Both trainers support:
- **Basic GAT Model**: Single-layer collaborative filtering
- **Enhanced GAT Model**: Multi-layer with genre awareness and explainability

Training strategies:
- **SimpleTrainer**: Fast training on all data, good for experiments
- **AdvancedTrainer**: Production-ready with validation, early stopping, and LR scheduling

## Model Training

### Training Options

```bash
# Basic training (fast, no validation)
make train

# Advanced training (recommended for production)
make train-improved

# Data generation and validation
make generate        # Generate realistic synthetic data
make profile         # Create data quality visualizations
make validate        # Run comprehensive data validation
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

**Basic GAT Model** (collaborative filtering only):
- **Validation Recall@10**: ~42%
- **Test Recall@10**: ~38%
- **Parameters**: 206,688 (< 1MB model)

**Enhanced GAT Model** (with genres and explainability):
- **Genre-aware recommendations**: ✅ Supported
- **Explainable predictions**: ✅ Attention weights + genre influence
- **Parameters**: ~500K+ (depending on genre count)
- **Data Quality**: 35 genres, realistic user behaviors, temporal patterns

## Model Evaluation

```bash
# Test the trained model
make test-model

# Compare all models
make compare-models

# Test enhanced model with explainability
python scripts/test_enhanced_model.py --user 0 --top-k 5 --verbose
```

The evaluation includes:
- **Standard metrics**: Recall@10, NDCG@10, validation performance
- **Genre-aware metrics**: Genre diversity, coverage, influence analysis
- **Explainability**: Attention weights, genre influence scores, recommendation reasoning

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
├── scripts/            # Data processing and validation scripts
└── src/                # Core implementation
    ├── models/         # GAT models (basic + enhanced)
    ├── trainers/       # Training strategies
    ├── explainability.py    # Recommendation explanation system
    ├── metrics_extended.py  # Genre-aware evaluation metrics
    └── visualization/  # Attention and data visualization tools
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

1. **✅ Genre Features** - Completed: Full genre-aware recommendations with explainability
2. **Model Versioning & Experiment Tracking** - Systematic experiment management and comparison
3. **Context-Aware Features** - Time-of-day, situational, and temporal recommendations  
4. **API Endpoint Development** - REST/GraphQL for serving recommendations
5. **Advanced Training Features** - Hyperparameter optimization, multi-objective training
6. **Real Music Data Integration** - Replace synthetic data with actual datasets

See [Future Enhancements](docs/future/) for the complete roadmap and implementation phases.
