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
      ↓
train.py (GAT model with BPR loss)
      ↓
model.ckpt + metrics.json
      ↓
quick_demo.ipynb (interactive recommendations with explanations)
```

## Model Performance

After 20 epochs of training on synthetic data:
- **Loss**: 0.6174 → 0.2497 (59% improvement)
- **Recall@10**: 7.9% → 40.9% (5x improvement)
- **Parameters**: 206,688 (lightweight model)

## Project Structure

```
spotify-engine/
├── data/               # Data files (gitignored)
├── docs/               # Documentation
│   ├── technical/      # Architecture details
│   └── future/         # Enhancement ideas
├── models/             # Trained model checkpoints
├── notebooks/          # Jupyter notebooks
├── scripts/            # Data processing scripts
└── src/                # Core implementation
    └── models/         # GAT model
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
- [Training Process](docs/technical/training.md)
- [Evaluation Metrics](docs/evaluation.md)
- [Future Enhancements](docs/future/)

## Next Steps

- Add real music data (currently using synthetic)
- Implement train/validation/test splits
- Add more sophisticated features (audio, genres)
- Create API endpoint for serving
- Deploy as web application
