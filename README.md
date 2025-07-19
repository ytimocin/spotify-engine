# Spotify Engine

Graph-based music recommendations with explainable AI using Graph Attention Networks.

## Quick Start

```bash
# Setup (requires Python 3.8-3.12)
python3.12 -m venv .venv  # or python3.11
source .venv/bin/activate

# Install dependencies
pip install torch torchvision torchaudio  # PyTorch
pip install -r requirements.txt           # Other packages
pip install torch-geometric               # Graph library

# Generate data
python scripts/generate_synthetic_data.py

# Train model
python scripts/prepare_mssd.py
python -m src.build_graph
python -m src.train

# View results
jupyter notebook notebooks/quick_demo.ipynb
```

## What It Does

Recommends music by learning from listening patterns and explains WHY each song was suggested using attention weights.

## Architecture

```text
Sessions → Graph (Users, Songs, Artists) → GAT Model → Explainable Recommendations
```

## Documentation

- [Technical Architecture](docs/technical/architecture.md)
- [Data Patterns](docs/technical/data-patterns.md)
- [Future Enhancements](docs/future/)
