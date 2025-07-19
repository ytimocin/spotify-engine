# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**spotify-engine** is a graph-based music recommendation system that provides explainable recommendations using Graph Attention Networks (GAT).

**Core Concept**: Transform Spotify listening sessions into a heterogeneous graph, train a GAT model, and use attention weights to provide human-readable explanations for recommendations.

## Development Commands

```bash
# Environment setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline (using Makefile)
make all        # Generate data + train model
make demo       # Launch Jupyter demo
make clean      # Remove generated files

# Individual steps
python scripts/generate_synthetic_data.py    # Create synthetic data
python scripts/prepare_mssd.py               # ETL to edge list
python -m src.build_graph                    # Build PyG graph
python -m src.train --epochs 20              # Train model

# Run demo
jupyter notebook notebooks/quick_demo.ipynb

# Validation and quality checks
python scripts/validate_data.py              # Verify data integrity
make lint                                    # Run linters
make format                                  # Auto-format code

# Testing (when implemented)
pytest tests/

# Linting and formatting
black src/ scripts/ --line-length=100        # Format code
isort src/ scripts/                          # Sort imports
flake8 src/ scripts/ --max-line-length=100   # Check style

# Type checking (when configured)
mypy src/ --ignore-missing-imports
```

## Architecture

### Data Flow Pipeline

```
raw_sessions.csv → prepare_mssd.py → aggregated_edge_list.parquet
                                            ↓
                                     build_graph.py → graph.pt
                                            ↓
                                        train.py → model.ckpt + metrics.json
                                            ↓
                                     quick_demo.ipynb (recommendations + explanations)
```

### Graph Structure

- **Nodes**: 
  - `listener`: Users in the dataset
  - `song`: Individual tracks
  - `artist`: Artists linked to songs
  
- **Edges**:
  - `listener → song`: Listening interactions with attributes:
    - `play_count`: Number of times played
    - `sec_ratio`: Listening duration ratio
    - `edge_weight`: 0.7 * sec_ratio + 0.3 * log1p(play_count)

### Key Components

1. **scripts/prepare_mssd.py**: ETL pipeline that aggregates raw sessions by (user_id, track_id) and calculates play metrics

2. **src/build_graph.py**: Constructs PyTorch Geometric HeteroData object with listener/song/artist nodes and weighted edges

3. **src/models/gat_recommender.py**: Single-layer GAT model with 32-dim embeddings and dot-product scoring

4. **src/train.py**: Training loop using neighbor sampling, BPR loss, and Recall@10 evaluation

5. **notebooks/quick_demo.ipynb**: Interactive demonstration showing top-5 recommendations with attention coefficients

## Implementation Notes

- The system uses mini-MSSD dataset (1% sample of Spotify's Music Streaming Sessions Dataset)
- Edge weights combine listening duration ratio (70%) and log-scaled play count (30%)
- Attention weights from GAT layers provide explainability for recommendations
- All scripts assume Python 3.10+, PyTorch ≥2.2, torch-geometric

## Data Requirements

The project expects:
- Raw session data with columns: `user_id`, `track_id`, `ms_played`, timestamps
- Track metadata linking songs to artists
- Data should be placed in `data/` directory (see `data/README.md` for download instructions)