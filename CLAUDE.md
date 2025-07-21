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
python -m src.train --epochs 20              # Basic training (SimpleTrainer)
python -m src.train_improved --epochs 50     # Advanced training (AdvancedTrainer)

# Enhanced data generation and validation
python scripts/generate_synthetic_data.py --config config/weekend_heavy.yaml
python scripts/validate_data.py             # Comprehensive data validation
python scripts/visualize_data_profile.py    # Generate data quality visualizations
make profile                                 # Generate profile report
make validate                                # Run all validation checks

# Enhanced model testing
python scripts/test_enhanced_model.py --user 0 --top-k 5 --verbose
make test-model                              # Test trained model
make compare-models                          # Compare model versions

# Run demo
jupyter notebook notebooks/quick_demo.ipynb

# Code quality
make format                                  # Auto-format code
make lint                                    # Run linters
make quality                                 # All quality checks
make fix                                     # Auto-fix code issues
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
  - `listener`: Users in the dataset with behavioral types (casual, regular, power)
  - `song`: Individual tracks with genre associations
  - `artist`: Artists linked to songs with genre memberships
  - `genre`: Music genres (35 total) with Zipf-distributed popularity
  
- **Edges**:
  - `listener → song`: Listening interactions with attributes:
    - `play_count`: Number of times played
    - `sec_ratio`: Listening duration ratio
    - `edge_weight`: 0.7 * sec_ratio + 0.3 * log1p(play_count)
  - `artist → genre`: Artist genre associations (1-3 genres per artist)
  - `song → genre`: Song genre inheritance from artists
  - `listener → genre`: User genre preferences with affinity scores

### Key Components

1. **scripts/generate_synthetic_data.py**: Enhanced data generation with realistic behavioral patterns
   - Beta distribution for completion rates (eliminates 100% spikes)
   - 35 genres with Zipf distribution for realistic popularity
   - User behavioral multipliers for session length and skip rates
   - Temporal patterns with reduced early morning activity

2. **scripts/prepare_mssd.py**: ETL pipeline that aggregates raw sessions by (user_id, track_id) and calculates play metrics

3. **src/build_graph.py**: Constructs PyTorch Geometric HeteroData object with listener/song/artist/genre nodes and weighted edges

4. **src/models/**: Model architectures
   - `gat_recommender.py`: Basic single-layer GAT model with 32-dim embeddings
   - `enhanced_gat_recommender.py`: Multi-layer GAT with genre awareness and explainability

5. **src/trainers/**: Modular training architecture
   - `BaseTrainer`: Abstract base class with common functionality
   - `SimpleTrainer`: Basic training on all data
   - `AdvancedTrainer`: Production training with validation, early stopping, LR scheduling

6. **src/explainability.py**: Recommendation explanation system with attention weights and genre influence analysis

7. **src/metrics_extended.py**: Extended evaluation metrics including genre-aware diversity and coverage measures

8. **src/visualization/**: Visualization tools for attention analysis and data profiling

9. **scripts/validate_data.py**: Comprehensive data quality validation with behavioral pattern checks

10. **scripts/visualize_data_profile.py**: Enhanced data profiling with genre analysis and temporal pattern visualization

11. **notebooks/quick_demo.ipynb**: Interactive demonstration showing recommendations with explainability

## Implementation Notes

- The system uses synthetic data by default (can be replaced with real data)
- Edge weights combine listening duration ratio (70%) and log-scaled play count (30%)
- Attention weights from GAT layers provide explainability for recommendations
- Modular trainer architecture allows different training strategies
- Python 3.8-3.12 supported (tested with 3.12), PyTorch ≥2.0, torch-geometric
- Model size: 206,688 parameters (< 1MB checkpoint)
- Performance benchmarks: Validation Recall@10 ~42%, Test Recall@10 ~38%

### Recent Improvements

- **✅ Complete Genre System**: 35 genres with Zipf distribution, genre-aware GAT model, user genre preferences
- **✅ Enhanced Synthetic Data**: Beta distribution for completion rates, user behavioral multipliers, realistic temporal patterns
- **✅ Explainability Framework**: Attention weight analysis, genre influence scoring, recommendation reasoning
- **✅ Advanced Model Architecture**: Multi-layer heterogeneous GAT with genre awareness and explainability
- **✅ Comprehensive Data Validation**: Quality checks for behavioral patterns, genre distributions, temporal patterns
- **✅ Enhanced Visualization**: Genre analysis, skip-completion coupling, temporal pattern validation
- **✅ Extended Evaluation Metrics**: Genre-aware diversity, coverage, and influence analysis
- **Performance Optimization**: Matrix operations for 3-5x speedup in session generation
- **Code Quality**: Comprehensive linting with flake8, pylint, mypy, and black formatting

## Data Requirements

The project expects:
- Raw session data with columns: `user_id`, `track_id`, `ms_played`, timestamps
- Track metadata linking songs to artists
- Data should be placed in `data/` directory (see `data/README.md` for download instructions)