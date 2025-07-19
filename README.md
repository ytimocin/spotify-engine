# Spotify Engine: Graph-Based Music Recommendations with Explainability

A prototype music recommendation system that uses Graph Attention Networks (GAT) to provide explainable song suggestions based on listening patterns.

## Problem Statement

Traditional recommendation systems often work as "black boxes" - they suggest items but can't explain why. This project demonstrates how Graph Neural Networks, specifically Graph Attention Networks, can provide both accurate recommendations AND human-readable explanations through attention weights.

## Key Innovation

By modeling music listening as a heterogeneous graph (listeners, songs, artists) and using attention mechanisms, we can:

1. Generate personalized music recommendations
2. Explain WHY each song was recommended by examining attention weights
3. Understand which listening patterns influenced each prediction

## Architecture Overview

```text
Synthetic Data Generation
         ↓
Raw Sessions → ETL Pipeline → Aggregated Edges
         ↓
Graph Construction (Heterogeneous: Listeners, Songs, Artists)  
         ↓
GAT Model Training (with attention weights)
         ↓
Explainable Recommendations
```

## Quick Start

1. **Setup Environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Generate Synthetic Data**

   ```bash
   python scripts/generate_synthetic_data.py --users 1000 --songs 5000
   ```

3. **Prepare Data**

   ```bash
   python scripts/prepare_mssd.py --input data/synthetic_sessions.csv --output data/edge_list.parquet
   ```

4. **Build Graph**

   ```bash
   python -m src.build_graph --edges data/edge_list.parquet --output data/graph.pt
   ```

5. **Train Model**

   ```bash
   python -m src.train --graph data/graph.pt --epochs 10
   ```

6. **Explore Recommendations**

   ```bash
   jupyter notebook notebooks/quick_demo.ipynb
   ```

## Technical Approach

### Graph Structure

- **Nodes**: Users (listeners), Songs, Artists
- **Edges**: User→Song interactions weighted by:
  - Listening frequency (how often)
  - Listening duration (how long)
  - Combined score: 0.7 × duration_ratio + 0.3 × log(play_count + 1)

### Model Architecture

- Single-layer Graph Attention Network (GAT)
- 32-dimensional node embeddings
- Dot-product scoring for recommendation ranking
- Attention coefficients preserved for explainability

### Training Strategy

- Bayesian Personalized Ranking (BPR) loss
- Negative sampling from unplayed songs
- Evaluation metric: Recall@10

## Expected Outputs

After training, the system can:

1. Recommend top-K songs for any user
2. Show attention heatmap explaining which of the user's previous listens influenced each recommendation
3. Provide similarity scores between songs based on listening patterns

## Project Structure

```text
spotify-engine/
├── scripts/         # Data generation and preprocessing
├── src/             # Core modules (graph building, model, training)
├── notebooks/       # Interactive demos
├── data/            # Generated and processed data
└── models/          # Saved model checkpoints
```

## Why This Approach?

1. **Explainability**: Unlike matrix factorization or deep learning approaches, GAT's attention weights directly show which user-song interactions influenced each recommendation

2. **Graph Structure**: Natural representation of music data - captures both direct interactions and multi-hop relationships (user→song→artist→other songs)

3. **Scalability**: GNN's message-passing paradigm scales well with sparse interaction data

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended but not required
- ~2GB RAM for default synthetic dataset size
