# Technical Architecture

## System Design

### Graph Structure

The system models music listening as a heterogeneous graph with three node types:

- **Listener Nodes**: Represent users in the system
- **Song Nodes**: Individual tracks with duration and metadata
- **Artist Nodes**: Music artists linked to their songs

### Edge Relationships

- **Listener → Song**: Primary interaction edges with attributes:
  - `play_count`: Number of times played
  - `sec_ratio`: Proportion of song duration listened
  - `edge_weight`: Composite score (0.7 × sec_ratio + 0.3 × log(play_count + 1))

### Model Architecture

**Graph Attention Network (GAT)**

- Single attention layer for interpretability
- 32-dimensional node embeddings
- Multi-head attention (4 heads)
- Dropout for regularization
- Final dot-product scorer for ranking

### Training Pipeline

1. **Neighbor Sampling**: Sample subgraphs for mini-batch training
2. **BPR Loss**: Bayesian Personalized Ranking for implicit feedback
3. **Negative Sampling**: Sample unplayed songs as negatives
4. **Evaluation**: Recall@10 on held-out interactions

### Explainability Mechanism

The attention weights α_ij from the GAT layer indicate how much listener i's previous interaction with song j influences new recommendations. These weights are:

- Normalized between 0-1
- Directly interpretable as influence scores
- Visualizable as heatmaps
