# Enhanced GAT Models

The spotify-engine supports two model architectures: a basic GAT model for collaborative filtering and an enhanced GAT model with genre awareness and explainability.

## Model Overview

### Basic GAT Model (`src/models/gat_recommender.py`)
- Single-layer Graph Attention Network
- 3 node types: user, song, artist
- ~206K parameters
- Fast training and inference

### Enhanced GAT Model (`src/models/enhanced_gat_recommender.py`)
- Multi-layer heterogeneous GAT
- 4 node types: user, song, artist, genre
- Built-in explainability features
- ~500K+ parameters (depends on genre count)

## Enhanced GAT Architecture

### Core Components

The Enhanced GAT model extends the basic architecture with:

1. **Genre-Aware Node Embeddings**
2. **Multi-Layer Heterogeneous Attention**
3. **Built-in Explainability Methods**
4. **Genre Influence Computation**

### Model Structure

```python
class EnhancedGATRecommender(nn.Module):
    def __init__(self, 
                 num_users: int, 
                 num_songs: int, 
                 num_artists: int,
                 num_genres: int = 0,
                 embedding_dim: int = 64,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 heads: int = 4,
                 dropout: float = 0.1,
                 use_genres: bool = True):
        
        # Node embeddings for all entity types
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.song_embedding = nn.Embedding(num_songs, embedding_dim)
        self.artist_embedding = nn.Embedding(num_artists, embedding_dim)
        
        if use_genres:
            self.genre_embedding = nn.Embedding(num_genres, embedding_dim)
        
        # Multi-layer heterogeneous GAT
        self.gat_layers = nn.ModuleList([
            HeteroGATConv(embedding_dim, hidden_dim, heads=heads)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * heads, embedding_dim)
```

### Heterogeneous Graph Attention

The model handles multiple edge types with specialized attention:

```python
def forward(self, x_dict, edge_index_dict, return_attention=False):
    """
    Forward pass with support for heterogeneous graphs.
    
    Args:
        x_dict: Node features by type {'user': ..., 'song': ..., 'artist': ..., 'genre': ...}
        edge_index_dict: Edge indices by type {('user', 'listens_to', 'song'): ...}
        return_attention: Whether to return attention weights
    """
    
    # Initial embeddings
    embeddings = {}
    for node_type, indices in x_dict.items():
        if node_type == 'user':
            embeddings[node_type] = self.user_embedding(indices)
        elif node_type == 'song':
            embeddings[node_type] = self.song_embedding(indices)
        elif node_type == 'artist':
            embeddings[node_type] = self.artist_embedding(indices)
        elif node_type == 'genre' and self.use_genres:
            embeddings[node_type] = self.genre_embedding(indices)
    
    attention_weights = []
    
    # Multi-layer attention propagation
    for layer in self.gat_layers:
        embeddings, attn = layer(embeddings, edge_index_dict, return_attention=True)
        if return_attention:
            attention_weights.append(attn)
    
    # Output projection
    for node_type in embeddings:
        embeddings[node_type] = self.output_proj(embeddings[node_type])
    
    if return_attention:
        return embeddings, attention_weights
    return embeddings
```

## Enhanced Features

### 1. Genre-Aware Recommendations

The model incorporates genre information at multiple levels:

```python
def recommend(self, user_id, x_dict, graph, k=10, exclude_known=True):
    """Generate genre-aware recommendations."""
    
    # Get embeddings with genre information
    embeddings = self.forward(x_dict, graph.edge_index_dict)
    
    user_emb = embeddings['user'][user_id]
    song_embs = embeddings['song']
    
    # Compute base collaborative scores
    scores = torch.matmul(song_embs, user_emb)
    
    # Add genre influence if available
    if self.use_genres and 'genre' in embeddings:
        genre_influence = self._compute_genre_influence(
            user_id, embeddings, graph
        )
        scores += self.genre_weight * genre_influence
    
    # Get top-k recommendations
    if exclude_known:
        # Mask out songs user has already listened to
        known_songs = self._get_user_songs(user_id, graph)
        scores[known_songs] = float('-inf')
    
    top_scores, top_songs = torch.topk(scores, k)
    return top_songs, top_scores, embeddings
```

### 2. Built-in Explainability

The model includes methods for explaining recommendations:

```python
def explain_recommendation(self, user_id, song_id, x_dict, graph):
    """
    Explain why a song was recommended to a user.
    
    Returns comprehensive explanation including:
    - Recommendation score
    - Genre influence analysis
    - Artist similarity
    - Attention weights
    """
    
    embeddings, attention_weights = self.forward(
        x_dict, graph.edge_index_dict, return_attention=True
    )
    
    # Base recommendation score
    user_emb = embeddings['user'][user_id]
    song_emb = embeddings['song'][song_id]
    base_score = torch.dot(user_emb, song_emb).item()
    
    explanation = {
        'user_id': user_id,
        'song_id': song_id,
        'score': base_score,
        'attention_weights': attention_weights
    }
    
    # Add genre influence if available
    if self.use_genres:
        genre_influence = self._explain_genre_influence(
            user_id, song_id, embeddings, graph
        )
        explanation['genre_influence'] = genre_influence
    
    # Add artist influence
    artist_influence = self._explain_artist_influence(
        user_id, song_id, embeddings, graph
    )
    explanation['artist_influence'] = artist_influence
    
    return explanation
```

### 3. Genre Influence Computation

Detailed analysis of how genres contribute to recommendations:

```python
def _compute_genre_influence(self, user_id, embeddings, graph):
    """Compute genre-based influence scores for all songs."""
    
    # Get user's genre preferences
    user_genres = self._get_user_genres(user_id, graph)
    if len(user_genres) == 0:
        return torch.zeros(embeddings['song'].size(0))
    
    # Get song-genre associations
    song_genre_matrix = self._get_song_genre_matrix(graph)
    
    # Compute genre influence scores
    user_genre_embs = embeddings['genre'][user_genres]
    genre_scores = torch.matmul(
        song_genre_matrix.float(), 
        user_genre_embs.mean(dim=0)
    )
    
    return genre_scores

def _explain_genre_influence(self, user_id, song_id, embeddings, graph):
    """Explain genre influence for a specific recommendation."""
    
    # Find common genres between user and song
    user_genres = set(self._get_user_genres(user_id, graph).tolist())
    song_genres = set(self._get_song_genres(song_id, graph).tolist())
    common_genres = user_genres & song_genres
    
    if not common_genres:
        return None
    
    # Calculate influence score for each common genre
    genre_influences = []
    for genre_id in common_genres:
        # User affinity for this genre
        user_genre_affinity = self._get_user_genre_affinity(user_id, genre_id, graph)
        
        # Song association with this genre
        song_genre_strength = self._get_song_genre_strength(song_id, genre_id, graph)
        
        # Combined influence
        influence_score = user_genre_affinity * song_genre_strength
        
        genre_influences.append({
            'genre_idx': genre_id,
            'user_affinity': user_genre_affinity,
            'song_strength': song_genre_strength,
            'contribution': influence_score
        })
    
    # Sort by influence strength
    genre_influences.sort(key=lambda x: x['contribution'], reverse=True)
    
    return {
        'num_common_genres': len(common_genres),
        'total_genres_user': len(user_genres),
        'total_genres_song': len(song_genres),
        'genre_details': genre_influences,
        'total_influence': sum(gi['contribution'] for gi in genre_influences)
    }
```

## Training Integration

### Model Selection

The enhanced model integrates with the existing trainer architecture:

```python
# In trainers/base_trainer.py
def _create_model(self):
    """Create model based on configuration."""
    
    model_config = self.model_config.copy()
    use_enhanced = model_config.pop("use_enhanced", False)
    
    if use_enhanced:
        from src.models.enhanced_gat_recommender import EnhancedGATRecommender
        model = EnhancedGATRecommender(**model_config)
    else:
        from src.models.gat_recommender import GATRecommender  
        model = GATRecommender(**model_config)
    
    return model
```

### Training Configuration

```python
# Training with enhanced model
model_config = {
    'num_users': graph['user'].num_nodes,
    'num_songs': graph['song'].num_nodes, 
    'num_artists': graph['artist'].num_nodes,
    'num_genres': graph['genre'].num_nodes,  # Only if genres available
    'embedding_dim': 64,
    'hidden_dim': 64,
    'num_layers': 2,
    'heads': 4,
    'dropout': 0.1,
    'use_enhanced': True,  # Triggers enhanced model
    'genre_weight': 0.1    # Weight for genre influence
}
```

## Performance Characteristics

### Model Comparison

| Feature | Basic GAT | Enhanced GAT |
|---------|-----------|--------------|
| **Parameters** | ~206K | ~500K+ |
| **Node Types** | 3 (user, song, artist) | 4 (+ genre) |
| **Layers** | 1 | 2+ (configurable) |
| **Attention Heads** | 4 | 4+ (configurable) |
| **Training Time** | ~2 min | ~5-10 min |
| **Memory Usage** | Low | Medium |
| **Explainability** | Limited | Comprehensive |
| **Cold Start** | Weak | Strong (genre-based) |

### Scalability

```python
# Memory usage scales with:
# - Number of genres (embedding table)
# - Number of attention heads
# - Number of layers
# - Graph size (edge counts)

def estimate_memory_usage(num_users, num_songs, num_artists, num_genres,
                         embedding_dim=64, num_layers=2, heads=4):
    """Estimate model memory usage in MB."""
    
    # Embedding tables
    embeddings_params = (
        num_users + num_songs + num_artists + num_genres
    ) * embedding_dim
    
    # Attention layers  
    attention_params = num_layers * heads * embedding_dim * embedding_dim * 3
    
    # Output projection
    output_params = embedding_dim * heads * embedding_dim
    
    total_params = embeddings_params + attention_params + output_params
    memory_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    return memory_mb
```

## Usage Examples

### Basic Training

```python
# Train enhanced model
python -m src.train_improved --epochs 50 --use-enhanced

# Or via configuration
config = {
    'model': {
        'use_enhanced': True,
        'num_layers': 3,
        'heads': 8,
        'genre_weight': 0.15
    }
}
```

### Testing and Evaluation

```python
# Test enhanced model with explainability
python scripts/test_enhanced_model.py --user 0 --top-k 5 --verbose

# Expected output:
# - Model statistics (parameters, genre support)
# - Forward pass validation
# - Top-K recommendations with scores
# - Detailed explanations for each recommendation
# - Genre influence analysis
# - Comprehensive evaluation metrics
```

### Custom Model Configuration

```python
# Create custom enhanced model
model = EnhancedGATRecommender(
    num_users=1000,
    num_songs=5000, 
    num_artists=1500,
    num_genres=35,
    embedding_dim=128,      # Larger embeddings
    hidden_dim=128,
    num_layers=3,           # Deeper architecture
    heads=8,                # More attention heads
    dropout=0.2,            # Higher dropout for regularization
    genre_weight=0.2        # Stronger genre influence
)
```

## Advanced Features

### Multi-Scale Attention

The enhanced model supports attention at multiple scales:

```python
def _multi_scale_attention(self, embeddings, edge_index_dict):
    """Apply attention at different scales."""
    
    # Local attention (direct connections)
    local_attn = self._compute_local_attention(embeddings, edge_index_dict)
    
    # Global attention (via genres)
    global_attn = self._compute_global_attention(embeddings, edge_index_dict)
    
    # Combine with learned weights
    combined = self.local_weight * local_attn + self.global_weight * global_attn
    
    return combined
```

### Dynamic Genre Weighting

Adapt genre influence based on user characteristics:

```python
def _adaptive_genre_weight(self, user_id, graph):
    """Compute adaptive genre weight based on user profile."""
    
    # New users rely more on genres (cold start)
    user_activity = self._get_user_activity_level(user_id, graph)
    
    # Users with diverse tastes get higher genre weight
    user_diversity = self._get_user_genre_diversity(user_id, graph)
    
    # Combine factors
    adaptive_weight = self.base_genre_weight * (
        1.0 + 0.5 * (1.0 - user_activity) +  # More weight for new users
        0.3 * user_diversity                   # More weight for diverse users
    )
    
    return min(adaptive_weight, 0.5)  # Cap at 50%
```

## Future Enhancements

### Planned Features

1. **Hierarchical Genres**: Support for genre taxonomies (e.g., Rock → Alternative Rock)
2. **Temporal Evolution**: Track how genre preferences change over time
3. **Cross-Modal Features**: Incorporate audio features alongside genres
4. **Dynamic Architectures**: Automatically adapt model complexity to data size

### Research Directions

1. **Genre Discovery**: Automatically discover new genre clusters
2. **Compositional Reasoning**: Understand genre combinations and fusion
3. **Cultural Context**: Incorporate geographical and cultural genre variations
4. **Causal Modeling**: Move from correlation to causal understanding of preferences

## Related Files

- `src/models/enhanced_gat_recommender.py` - Enhanced model implementation
- `src/models/gat_recommender.py` - Basic model for comparison
- `src/trainers/base_trainer.py` - Training infrastructure supporting both models
- `scripts/test_enhanced_model.py` - Testing and demonstration script
- `src/explainability.py` - External explainability framework
- `src/metrics_extended.py` - Enhanced evaluation metrics