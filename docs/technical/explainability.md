# Explainability System

The spotify-engine provides comprehensive explainability for music recommendations through attention analysis and genre influence scoring.

## Overview

The explainability system helps users understand **why** specific songs were recommended by analyzing:
- **Attention weights** from the Graph Attention Network layers
- **Genre influence** based on user preferences and song genres
- **Artist similarity** through graph connectivity
- **Listening pattern analysis** based on historical behavior

## Architecture

### Core Components

1. **RecommendationExplainer** (`src/explainability.py`)
   - Main interface for generating explanations
   - Combines multiple explanation signals
   - Formats user-friendly explanations

2. **Enhanced GAT Model** (`src/models/enhanced_gat_recommender.py`)
   - Built-in explanation methods
   - Attention weight extraction
   - Genre influence computation

3. **Attention Visualization** (`src/visualization/attention_viz.py`)
   - Visual analysis of attention patterns
   - Genre attention heatmaps
   - Cross-attention analysis

## Explanation Types

### 1. Attention-Based Explanations

The GAT model uses attention mechanisms to weight the importance of different nodes and relationships:

```python
# Extract attention weights for a user-song pair
attention_weights = model.get_attention_weights(user_id, song_id, graph)

# Attention weights show:
# - Which artists influenced the recommendation
# - Which genres were most important
# - How strongly user preferences align with song features
```

**Example Output:**
```
Attention Analysis:
- Artist "The Beatles": 0.24 attention weight
- Genre "Rock": 0.31 attention weight  
- User-Artist connection: 0.18 weight
```

### 2. Genre Influence Analysis

Analyzes how user genre preferences contribute to recommendations:

```python
def explain_recommendation(self, user_id, song_id, x_dict, graph):
    # Calculate genre overlap between user preferences and song
    common_genres = self._find_common_genres(user_id, song_id, graph)
    
    # Score genre influence based on:
    # - User's affinity for each genre
    # - Song's association with genres
    # - Genre popularity and rarity
    
    genre_scores = []
    for genre_id in common_genres:
        influence = self._calculate_genre_influence(
            user_id, song_id, genre_id, graph
        )
        genre_scores.append({
            'genre_idx': genre_id,
            'contribution': influence,
            'user_affinity': user_genre_affinity[genre_id],
            'genre_rarity': 1.0 / genre_popularity[genre_id]
        })
```

### 3. Artist Similarity Explanations

Explains recommendations based on artist relationships:

```python
def _explain_artist_influence(self, user_id, song_id, graph):
    # Find artists user has listened to
    user_artists = graph['user', 'listens_to', 'song']['edge_index'][1]
    
    # Calculate similarity between recommendation artist and user's artists
    artist_similarities = cosine_similarity(
        artist_embeddings[song_artist], 
        artist_embeddings[user_artists]
    )
    
    return {
        'artist_idx': song_artist,
        'similarity': max_similarity,
        'similar_to': most_similar_user_artist
    }
```

## Usage Examples

### Basic Explanation

```python
from src.explainability import RecommendationExplainer

# Initialize explainer
explainer = RecommendationExplainer(graph, model)

# Get explanation for a recommendation
explanation = explainer.explain_recommendation(user_id=42, song_id=1337)

# Format for display
formatted = format_explanation(explanation, verbose=True)
print(formatted)
```

**Output:**
```
Recommendation Explanation for Song 1337:

🎵 Primary Reasons:
• Genre Match: You enjoy "Rock" music (affinity: 0.85)
  This song is 78% associated with Rock
• Artist Similarity: Similar to "Led Zeppelin" in your listening history
  Similarity score: 0.72

🎯 Attention Analysis:
• Genre "Rock" received 31% of model attention
• Artist connection weighted at 24%
• Your listening pattern match: 18%

📊 Recommendation Score: 0.84/1.0
```

### Programmatic Analysis

```python
# Analyze multiple recommendations
user_recs = model.recommend(user_id=42, k=10)

explanations = []
for song_id, score in user_recs:
    explanation = explainer.explain_recommendation(42, song_id)
    explanations.append({
        'song_id': song_id,
        'score': score,
        'explanation': explanation
    })

# Find most genre-influenced recommendations
genre_influenced = [
    exp for exp in explanations 
    if exp['explanation']['genre_influence_score'] > 0.5
]
```

### Testing Enhanced Model

```python
# Test script usage
python scripts/test_enhanced_model.py --user 0 --top-k 5 --verbose

# Expected output includes:
# - Top-K recommendations with scores
# - Explanation for each recommendation
# - Genre influence analysis
# - Attention weight visualization
```

## Explanation Quality Metrics

The system includes metrics to evaluate explanation quality:

### 1. Genre Coherence
Measures how well genre explanations align with user preferences:

```python
def calculate_genre_coherence(user_id, explanations):
    user_genre_prefs = get_user_genre_preferences(user_id)
    
    coherence_scores = []
    for exp in explanations:
        # Check if explained genres match user's top preferences
        explained_genres = exp['genre_influence']
        overlap = len(set(explained_genres) & set(user_genre_prefs))
        coherence = overlap / len(explained_genres)
        coherence_scores.append(coherence)
    
    return np.mean(coherence_scores)
```

### 2. Attention Consistency
Ensures attention weights are meaningful and consistent:

```python
def validate_attention_weights(attention_weights):
    # Check that weights sum to approximately 1.0
    weight_sums = attention_weights.sum(dim=-1)
    
    # Check for reasonable distribution (not too peaked)
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8))
    
    return {
        'sum_validity': torch.allclose(weight_sums, torch.ones_like(weight_sums)),
        'entropy': entropy.item(),
        'distribution_health': entropy > 0.5  # Minimum entropy threshold
    }
```

## Implementation Details

### Memory Efficiency

The explainer uses efficient computation to handle large graphs:

```python
class RecommendationExplainer:
    def __init__(self, graph, model, cache_size=1000):
        self.graph = graph
        self.model = model
        self.attention_cache = LRUCache(cache_size)
        self.genre_cache = LRUCache(cache_size)
```

### Batch Explanations

For explaining multiple recommendations efficiently:

```python
def batch_explain(self, user_ids, song_ids, top_k=5):
    # Batch compute embeddings
    user_embeddings = self.model.get_user_embeddings(user_ids)
    song_embeddings = self.model.get_song_embeddings(song_ids)
    
    # Vectorized attention computation
    attention_scores = self._batch_attention_analysis(
        user_embeddings, song_embeddings
    )
    
    # Parallel genre influence computation
    genre_influences = self._batch_genre_analysis(user_ids, song_ids)
    
    return self._format_batch_explanations(
        attention_scores, genre_influences
    )
```

## Configuration

Explainability behavior can be customized:

```python
explainer_config = {
    'attention_threshold': 0.1,      # Minimum attention weight to include
    'genre_influence_threshold': 0.05,  # Minimum genre influence to report
    'max_genres_per_explanation': 3,    # Limit explanation complexity
    'include_negative_evidence': False, # Show why songs were NOT recommended
    'explanation_style': 'detailed'     # 'brief', 'detailed', or 'technical'
}

explainer = RecommendationExplainer(
    graph, model, config=explainer_config
)
```

## Future Enhancements

### Planned Features

1. **Counterfactual Explanations**: "If you listened to more jazz, you'd get this recommendation"
2. **Temporal Explanations**: "Recommended because you listened to similar songs yesterday"
3. **Social Explanations**: "Users with similar taste also enjoyed this"
4. **Interactive Explanations**: Allow users to probe deeper into specific aspects

### Research Directions

1. **Causal Inference**: Move beyond correlation to understand causal relationships
2. **User-Specific Explanations**: Adapt explanation style to user preferences
3. **Explanation Feedback**: Learn from user feedback on explanation quality
4. **Multi-Modal Explanations**: Include audio features in explanations

## Related Files

- `src/explainability.py` - Core explainability implementation
- `src/models/enhanced_gat_recommender.py` - Model with built-in explanation methods
- `src/visualization/attention_viz.py` - Attention visualization tools
- `scripts/test_enhanced_model.py` - Testing and demonstration script
- `src/metrics_extended.py` - Extended metrics including explanation quality