# Future Enhancement: Genre Features

## Overview

Adding genre information would enable content-based filtering to complement collaborative filtering.

## Implementation Approach

### 1. Data Model Extension

```python
# In generate_artists()
genres = ['pop', 'rock', 'hip-hop', 'electronic', 'jazz', 'classical', 'country', 'r&b']
artist_genres = random.sample(genres, k=random.randint(1, 3))
```

### 2. Graph Structure

Add genre nodes as a fourth node type:

- Artist → Genre edges
- Song → Genre edges (inherited from artist)

### 3. Model Enhancement

- Add genre embeddings
- Include genre-based attention heads
- Combine collaborative and content signals

## Benefits

1. **Cold Start**: Better recommendations for new songs/artists
2. **Diversity**: Genre-aware diversification
3. **Explanations**: "Recommended because you like jazz"

## Trade-offs

- Increased model complexity
- More parameters to train
- Risk of overfitting on sparse genres
