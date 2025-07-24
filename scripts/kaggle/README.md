# Kaggle Pipeline Scripts

## prepare_data.py

Transforms raw Kaggle CSV files into structured parquet files with entities (playlists, tracks, artists, albums) and their relationships.

### Example Output

```text
data/kaggle/processed/
├── playlists.parquet      # Playlist metadata (playlist_id, owner_id, playlist_name)
├── tracks.parquet          # Track metadata with audio features (track_id, track_name, album_id, danceability, energy, etc.)
├── artists.parquet         # Artist metadata (artist_id, artist_name)
├── albums.parquet          # Album metadata (album_id, album_name)
├── playlist_tracks.parquet # Playlist-track relationships (playlist_id, track_id, position)
├── track_artists.parquet   # Track-artist relationships (track_id, artist_id)
├── track_albums.parquet    # Track-album relationships (track_id, album_id)
└── metadata.json           # Processing metadata with stats and timestamp
```

## build_graph.py

Constructs a PyTorch Geometric heterogeneous graph from the processed parquet files with normalized features and edge attributes.

### Example Output

```text
data/kaggle/playlist_graph.pt

Graph Summary:
  Nodes:
    playlist: 10,681 nodes, 1 features
    track: 86,568 nodes, 11 features (danceability, energy, valence, tempo, etc.)
    artist: 10,099 nodes, 1 features
    album: 18,050 nodes, 1 features
  Edges:
    ('playlist', 'contains', 'track'): 523,067 edges with position features
    ('track', 'in_playlist', 'playlist'): 523,067 edges
    ('track', 'by', 'artist'): 86,568 edges
    ('artist', 'created', 'track'): 86,568 edges
    ('track', 'from_album', 'album'): 86,568 edges
    ('album', 'contains', 'track'): 86,568 edges
```

## train.py

Trains a Graph Attention Network model for track recommendation using BPR loss with early stopping and learning rate scheduling.

### Testing Training Parameters

Two scripts are provided for testing different training configurations:

#### 1. quick_train_test.py - Manual parameter testing

```bash
# Quick test with custom parameters
python scripts/kaggle/quick_train_test.py --lr 0.01 --batch-size 128 --epochs 5

# Test with 10% of data for faster iteration
python scripts/kaggle/quick_train_test.py --sample-edges 0.1 --epochs 10 --plot

# Test different model sizes
python scripts/kaggle/quick_train_test.py --hidden-dim 32 --lr 0.001
python scripts/kaggle/quick_train_test.py --hidden-dim 128 --lr 0.1
```

#### 2. test_training_params.py - Systematic parameter search

```bash
# Quick grid search (10% data, 10 epochs)
python scripts/kaggle/test_training_params.py --max-epochs 10 --sample-ratio 0.1

# Full parameter search
python scripts/kaggle/test_training_params.py --max-epochs 15 --sample-ratio 1.0
```

This will test combinations of:

- Learning rates: [0.001, 0.01, 0.1]
- Batch sizes: [64, 128, 256]
- Hidden dimensions: [32, 64, 128]
- Patience values: [2, 3, 5]

Results are saved to a CSV file and the best configurations are displayed.
