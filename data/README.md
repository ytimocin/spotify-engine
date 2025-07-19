# Data Directory

This directory contains the synthetic music listening data used for training the graph-based recommendation model.

## Data Generation

To generate synthetic data, run:

```bash
python scripts/generate_synthetic_data.py --users 1000 --songs 5000 --artists 500
```

This creates four CSV files:

### 1. `synthetic_sessions.csv` (Main file)

The primary listening session data with columns:

- `user_id`: Unique user identifier (e.g., U0001)
- `track_id`: Unique song identifier (e.g., T00123)
- `artist_id`: Artist identifier (e.g., A0042)
- `timestamp`: ISO format timestamp of when the song was played
- `ms_played`: Milliseconds the user listened to the song
- `track_duration_ms`: Total duration of the track in milliseconds

### 2. `synthetic_artists.csv` (Metadata)

Artist information:

- `artist_id`: Unique artist identifier
- `artist_name`: Artist name
- `popularity`: Popularity score (0-1, following power-law distribution)

### 3. `synthetic_songs.csv` (Metadata)

Song information:

- `track_id`: Unique track identifier
- `track_name`: Track name
- `artist_id`: Associated artist
- `artist_name`: Artist name (denormalized for convenience)
- `duration_ms`: Track duration
- `popularity`: Song popularity (0-1, correlated with artist popularity)

### 4. `synthetic_users.csv` (Metadata)

User profiles:

- `user_id`: Unique user identifier
- `user_type`: User category (casual/regular/power)
- `activity_level`: Activity multiplier (0-1)

## Data Characteristics

The synthetic data mimics real music listening patterns:

1. **Power-law distribution**: Popular songs/artists get exponentially more plays
2. **User preferences**: Each user has preferred artists they listen to more frequently
3. **Listening behaviors**: Mix of full plays, skips, and partial listens
4. **Time patterns**: Sessions distributed across different times of day
5. **Realistic durations**: Songs range from 30 seconds to 10 minutes

## Data Processing Pipeline

```text
synthetic_sessions.csv 
    ↓ (prepare_mssd.py)
edge_list.parquet (aggregated user-song interactions)
    ↓ (build_graph.py)
graph.pt (PyTorch Geometric HeteroData object)
```

## Alternative: Real Data

If you prefer to use real data, the mini-MSSD dataset can be obtained from:

- Kaggle: Search for "Spotify Million Playlist Dataset" (requires account)
- Academic sources with proper licensing

Note: For interview purposes, synthetic data is recommended to avoid licensing concerns.
