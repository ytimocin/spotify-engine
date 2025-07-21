# Data Directory

This directory contains the synthetic music listening data used for training the graph-based recommendation model.

## Data Generation

To generate synthetic data, run:

```bash
python scripts/generate_synthetic_data.py --users 1000 --songs 5000 --artists 500
```

This creates eight CSV files:

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

### 5. `synthetic_genres.csv` (Metadata)

Genre definitions:

- `genre_id`: Unique genre identifier (e.g., G001)
- `genre_name`: Genre name (e.g., Pop, Rock, Hip Hop)
- `popularity`: Genre popularity score (0-1)

### 6. `synthetic_artist_genres.csv` (Relationships)

Artist-to-genre mappings:

- `artist_id`: Artist identifier
- `genre_id`: Genre identifier
- `genre_name`: Genre name (denormalized)

Each artist has 1-3 genres, with popular artists more likely to be in popular genres.

### 7. `synthetic_song_genres.csv` (Relationships)

Song-to-genre mappings (inherited from artists):

- `track_id`: Track identifier
- `genre_id`: Genre identifier
- `genre_name`: Genre name (denormalized)

### 8. `synthetic_user_genre_preferences.csv` (Preferences)

User genre preferences with affinity scores:

- `user_id`: User identifier
- `genre_id`: Genre identifier
- `genre_name`: Genre name (denormalized)
- `affinity_score`: How much the user likes this genre (0-1)

User preferences vary by type:
- Casual users: 1-3 focused genres
- Regular users: 2-5 genres with moderate diversity
- Power users: 3-8 genres, more exploration

## Data Characteristics

The synthetic data mimics real music listening patterns:

1. **Power-law distribution**: Popular songs/artists get exponentially more plays
2. **User preferences**: Each user has preferred artists they listen to more frequently
3. **Genre-based preferences**: Users listen to songs based on their genre affinities
4. **Listening behaviors**: Mix of full plays, skips, and partial listens
5. **Time patterns**: Sessions distributed across different times of day
6. **Realistic durations**: Songs range from 30 seconds to 10 minutes

## Data Quality Validation

The data generator includes built-in validation to ensure data quality:

- **Session validation**: Checks that ms_played ≤ track_duration_ms
- **Reference integrity**: Ensures all track_ids and user_ids exist
- **Graph connectivity**: Validates that all users have sessions and most songs are played
- **Genre coverage**: Ensures all genres are used and artists have appropriate genres

Quality metrics tracked:
- Skip rate (songs played < 30 seconds)
- Completion rate (songs played ≥ 80% of duration)
- Average songs per listening session
- User listening diversity (unique songs/artists per user)

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
