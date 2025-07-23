# Kaggle Spotify Dataset

This directory should contain the Kaggle Spotify playlist dataset CSV files. The files are not included in the repository due to their large size.

## Required Files

1. **spotify_dataset.csv** (~1.2 GB)
   - Contains playlist membership data
   - Columns: `user_id`, `artistname`, `trackname`, `playlistname`
   - ~12.9M rows

2. **tracks_features.csv** (~346 MB)
   - Contains audio features for tracks
   - Columns: track metadata and audio features (danceability, energy, etc.)
   - ~1.2M rows

## Download Instructions

1. Download the datasets from Kaggle (you'll need a Kaggle account):
   - Playlists: https://www.kaggle.com/api/v1/datasets/download/andrewmvd/spotify-playlists
   - Track Features: https://www.kaggle.com/api/v1/datasets/download/rodolfofigueroa/spotify-12m-songs

2. Place both CSV files in this directory:
   ```
   data/kaggle/
   ├── spotify_dataset.csv
   ├── tracks_features.csv
   └── README.md (this file)
   ```

## Validation

After downloading, verify the files:

```bash
# Check file sizes
ls -lh data/kaggle/*.csv

# Expected output:
# -rw-r--r--  1 user  group  1.1G  spotify_dataset.csv
# -rw-r--r--  1 user  group  330M  tracks_features.csv

# Check row counts
wc -l data/kaggle/*.csv

# Expected output:
# 12891571 spotify_dataset.csv
# 1204026  tracks_features.csv
```

## Data Format

### spotify_dataset.csv
```csv
"user_id", "artistname", "trackname", "playlistname"
"9cc0cfd4d7d7885102480dd99e7a90d6","Elvis Costello","(The Angels Wanna Wear My) Red Shoes","HARD ROCK 2010"
...
```

### tracks_features.csv
```csv
id,name,album,album_id,artists,artist_ids,track_number,disc_number,explicit,danceability,energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness,valence,tempo,duration_ms,time_signature,year,release_date
...
```

## Notes

- The `artists` column in tracks_features.csv contains Python list strings like `['Artist Name']`
- Some track/artist names contain special characters and quotes
- Match rate between the two datasets is approximately 25% due to naming variations