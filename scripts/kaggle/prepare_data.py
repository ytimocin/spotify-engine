"""
Prepare Kaggle playlist data for graph construction.

This script loads the raw Kaggle CSV files and creates structured data
for the playlist-based recommendation system.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_kaggle_data(kaggle_dir: Path):
    """Load raw Kaggle CSV files."""
    logger.info("Loading Kaggle CSV files...")

    # Load playlist data
    # Check if sample file exists for testing
    sample_path = kaggle_dir / "spotify_dataset_sample.csv"
    if sample_path.exists():
        playlist_path = sample_path
        logger.info("Using sample file for testing")
    else:
        playlist_path = kaggle_dir / "spotify_dataset.csv"
    # Handle potential quoting issues in the CSV
    # The CSV has embedded quotes in some track names
    playlists_df = pd.read_csv(
        playlist_path,
        skipinitialspace=True,  # Skip spaces after delimiter
        encoding="utf-8",
        escapechar="\\",  # Use backslash as escape character
        on_bad_lines="skip",
    )  # Skip lines that can't be parsed

    # Fix column names
    playlists_df.columns = playlists_df.columns.str.strip().str.strip('"')

    # Load track features
    tracks_path = kaggle_dir / "tracks_features.csv"
    tracks_df = pd.read_csv(tracks_path)

    logger.info(f"Loaded {len(playlists_df):,} playlist entries")
    logger.info(f"Loaded {len(tracks_df):,} tracks with features")

    return playlists_df, tracks_df


def create_playlist_entities(playlists_df: pd.DataFrame):
    """Create unique playlist entities."""
    logger.info("Creating playlist entities...")

    # Create unique playlist ID from user_id + playlist_name
    playlists_df["playlist_id"] = (
        playlists_df["user_id"].astype(str) + "_" + playlists_df["playlistname"].astype(str)
    )

    # Get unique playlists
    playlists = (
        playlists_df.groupby("playlist_id")
        .agg({"user_id": "first", "playlistname": "first", "trackname": "count"})
        .reset_index()
    )

    playlists.columns = ["playlist_id", "owner_id", "playlist_name", "track_count"]

    logger.info(f"Created {len(playlists):,} unique playlists")
    return playlists


def match_tracks(playlists_df: pd.DataFrame, tracks_df: pd.DataFrame):
    """Match playlist tracks with track features."""
    logger.info("Matching tracks between datasets...")

    # Parse the artist list format in tracks_df
    import ast

    def parse_artists(artist_str):
        """Parse artist string that's in Python list format."""
        try:
            if pd.isna(artist_str) or artist_str == "":
                return ""
            # Parse the list and get the first artist
            artists_list = ast.literal_eval(artist_str)
            if isinstance(artists_list, list) and len(artists_list) > 0:
                # Join multiple artists with ' & ' to match common patterns
                return " & ".join(artists_list)
            return ""
        except (ValueError, SyntaxError):
            # If parsing fails, return the original string
            return str(artist_str)

    tracks_df["artists_parsed"] = tracks_df["artists"].apply(parse_artists)

    # Create track lookup key
    playlists_df["track_key"] = (
        playlists_df["trackname"].str.lower().str.strip()
        + "_"
        + playlists_df["artistname"].str.lower().str.strip()
    )

    tracks_df["track_key"] = (
        tracks_df["name"].str.lower().str.strip()
        + "_"
        + tracks_df["artists_parsed"].str.lower().str.strip()
    )

    # Match tracks
    # Select available columns from tracks_df
    feature_cols = [
        "id",
        "track_key",
        "name",
        "artists",
        "artists_parsed",
        "danceability",
        "energy",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "valence",
        "tempo",
        "duration_ms",
    ]

    # Add popularity if it exists
    if "popularity" in tracks_df.columns:
        feature_cols.append("popularity")

    matched = playlists_df.merge(tracks_df[feature_cols], on="track_key", how="inner")

    match_rate = len(matched) / len(playlists_df) * 100
    logger.info(f"Matched {len(matched):,}/{len(playlists_df):,} entries ({match_rate:.1f}%)")

    # Show some examples of matches and non-matches for debugging
    if len(matched) > 0:
        logger.info("Sample matches:")
        sample = matched[["trackname", "artistname", "name", "artists_parsed"]].head(3)
        for _, row in sample.iterrows():
            logger.info(f"  Playlist: '{row['artistname']}' - '{row['trackname']}'")
            logger.info(f"  Features: '{row['artists_parsed']}' - '{row['name']}'")

    # Show some unmatched examples
    unmatched = playlists_df[~playlists_df["track_key"].isin(matched["track_key"])]
    if len(unmatched) > 0:
        logger.info("Sample non-matches from playlist:")
        for _, row in unmatched.head(3).iterrows():
            logger.info(f"  '{row['artistname']}' - '{row['trackname']}' (key: {row['track_key']})")

    return matched


def extract_artists(matched_df: pd.DataFrame):
    """Extract unique artists from matched data."""
    logger.info("Extracting artists...")

    # Get unique artists using the parsed artist names
    artists = matched_df[["artists_parsed"]].drop_duplicates().reset_index(drop=True)
    artists["artist_id"] = "A" + artists.index.astype(str).str.zfill(6)
    artists.columns = ["artist_name", "artist_id"]

    # Add artist_id back to matched data
    matched_df = matched_df.merge(artists, left_on="artists_parsed", right_on="artist_name")

    logger.info(f"Extracted {len(artists):,} unique artists")
    return matched_df, artists


def infer_genres(tracks_df: pd.DataFrame):
    """Infer genres from audio features using simple rules."""
    logger.info("Inferring genres from audio features...")

    # Define genre mapping rules
    def map_to_genre(row):
        if row["danceability"] > 0.7 and row["energy"] > 0.7:
            return "Electronic"
        elif row["acousticness"] > 0.8:
            return "Folk" if row["energy"] < 0.5 else "Acoustic Rock"
        elif row["speechiness"] > 0.3:
            return "Hip Hop"
        elif row["instrumentalness"] > 0.8:
            return "Classical" if row["acousticness"] > 0.7 else "Electronic"
        elif row["energy"] > 0.8 and row["acousticness"] < 0.3:
            return "Metal" if row["tempo"] > 140 else "Rock"
        elif row["valence"] > 0.7 and row["danceability"] > 0.6:
            return "Pop"
        elif row["valence"] < 0.3 and row["acousticness"] > 0.5:
            return "Blues"
        elif row["danceability"] > 0.6 and row["tempo"] > 120:
            return "Dance"
        else:
            return "Pop"  # Default

    # Apply genre mapping
    tracks_df["genre"] = tracks_df.apply(map_to_genre, axis=1)

    # Create genre entities
    genres = pd.DataFrame({"genre_name": tracks_df["genre"].unique()})
    genres["genre_id"] = "G" + genres.index.astype(str).str.zfill(3)

    # Add genre_id to tracks
    tracks_df = tracks_df.merge(genres, left_on="genre", right_on="genre_name")

    logger.info(f"Inferred {len(genres)} genres")
    return tracks_df, genres


def create_playlist_tracks(matched_df: pd.DataFrame):
    """Create playlist-track relationships."""
    logger.info("Creating playlist-track relationships...")

    # Add position in playlist (order by appearance)
    matched_df["position"] = matched_df.groupby("playlist_id").cumcount()

    playlist_tracks = matched_df[["playlist_id", "id", "position"]].copy()
    playlist_tracks.columns = ["playlist_id", "track_id", "position"]

    logger.info(f"Created {len(playlist_tracks):,} playlist-track relationships")
    return playlist_tracks


def save_processed_data(
    output_dir: Path,
    playlists,
    tracks,
    artists,
    genres,
    playlist_tracks,
    track_genres,
    artist_genres,
):
    """Save all processed data to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {output_dir}")

    # Save entities
    playlists.to_parquet(output_dir / "playlists.parquet", index=False)
    tracks.to_parquet(output_dir / "tracks.parquet", index=False)
    artists.to_parquet(output_dir / "artists.parquet", index=False)
    genres.to_parquet(output_dir / "genres.parquet", index=False)

    # Save relationships
    playlist_tracks.to_parquet(output_dir / "playlist_tracks.parquet", index=False)
    track_genres.to_parquet(output_dir / "track_genres.parquet", index=False)
    artist_genres.to_parquet(output_dir / "artist_genres.parquet", index=False)

    logger.info("Data preparation complete!")


def main():
    """Main function to prepare Kaggle playlist data."""
    parser = argparse.ArgumentParser(description="Prepare Kaggle playlist data")
    parser.add_argument(
        "--kaggle-dir",
        type=str,
        default="data/kaggle",
        help="Directory containing Kaggle CSV files",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/kaggle", help="Output directory for processed data"
    )
    parser.add_argument("--sample", type=int, help="Sample N playlists for testing")

    args = parser.parse_args()

    # Load data
    playlists_df, tracks_df = load_kaggle_data(Path(args.kaggle_dir))

    # Sample if requested
    if args.sample:
        logger.info(f"Sampling {args.sample} playlists...")
        unique_playlists = playlists_df["playlistname"].unique()
        sampled = np.random.choice(
            unique_playlists, size=min(args.sample, len(unique_playlists)), replace=False
        )
        playlists_df = playlists_df[playlists_df["playlistname"].isin(sampled)]

    # Process data
    playlists = create_playlist_entities(playlists_df)
    matched_df = match_tracks(playlists_df, tracks_df)
    matched_df, artists = extract_artists(matched_df)

    # Get unique tracks with features
    track_cols = [
        "id",
        "name",
        "artist_id",
        "duration_ms",
        "danceability",
        "energy",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "valence",
        "tempo",
    ]

    # Add popularity if it exists
    if "popularity" in matched_df.columns:
        track_cols.insert(4, "popularity")  # Insert after duration_ms

    tracks = matched_df[track_cols].drop_duplicates("id")

    # Rename columns
    new_col_names = ["track_id", "track_name", "artist_id", "duration_ms"]
    if "popularity" in matched_df.columns:
        new_col_names.append("popularity")
    new_col_names.extend(
        [
            "danceability",
            "energy",
            "acousticness",
            "speechiness",
            "instrumentalness",
            "valence",
            "tempo",
        ]
    )

    tracks.columns = new_col_names

    # Infer genres
    tracks, genres = infer_genres(tracks)

    # Create relationships
    playlist_tracks = create_playlist_tracks(matched_df)

    # Create genre relationships
    track_genres = tracks[["track_id", "genre_id"]].copy()

    # Simple artist-genre mapping (most common genre for their tracks)
    artist_genres = tracks.groupby(["artist_id", "genre_id"]).size().reset_index(name="count")
    artist_genres = artist_genres.loc[artist_genres.groupby("artist_id")["count"].idxmax()]
    artist_genres = artist_genres[["artist_id", "genre_id"]]

    # Save data
    save_processed_data(
        Path(args.output_dir),
        playlists,
        tracks,
        artists,
        genres,
        playlist_tracks,
        track_genres,
        artist_genres,
    )

    # Print summary
    print("\nData Summary:")
    print(f"  Playlists: {len(playlists):,}")
    print(f"  Tracks: {len(tracks):,}")
    print(f"  Artists: {len(artists):,}")
    print(f"  Genres: {len(genres):,}")
    print(f"  Playlist-Track edges: {len(playlist_tracks):,}")


if __name__ == "__main__":
    main()
