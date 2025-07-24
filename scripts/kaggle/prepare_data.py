"""
Simplified data preparation for Kaggle playlist data.

This script loads the raw Kaggle CSV files and creates structured data
for the playlist-based recommendation system without any synthetic features.
"""

import argparse
import ast
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_artist_list(artist_str):
    """Parse artist string that's in Python list format."""
    try:
        if pd.isna(artist_str) or artist_str == "":
            return ""
        artists_list = ast.literal_eval(artist_str)
        if isinstance(artists_list, list) and len(artists_list) > 0:
            # Take first artist for simplicity
            return artists_list[0]
        return ""
    except (ValueError, SyntaxError):
        return str(artist_str)


def load_and_match_data(playlist_path: Path, tracks_path: Path, sample_size: int = None):
    """Load and match playlist and track data."""
    logger.info("Loading CSV files...")

    # Load playlist data
    playlists_df = pd.read_csv(
        playlist_path,
        skipinitialspace=True,
        encoding="utf-8",
        escapechar="\\",
        on_bad_lines="skip",
    )

    # Fix column names
    playlists_df.columns = playlists_df.columns.str.strip().str.strip('"')

    # Load track features
    tracks_df = pd.read_csv(tracks_path)

    logger.info(f"Loaded {len(playlists_df):,} playlist entries")
    logger.info(f"Loaded {len(tracks_df):,} tracks with features")

    # Sample if requested
    if sample_size:
        logger.info(f"Sampling {sample_size} playlists...")
        unique_playlists = playlists_df["playlistname"].unique()
        sampled = pd.DataFrame(unique_playlists[:sample_size], columns=["playlistname"])
        playlists_df = playlists_df.merge(sampled, on="playlistname")

    # Parse artist names
    tracks_df["artist_name"] = tracks_df["artists"].apply(parse_artist_list)

    # Create normalized matching keys
    playlists_df["track_key"] = (
        playlists_df["trackname"].str.lower().str.strip()
        + "_"
        + playlists_df["artistname"].str.lower().str.strip()
    )

    tracks_df["track_key"] = (
        tracks_df["name"].str.lower().str.strip()
        + "_"
        + tracks_df["artist_name"].str.lower().str.strip()
    )

    # Match tracks
    matched = playlists_df.merge(tracks_df, on="track_key", how="inner")

    match_rate = len(matched) / len(playlists_df) * 100
    logger.info(f"Matched {len(matched):,}/{len(playlists_df):,} entries ({match_rate:.1f}%)")

    return matched


def extract_entities(matched_df: pd.DataFrame):
    """Extract entity dataframes from matched data."""
    logger.info("Extracting entities...")

    # Create unique playlist ID
    matched_df["playlist_id"] = (
        matched_df["user_id"].astype(str) + "_" + matched_df["playlistname"].astype(str)
    )

    # Extract playlists
    playlists = (
        matched_df[["playlist_id", "user_id", "playlistname"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    playlists.columns = ["playlist_id", "owner_id", "playlist_name"]

    # Extract tracks with features
    audio_features = [
        "danceability",
        "energy",
        "valence",
        "tempo",
        "acousticness",
        "speechiness",
        "instrumentalness",
    ]

    # Check which optional features exist
    optional_features = ["key", "mode", "loudness", "time_signature", "duration_ms"]
    available_features = [f for f in optional_features if f in matched_df.columns]

    track_columns = ["id", "name", "album_id"] + audio_features + available_features
    tracks = matched_df[track_columns].drop_duplicates(subset=["id"]).reset_index(drop=True)
    tracks.rename(columns={"id": "track_id", "name": "track_name"}, inplace=True)
    
    # Validate and clean audio features
    for feature in audio_features + available_features:
        if feature in tracks.columns:
            # Fill NaN values with median to prevent training issues
            median_val = tracks[feature].median()
            tracks[feature] = tracks[feature].fillna(median_val)
            # Clip normalized features to valid [0, 1] range
            if feature in ["danceability", "energy", "valence", "acousticness", 
                          "speechiness", "instrumentalness", "mode"]:
                tracks[feature] = tracks[feature].clip(0, 1)

    # Extract artists
    artists = matched_df[["artist_ids", "artist_name"]].drop_duplicates().reset_index(drop=True)
    # Use first artist ID from the list
    artists["artist_id"] = artists["artist_ids"].apply(
        lambda x: ast.literal_eval(x)[0] if x and x != "[]" else "unknown"
    )
    artists = artists[["artist_id", "artist_name"]].drop_duplicates()

    # Extract albums
    albums = matched_df[["album_id", "album"]].drop_duplicates().reset_index(drop=True)
    albums.columns = ["album_id", "album_name"]

    logger.info(
        f"Extracted {len(playlists):,} playlists, {len(tracks):,} tracks, {len(artists):,} artists, {len(albums):,} albums"
    )

    return playlists, tracks, artists, albums, matched_df


def create_relationships(matched_df: pd.DataFrame, tracks: pd.DataFrame):
    """Create relationship dataframes."""
    logger.info("Creating relationships...")

    # Playlist-Track relationships
    playlist_tracks = (
        matched_df[["playlist_id", "id"]].rename(columns={"id": "track_id"}).drop_duplicates()
    )

    # Add position in playlist
    playlist_tracks["position"] = playlist_tracks.groupby("playlist_id").cumcount()
    
    # Add normalized position for edge features (0 = start, 1 = end)
    playlist_sizes = playlist_tracks.groupby("playlist_id").size()
    playlist_tracks["norm_position"] = playlist_tracks.apply(
        lambda x: x["position"] / max(1, playlist_sizes[x["playlist_id"]] - 1) 
        if playlist_sizes[x["playlist_id"]] > 1 else 0.5, 
        axis=1
    )

    # Track-Artist relationships
    # First, create a mapping from track to its primary artist
    track_artist_map = {}
    for _, row in matched_df.iterrows():
        track_id = row["id"]
        artist_ids = ast.literal_eval(row["artist_ids"]) if row["artist_ids"] else []
        if artist_ids and track_id not in track_artist_map:
            track_artist_map[track_id] = artist_ids[0]

    track_artists = pd.DataFrame(list(track_artist_map.items()), columns=["track_id", "artist_id"])

    # Track-Album relationships
    track_albums = tracks[["track_id", "album_id"]].drop_duplicates()

    logger.info(f"Created {len(playlist_tracks):,} playlist-track relationships")
    logger.info(f"Created {len(track_artists):,} track-artist relationships")
    logger.info(f"Created {len(track_albums):,} track-album relationships")

    return playlist_tracks, track_artists, track_albums


def save_processed_data(
    output_dir: Path,
    playlists,
    tracks,
    artists,
    albums,
    playlist_tracks,
    track_artists,
    track_albums,
):
    """Save all processed data to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving processed data to {output_dir}")

    # Save entities
    playlists.to_parquet(output_dir / "playlists.parquet", index=False)
    tracks.to_parquet(output_dir / "tracks.parquet", index=False)
    artists.to_parquet(output_dir / "artists.parquet", index=False)
    albums.to_parquet(output_dir / "albums.parquet", index=False)

    # Save relationships
    playlist_tracks.to_parquet(output_dir / "playlist_tracks.parquet", index=False)
    track_artists.to_parquet(output_dir / "track_artists.parquet", index=False)
    track_albums.to_parquet(output_dir / "track_albums.parquet", index=False)

    # Save metadata for tracking and debugging
    metadata = {
        "created_at": datetime.now().isoformat(),
        "version": "1.0",
        "stats": {
            "n_playlists": len(playlists),
            "n_tracks": len(tracks),
            "n_artists": len(artists),
            "n_albums": len(albums),
            "n_edges": {
                "playlist_tracks": len(playlist_tracks),
                "track_artists": len(track_artists),
                "track_albums": len(track_albums),
            }
        },
        "feature_columns": list(tracks.columns)
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Data preparation complete!")


def main():
    """Main function to prepare Kaggle playlist data."""
    parser = argparse.ArgumentParser(description="Prepare Kaggle playlist data")
    parser.add_argument(
        "--playlist-file",
        type=str,
        default="data/kaggle/spotify_dataset.csv",
        help="Path to playlist CSV file",
    )
    parser.add_argument(
        "--tracks-file",
        type=str,
        default="data/kaggle/tracks_features.csv",
        help="Path to tracks features CSV file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/kaggle/processed",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--sample",
        type=int,
        help="Number of playlists to sample for testing",
    )

    args = parser.parse_args()

    # Check if sample file exists for testing
    playlist_path = Path(args.playlist_file)
    if args.sample is None:
        sample_path = playlist_path.parent / "spotify_dataset_sample.csv"
        if sample_path.exists():
            playlist_path = sample_path
            logger.info("Using sample file for testing")

    # Load and match data
    matched_df = load_and_match_data(playlist_path, Path(args.tracks_file), args.sample)

    # Extract entities
    playlists, tracks, artists, albums, matched_df = extract_entities(matched_df)

    # Create relationships
    playlist_tracks, track_artists, track_albums = create_relationships(matched_df, tracks)

    # Save data
    save_processed_data(
        Path(args.output_dir),
        playlists,
        tracks,
        artists,
        albums,
        playlist_tracks,
        track_artists,
        track_albums,
    )

    # Print summary
    print("\nData Summary:")
    print(f"  Playlists: {len(playlists):,}")
    print(f"  Tracks: {len(tracks):,}")
    print(f"  Artists: {len(artists):,}")
    print(f"  Albums: {len(albums):,}")
    print(f"  Playlist-Track edges: {len(playlist_tracks):,}")
    print(f"  Track-Artist edges: {len(track_artists):,}")
    print(f"  Track-Album edges: {len(track_albums):,}")

    # Show available features
    feature_cols = [col for col in tracks.columns if col not in ["track_id", "track_name"]]
    print(f"\nAvailable features: {', '.join(feature_cols)}")


if __name__ == "__main__":
    main()
