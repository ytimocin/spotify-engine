"""
ETL pipeline to prepare music session data for graph construction.

Aggregates raw listening sessions into user-song edges with:
- play_count: how many times played
- total_ms: total milliseconds listened
- avg_completion_ratio: average listening completion
- genre_affinity_score: weighted by user's genre preferences
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def prepare_edge_data(
    sessions_df: pd.DataFrame,
    user_genre_prefs_df: pd.DataFrame = None,
    song_genres_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Aggregate sessions into user-song edges with genre affinity.

    Returns DataFrame with columns:
    - user_id, track_id, artist_id
    - play_count, total_ms_played, avg_completion_ratio
    - genre_affinity_score (if genre data provided)
    """
    # Group by user-song pairs
    edge_data = (
        sessions_df.groupby(["user_id", "track_id", "artist_id"])
        .agg({"ms_played": ["count", "sum"], "track_duration_ms": "first"})
        .reset_index()
    )

    # Flatten column names
    edge_data.columns = [
        "user_id",
        "track_id",
        "artist_id",
        "play_count",
        "total_ms_played",
        "track_duration_ms",
    ]

    # Calculate average completion ratio
    edge_data["avg_ms_per_play"] = edge_data["total_ms_played"] / edge_data["play_count"]
    edge_data["avg_completion_ratio"] = (
        edge_data["avg_ms_per_play"] / edge_data["track_duration_ms"]
    )

    # Cap completion ratio at 1.0 (in case of rounding errors)
    edge_data["avg_completion_ratio"] = edge_data["avg_completion_ratio"].clip(upper=1.0)

    # Add genre affinity scores if data is available
    if user_genre_prefs_df is not None and song_genres_df is not None:
        edge_data = add_genre_affinity_scores(edge_data, user_genre_prefs_df, song_genres_df)

    # Drop intermediate columns
    edge_data = edge_data.drop(columns=["avg_ms_per_play", "track_duration_ms"])

    return edge_data


def add_genre_affinity_scores(
    edge_data: pd.DataFrame, user_genre_prefs_df: pd.DataFrame, song_genres_df: pd.DataFrame
) -> pd.DataFrame:
    """Calculate and add genre affinity scores to edges."""
    # Create user-genre affinity lookup
    user_genre_dict = {}
    for user_id in user_genre_prefs_df["user_id"].unique():
        user_prefs = user_genre_prefs_df[user_genre_prefs_df["user_id"] == user_id]
        user_genre_dict[user_id] = dict(zip(user_prefs["genre_id"], user_prefs["affinity_score"]))

    # Create song-genres lookup (songs can have multiple genres)
    song_genre_dict = {}
    for track_id in song_genres_df["track_id"].unique():
        song_genres = song_genres_df[song_genres_df["track_id"] == track_id]["genre_id"].tolist()
        song_genre_dict[track_id] = song_genres

    # Calculate affinity scores
    affinity_scores = []
    for _, row in edge_data.iterrows():
        user_id = row["user_id"]
        track_id = row["track_id"]

        # Get user's genre preferences
        user_prefs = user_genre_dict.get(user_id, {})

        # Get song's genres
        song_genres = song_genre_dict.get(track_id, [])

        # Calculate average affinity across all song genres
        if user_prefs and song_genres:
            affinities = [user_prefs.get(genre, 0.1) for genre in song_genres]
            avg_affinity = np.mean(affinities)
        else:
            # Default affinity if no preference data
            avg_affinity = 0.1

        affinity_scores.append(avg_affinity)

    edge_data["genre_affinity_score"] = affinity_scores

    return edge_data


def prepare_genre_mappings(data_dir: str):
    """Save genre mappings as separate files for graph construction."""
    # These will be used directly by build_graph.py
    try:
        # Load genre data
        genres_df = pd.read_csv(f"{data_dir}/synthetic_genres.csv")
        artist_genres_df = pd.read_csv(f"{data_dir}/synthetic_artist_genres.csv")
        song_genres_df = pd.read_csv(f"{data_dir}/synthetic_song_genres.csv")
        user_genre_prefs_df = pd.read_csv(f"{data_dir}/synthetic_user_genre_preferences.csv")

        # Save as parquet for faster loading in graph construction
        genres_df.to_parquet(f"{data_dir}/genres.parquet", index=False)
        artist_genres_df.to_parquet(f"{data_dir}/artist_genres.parquet", index=False)
        song_genres_df.to_parquet(f"{data_dir}/song_genres.parquet", index=False)
        user_genre_prefs_df.to_parquet(f"{data_dir}/user_genre_preferences.parquet", index=False)

        print("\nGenre mappings saved:")
        print(f"- Genres: {len(genres_df)} genres")
        print(f"- Artist-Genre: {len(artist_genres_df)} mappings")
        print(f"- Song-Genre: {len(song_genres_df)} mappings")
        print(f"- User-Genre: {len(user_genre_prefs_df)} preferences")

        return True
    except FileNotFoundError as e:
        print(f"\nWarning: Genre data not found ({e})")
        print("Proceeding without genre information...")
        return False


def main():
    """Prepare music session data for graph construction."""
    parser = argparse.ArgumentParser(
        description="Prepare music session data for graph construction"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic_sessions.csv",
        help="Input sessions CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/edge_list.parquet",
        help="Output edge list Parquet file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing all data files",
    )
    parser.add_argument(
        "--include-genres",
        action="store_true",
        default=True,
        help="Include genre affinity scores in edge data",
    )

    args = parser.parse_args()

    print(f"Loading sessions from: {args.input}")
    sessions_df = pd.read_csv(args.input)
    print(f"Loaded {len(sessions_df):,} sessions")

    # Try to load genre data if requested
    user_genre_prefs_df = None
    song_genres_df = None
    genre_data_available = False

    if args.include_genres:
        try:
            print("\nLoading genre data...")
            user_genre_prefs_df = pd.read_csv(
                f"{args.data_dir}/synthetic_user_genre_preferences.csv"
            )
            song_genres_df = pd.read_csv(f"{args.data_dir}/synthetic_song_genres.csv")
            print(f"- Loaded {len(user_genre_prefs_df):,} user genre preferences")
            print(f"- Loaded {len(song_genres_df):,} song-genre mappings")
            genre_data_available = True
        except FileNotFoundError:
            print("Genre data not found, proceeding without genre affinity scores")

    print("\nAggregating into edges...")
    edge_data = prepare_edge_data(sessions_df, user_genre_prefs_df, song_genres_df)
    print(f"Created {len(edge_data):,} unique user-song edges")

    # Show statistics
    print("\nEdge Statistics:")
    print(f"- Users: {edge_data['user_id'].nunique():,}")
    print(f"- Songs: {edge_data['track_id'].nunique():,}")
    print(f"- Artists: {edge_data['artist_id'].nunique():,}")
    print(f"- Avg plays per edge: {edge_data['play_count'].mean():.1f}")
    print(f"- Avg completion ratio: {edge_data['avg_completion_ratio'].mean():.1%}")

    if genre_data_available and "genre_affinity_score" in edge_data.columns:
        print(f"- Avg genre affinity: {edge_data['genre_affinity_score'].mean():.3f}")
        print(
            f"- Genre affinity range: [{edge_data['genre_affinity_score'].min():.3f}, "
            f"{edge_data['genre_affinity_score'].max():.3f}]"
        )

    # Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    edge_data.to_parquet(output_path, index=False)
    print(f"\nSaved edge data to: {args.output}")

    # Prepare genre mappings for graph construction
    if genre_data_available:
        prepare_genre_mappings(args.data_dir)

    # Preview
    print("\nSample edges:")
    print(edge_data.head())


if __name__ == "__main__":
    main()
