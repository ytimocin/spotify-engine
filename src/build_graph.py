"""
Build heterogeneous graph from edge data for music recommendations.

Creates a PyTorch Geometric HeteroData object with:
- Node types: user, song, artist, genre
- Edge types:
  - user->song (listens): with play_count, completion_ratio, edge_weight, genre_affinity
  - song->artist (by): which artist created the song
  - user->genre (prefers): user genre preferences with affinity scores
  - song->genre (has): song genre associations
  - artist->genre (performs): artist genre associations
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_music_graph(  # noqa: C901
    edge_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    genres_df: pd.DataFrame = None,
    user_genre_prefs_df: pd.DataFrame = None,
    song_genres_df: pd.DataFrame = None,
    artist_genres_df: pd.DataFrame = None,
) -> HeteroData:
    """
    Build heterogeneous graph from edge data.

    Args:
        edge_df: DataFrame with user-song edges
        songs_df: DataFrame with song metadata (track_id, artist_id)
        genres_df: DataFrame with genre information (optional)
        user_genre_prefs_df: DataFrame with user genre preferences (optional)
        song_genres_df: DataFrame with song-genre mappings (optional)
        artist_genres_df: DataFrame with artist-genre mappings (optional)

    Returns:
        HeteroData graph object
    """
    # Create node mappings
    users = edge_df["user_id"].unique()
    songs = edge_df["track_id"].unique()
    artists = edge_df["artist_id"].unique()

    # Handle genres if available
    has_genres = genres_df is not None
    if has_genres:
        genres = genres_df["genre_id"].unique()
        genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    else:
        genres = []

    user_to_idx = {user: idx for idx, user in enumerate(users)}
    song_to_idx = {song: idx for idx, song in enumerate(songs)}
    artist_to_idx = {artist: idx for idx, artist in enumerate(artists)}

    print(
        f"Graph nodes: {len(users)} users, {len(songs)} songs, {len(artists)} artists",
        f", {len(genres)} genres" if has_genres else "",
    )

    # Create graph
    graph = HeteroData()

    # Add node counts
    graph["user"].num_nodes = len(users)
    graph["song"].num_nodes = len(songs)
    graph["artist"].num_nodes = len(artists)
    if has_genres:
        graph["genre"].num_nodes = len(genres)

    # Create user->song edges
    edge_index = torch.tensor(
        [
            [user_to_idx[row.user_id] for _, row in edge_df.iterrows()],
            [song_to_idx[row.track_id] for _, row in edge_df.iterrows()],
        ],
        dtype=torch.long,
    )

    # Calculate edge weights (combining play frequency and completion)
    play_counts = edge_df["play_count"].values
    completion_ratios = edge_df["avg_completion_ratio"].values

    # Normalize play counts (log scale to handle outliers)
    normalized_plays = np.log1p(play_counts) / np.log1p(play_counts.max())

    # Combine: 70% completion ratio + 30% play frequency
    edge_weights = 0.7 * completion_ratios + 0.3 * normalized_plays

    # Check if genre affinity scores are available
    if "genre_affinity_score" in edge_df.columns:
        genre_affinities = edge_df["genre_affinity_score"].values
        # Adjust edge weights to include genre affinity: 60% completion + 20% play + 20% genre
        edge_weights = 0.6 * completion_ratios + 0.2 * normalized_plays + 0.2 * genre_affinities
        edge_attrs = np.column_stack(
            [play_counts, completion_ratios, genre_affinities, edge_weights]
        )
    else:
        edge_attrs = np.column_stack([play_counts, completion_ratios, edge_weights])

    # Store edges and attributes
    graph["user", "listens", "song"].edge_index = edge_index
    graph["user", "listens", "song"].edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

    # Create song->artist edges
    song_artist_edges = []
    for _, row in songs_df[songs_df["track_id"].isin(songs)].iterrows():
        if row["track_id"] in song_to_idx and row["artist_id"] in artist_to_idx:
            song_artist_edges.append(
                [song_to_idx[row["track_id"]], artist_to_idx[row["artist_id"]]]
            )

    if song_artist_edges:
        graph["song", "by", "artist"].edge_index = torch.tensor(
            song_artist_edges, dtype=torch.long
        ).t()

    # Add genre-related edges if genre data is available
    if has_genres:
        # User->Genre edges (preferences)
        if user_genre_prefs_df is not None:
            user_genre_edges = []
            user_genre_affinities = []

            for _, row in user_genre_prefs_df.iterrows():
                if row["user_id"] in user_to_idx and row["genre_id"] in genre_to_idx:
                    user_genre_edges.append(
                        [user_to_idx[row["user_id"]], genre_to_idx[row["genre_id"]]]
                    )
                    user_genre_affinities.append(row["affinity_score"])

            if user_genre_edges:
                graph["user", "prefers", "genre"].edge_index = torch.tensor(
                    user_genre_edges, dtype=torch.long
                ).t()
                graph["user", "prefers", "genre"].edge_attr = torch.tensor(
                    user_genre_affinities, dtype=torch.float32
                ).unsqueeze(1)

        # Song->Genre edges
        if song_genres_df is not None:
            song_genre_edges = []

            for _, row in song_genres_df.iterrows():
                if row["track_id"] in song_to_idx and row["genre_id"] in genre_to_idx:
                    song_genre_edges.append(
                        [song_to_idx[row["track_id"]], genre_to_idx[row["genre_id"]]]
                    )

            if song_genre_edges:
                graph["song", "has", "genre"].edge_index = torch.tensor(
                    song_genre_edges, dtype=torch.long
                ).t()

        # Artist->Genre edges
        if artist_genres_df is not None:
            artist_genre_edges = []

            for _, row in artist_genres_df.iterrows():
                if row["artist_id"] in artist_to_idx and row["genre_id"] in genre_to_idx:
                    artist_genre_edges.append(
                        [artist_to_idx[row["artist_id"]], genre_to_idx[row["genre_id"]]]
                    )

            if artist_genre_edges:
                graph["artist", "performs", "genre"].edge_index = torch.tensor(
                    artist_genre_edges, dtype=torch.long
                ).t()

    # Print edge statistics
    print("Graph edges:")
    print(f"- User-Song: {edge_index.shape[1]}")
    print(f"- Song-Artist: {len(song_artist_edges)}")

    if has_genres:
        if ("user", "prefers", "genre") in graph.edge_types:
            print(f"- User-Genre: {graph['user', 'prefers', 'genre'].edge_index.shape[1]}")
        if ("song", "has", "genre") in graph.edge_types:
            print(f"- Song-Genre: {graph['song', 'has', 'genre'].edge_index.shape[1]}")
        if ("artist", "performs", "genre") in graph.edge_types:
            print(f"- Artist-Genre: {graph['artist', 'performs', 'genre'].edge_index.shape[1]}")

    return graph


def main():
    """Build graph from edge data and save to disk."""
    parser = argparse.ArgumentParser(description="Build graph from edge data")
    parser.add_argument(
        "--edges", type=str, default="data/edge_list.parquet", help="Input edge list Parquet file"
    )
    parser.add_argument(
        "--songs", type=str, default="data/synthetic_songs.csv", help="Songs metadata CSV file"
    )
    parser.add_argument("--output", type=str, default="data/graph.pt", help="Output graph file")
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Directory containing genre data files"
    )
    parser.add_argument(
        "--include-genres", action="store_true", default=True, help="Include genre nodes and edges"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading edge data from: {args.edges}")
    edge_df = pd.read_parquet(args.edges)

    print(f"Loading songs metadata from: {args.songs}")
    songs_df = pd.read_csv(args.songs)

    # Load genre data if requested
    genres_df = None
    user_genre_prefs_df = None
    song_genres_df = None
    artist_genres_df = None

    if args.include_genres:
        try:
            print("\nLoading genre data...")
            # Try parquet files first (faster), fall back to CSV
            try:
                genres_df = pd.read_parquet(f"{args.data_dir}/genres.parquet")
                user_genre_prefs_df = pd.read_parquet(
                    f"{args.data_dir}/user_genre_preferences.parquet"
                )
                song_genres_df = pd.read_parquet(f"{args.data_dir}/song_genres.parquet")
                artist_genres_df = pd.read_parquet(f"{args.data_dir}/artist_genres.parquet")
            except FileNotFoundError:
                # Fall back to CSV files
                genres_df = pd.read_csv(f"{args.data_dir}/synthetic_genres.csv")
                user_genre_prefs_df = pd.read_csv(
                    f"{args.data_dir}/synthetic_user_genre_preferences.csv"
                )
                song_genres_df = pd.read_csv(f"{args.data_dir}/synthetic_song_genres.csv")
                artist_genres_df = pd.read_csv(f"{args.data_dir}/synthetic_artist_genres.csv")

            print(f"- Loaded {len(genres_df)} genres")
            print(f"- Loaded {len(user_genre_prefs_df)} user-genre preferences")
            print(f"- Loaded {len(song_genres_df)} song-genre mappings")
            print(f"- Loaded {len(artist_genres_df)} artist-genre mappings")
        except FileNotFoundError as e:
            print(f"Warning: Genre data not found ({e}), building graph without genres")
            args.include_genres = False

    # Build graph
    print("\nBuilding graph...")
    graph = build_music_graph(
        edge_df, songs_df, genres_df, user_genre_prefs_df, song_genres_df, artist_genres_df
    )

    # Save graph
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    torch.save(graph, output_path)
    print(f"\nSaved graph to: {args.output}")

    # Print summary
    print("\nGraph Summary:")
    print(f"- User nodes: {graph['user'].num_nodes}")
    print(f"- Song nodes: {graph['song'].num_nodes}")
    print(f"- Artist nodes: {graph['artist'].num_nodes}")
    if args.include_genres and "genre" in graph.node_types:
        print(f"- Genre nodes: {graph['genre'].num_nodes}")
    print("\nEdge types:")
    for edge_type in graph.edge_types:
        src, rel, dst = edge_type
        num_edges = graph[edge_type].edge_index.shape[1]
        print(f"- {src}->{dst} ({rel}): {num_edges} edges")
    print(f"\nUser-Song edge attributes shape: {graph['user', 'listens', 'song'].edge_attr.shape}")


if __name__ == "__main__":
    main()
