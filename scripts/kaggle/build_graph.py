"""
Build heterogeneous graph from Kaggle playlist data.

Creates a PyTorch Geometric HeteroData object with:
- Node types: playlist, track, artist, genre
- Edge types:
  - playlist->track (contains)
  - track->artist (by)
  - track->genre (has_genre)
  - artist->genre (performs_genre)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: Path):
    """Load processed Kaggle data from parquet files."""
    logger.info(f"Loading processed data from {data_dir}")

    # Load entities
    playlists = pd.read_parquet(data_dir / "playlists.parquet")
    tracks = pd.read_parquet(data_dir / "tracks.parquet")
    artists = pd.read_parquet(data_dir / "artists.parquet")
    genres = pd.read_parquet(data_dir / "genres.parquet")

    # Load relationships
    playlist_tracks = pd.read_parquet(data_dir / "playlist_tracks.parquet")
    track_genres = pd.read_parquet(data_dir / "track_genres.parquet")
    artist_genres = pd.read_parquet(data_dir / "artist_genres.parquet")

    return {
        "playlists": playlists,
        "tracks": tracks,
        "artists": artists,
        "genres": genres,
        "playlist_tracks": playlist_tracks,
        "track_genres": track_genres,
        "artist_genres": artist_genres,
    }


def create_node_mappings(data):
    """Create ID to index mappings for all node types."""
    logger.info("Creating node mappings...")

    mappings = {}

    # Create mappings for each entity type
    for entity_type in ["playlists", "tracks", "artists", "genres"]:
        df = data[entity_type]
        id_col = f"{entity_type[:-1]}_id"  # Remove 's' from plural

        # Get unique IDs
        unique_ids = df[id_col].unique()
        mappings[entity_type] = {id_: idx for idx, id_ in enumerate(unique_ids)}

        logger.info(f"  {entity_type}: {len(unique_ids)} nodes")

    return mappings


def create_edge_indices(data, mappings):
    """Create edge indices for all relationship types."""
    logger.info("Creating edge indices...")

    edges = {}

    # Playlist->Track edges
    pt_df = data["playlist_tracks"]
    playlist_idx = [mappings["playlists"][pid] for pid in pt_df["playlist_id"]]
    track_idx = [mappings["tracks"][tid] for tid in pt_df["track_id"]]

    edges[("playlist", "contains", "track")] = torch.tensor(
        [playlist_idx, track_idx], dtype=torch.long
    )
    edges[("track", "in_playlist", "playlist")] = torch.tensor(
        [track_idx, playlist_idx], dtype=torch.long
    )

    # Track->Artist edges
    tracks_df = data["tracks"]
    track_idx = [mappings["tracks"][tid] for tid in tracks_df["track_id"]]
    artist_idx = [mappings["artists"][aid] for aid in tracks_df["artist_id"]]

    edges[("track", "by", "artist")] = torch.tensor([track_idx, artist_idx], dtype=torch.long)
    edges[("artist", "created", "track")] = torch.tensor([artist_idx, track_idx], dtype=torch.long)

    # Track->Genre edges
    tg_df = data["track_genres"]
    track_idx = [mappings["tracks"][tid] for tid in tg_df["track_id"]]
    genre_idx = [mappings["genres"][gid] for gid in tg_df["genre_id"]]

    edges[("track", "has_genre", "genre")] = torch.tensor([track_idx, genre_idx], dtype=torch.long)
    edges[("genre", "includes_track", "track")] = torch.tensor(
        [genre_idx, track_idx], dtype=torch.long
    )

    # Artist->Genre edges
    ag_df = data["artist_genres"]
    artist_idx = [mappings["artists"][aid] for aid in ag_df["artist_id"]]
    genre_idx = [mappings["genres"][gid] for gid in ag_df["genre_id"]]

    edges[("artist", "performs_genre", "genre")] = torch.tensor(
        [artist_idx, genre_idx], dtype=torch.long
    )
    edges[("genre", "performed_by", "artist")] = torch.tensor(
        [genre_idx, artist_idx], dtype=torch.long
    )

    # Log edge counts
    for edge_type, edge_index in edges.items():
        logger.info(f"  {edge_type}: {edge_index.shape[1]} edges")

    return edges


def create_node_features(data, mappings):
    """Create feature tensors for nodes."""
    logger.info("Creating node features...")

    features = {}

    # Playlist features: track count, avg audio features
    playlists_df = data["playlists"]
    playlist_tracks_df = data["playlist_tracks"]
    tracks_df = data["tracks"]

    # Calculate average audio features per playlist using vectorized operations
    logger.info("Computing playlist features using vectorized operations...")

    # Define audio feature columns
    audio_cols = [
        "danceability",
        "energy",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "valence",
        "tempo",
    ]

    # Merge playlist_tracks with tracks to get all features at once
    merged = playlist_tracks_df.merge(
        tracks_df[["track_id"] + audio_cols], on="track_id", how="left"
    )

    # Calculate mean features for each playlist
    playlist_means = merged.groupby("playlist_id")[audio_cols].mean()

    # Calculate track counts (normalized by 100)
    track_counts = merged.groupby("playlist_id").size() / 100.0
    track_counts.name = "track_count"

    # Combine track counts with audio features
    playlist_features_df = pd.concat([track_counts, playlist_means], axis=1)

    # Ensure all playlists are included (even those with no tracks)
    all_playlist_ids = playlists_df["playlist_id"].unique()
    playlist_features_df = playlist_features_df.reindex(all_playlist_ids, fill_value=0.0)

    # Convert to numpy array in the correct order
    feature_cols = ["track_count"] + audio_cols
    playlist_features = playlist_features_df[feature_cols].values

    features["playlist"] = torch.tensor(playlist_features, dtype=torch.float32)

    # Track features: audio features + popularity (if available)
    audio_cols = [
        "danceability",
        "energy",
        "acousticness",
        "speechiness",
        "instrumentalness",
        "valence",
        "tempo",
    ]

    # Add popularity if it exists
    if "popularity" in tracks_df.columns:
        audio_cols.append("popularity")

    track_features = tracks_df[audio_cols].values

    # Normalize tempo
    track_features[:, 6] = track_features[:, 6] / 200.0  # Tempo

    # Normalize popularity if it exists
    if "popularity" in tracks_df.columns:
        track_features[:, 7] = track_features[:, 7] / 100.0  # Popularity

    features["track"] = torch.tensor(track_features, dtype=torch.float32)

    # Artist features: track count (placeholder for now)
    artist_track_counts = tracks_df.groupby("artist_id").size()
    artist_features = []

    for aid in data["artists"]["artist_id"]:
        count = artist_track_counts.get(aid, 1)
        artist_features.append([np.log1p(count) / 10.0])  # Log scale, normalize

    features["artist"] = torch.tensor(np.array(artist_features), dtype=torch.float32)

    # Genre features: one-hot encoding
    n_genres = len(data["genres"])
    genre_features = torch.eye(n_genres)
    features["genre"] = genre_features

    # Log feature dimensions
    for node_type, feat in features.items():
        logger.info(f"  {node_type}: {feat.shape}")

    return features


def build_playlist_graph(data_dir: Path, output_path: Path):
    """Build and save the playlist graph."""
    # Load data
    data = load_processed_data(data_dir)

    # Create mappings
    mappings = create_node_mappings(data)

    # Create graph
    graph = HeteroData()

    # Add node counts
    graph["playlist"].num_nodes = len(mappings["playlists"])
    graph["track"].num_nodes = len(mappings["tracks"])
    graph["artist"].num_nodes = len(mappings["artists"])
    graph["genre"].num_nodes = len(mappings["genres"])

    # Add edges
    edges = create_edge_indices(data, mappings)
    for edge_type, edge_index in edges.items():
        graph[edge_type].edge_index = edge_index

    # Add node features
    features = create_node_features(data, mappings)
    for node_type, feat in features.items():
        graph[node_type].x = feat

    # Save mappings for later use
    graph.mappings = mappings

    # Save graph
    logger.info(f"Saving graph to {output_path}")
    torch.save(graph, output_path)

    # Print summary
    print("\nGraph Summary:")
    print("  Nodes:")
    for node_type in ["playlist", "track", "artist", "genre"]:
        print(f"    {node_type}: {graph[node_type].num_nodes}")
    print("  Edges:")
    for edge_type in graph.edge_types:
        print(f"    {edge_type}: {graph[edge_type].edge_index.shape[1]}")

    return graph


def main():
    """Main function to build playlist graph."""
    parser = argparse.ArgumentParser(description="Build playlist graph from Kaggle data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kaggle",
        help="Directory containing processed Kaggle data",
    )
    parser.add_argument(
        "--output", type=str, default="data/kaggle/playlist_graph.pt", help="Output path for graph"
    )

    args = parser.parse_args()

    # Add numpy import that was missing
    global np
    import numpy as np

    build_playlist_graph(Path(args.data_dir), Path(args.output))


if __name__ == "__main__":
    main()
