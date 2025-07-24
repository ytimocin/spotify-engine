"""
Simplified graph building for Kaggle playlist data.

Creates a PyTorch Geometric HeteroData object with standard patterns.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: Path):
    """Load processed data from parquet files."""
    logger.info(f"Loading processed data from {data_dir}")
    
    # Validate all required files exist
    required_files = [
        "playlists.parquet", "tracks.parquet", "artists.parquet", 
        "albums.parquet", "playlist_tracks.parquet", 
        "track_artists.parquet", "track_albums.parquet"
    ]
    
    for file in required_files:
        if not (data_dir / file).exists():
            raise FileNotFoundError(
                f"Required file {file} not found in {data_dir}. "
                f"Please run prepare_data.py first."
            )

    # Load entities
    playlists = pd.read_parquet(data_dir / "playlists.parquet")
    tracks = pd.read_parquet(data_dir / "tracks.parquet")
    artists = pd.read_parquet(data_dir / "artists.parquet")
    albums = pd.read_parquet(data_dir / "albums.parquet")

    # Load relationships
    playlist_tracks = pd.read_parquet(data_dir / "playlist_tracks.parquet")
    track_artists = pd.read_parquet(data_dir / "track_artists.parquet")
    track_albums = pd.read_parquet(data_dir / "track_albums.parquet")

    logger.info(
        f"Loaded {len(playlists):,} playlists, {len(tracks):,} tracks, {len(artists):,} artists, {len(albums):,} albums"
    )

    return {
        "playlists": playlists,
        "tracks": tracks,
        "artists": artists,
        "albums": albums,
        "playlist_tracks": playlist_tracks,
        "track_artists": track_artists,
        "track_albums": track_albums,
    }


def create_node_features(data):
    """Create feature tensors for nodes."""
    logger.info("Creating node features...")

    playlists_df = data["playlists"]
    tracks_df = data["tracks"]
    artists_df = data["artists"]
    playlist_tracks_df = data["playlist_tracks"]
    track_artists_df = data["track_artists"]

    # Playlist features: track count (normalized)
    track_counts = playlist_tracks_df.groupby("playlist_id").size()
    playlist_features = []
    for pid in playlists_df["playlist_id"]:
        count = track_counts.get(pid, 0)
        # Simple normalization: log scale
        playlist_features.append([np.log1p(count) / 5.0])

    # Track features: audio features (already normalized 0-1)
    audio_features = [
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "speechiness",
        "instrumentalness",
    ]

    # Add optional features if they exist
    for feat in ["key", "mode", "loudness", "time_signature"]:
        if feat in tracks_df.columns:
            audio_features.append(feat)

    track_features = tracks_df[audio_features].values

    # Domain-based normalization ensures consistency across different datasets
    # Tempo: Most music falls between 40-240 BPM
    if "tempo" in tracks_df.columns:
        tempo_values = tracks_df["tempo"].values.reshape(-1, 1)
        # Standard tempo range: 40-240 BPM (covers almost all music)
        # Normalize to [0,1] with clipping for outliers
        tempo_normalized = np.clip((tempo_values - 40) / (240 - 40), 0, 1)
        track_features = np.hstack([track_features, tempo_normalized])

    # Normalize optional features using musical domain knowledge
    # Key: Musical keys are numbered 0-11 (C, C#, D, ... B)
    if "key" in audio_features:
        key_idx = audio_features.index("key")
        track_features[:, key_idx] = track_features[:, key_idx] / 11.0

    # Loudness: Industry standard range is -60 to 0 dB
    if "loudness" in audio_features:
        loudness_idx = audio_features.index("loudness")
        # Standard loudness range: -60 to 0 dB
        # Values above 0 are clipped to 1
        track_features[:, loudness_idx] = np.clip((track_features[:, loudness_idx] + 60) / 60, 0, 1)

    # Time signature: Common values are 3, 4, 5, rarely above 7
    if "time_signature" in audio_features:
        ts_idx = audio_features.index("time_signature")
        # Standard time signatures: 3/4, 4/4, 5/4, 6/8, 7/8 etc.
        # Maximum reasonable value is 7
        track_features[:, ts_idx] = np.clip(track_features[:, ts_idx] / 7, 0, 1)

    # Artist features: track count (normalized)
    artist_track_counts = track_artists_df.groupby("artist_id").size()
    artist_features = []
    for aid in artists_df["artist_id"]:
        count = artist_track_counts.get(aid, 0)
        artist_features.append([np.log1p(count) / 5.0])

    # Album features: track count (normalized)
    albums_df = data["albums"]
    track_albums_df = data["track_albums"]
    album_track_counts = track_albums_df.groupby("album_id").size()
    album_features = []
    for aid in albums_df["album_id"]:
        count = album_track_counts.get(aid, 0)
        album_features.append([np.log1p(count) / 5.0])

    return {
        "playlist": torch.tensor(playlist_features, dtype=torch.float32),
        "track": torch.tensor(track_features, dtype=torch.float32),
        "artist": torch.tensor(artist_features, dtype=torch.float32),
        "album": torch.tensor(album_features, dtype=torch.float32),
    }


def create_edge_indices(data):
    """Create edge indices for all relationships."""
    logger.info("Creating edge indices...")

    # Create ID to index mappings
    playlists_df = data["playlists"]
    tracks_df = data["tracks"]
    artists_df = data["artists"]
    albums_df = data["albums"]

    playlist_to_idx = {pid: idx for idx, pid in enumerate(playlists_df["playlist_id"])}
    track_to_idx = {tid: idx for idx, tid in enumerate(tracks_df["track_id"])}
    artist_to_idx = {aid: idx for idx, aid in enumerate(artists_df["artist_id"])}
    album_to_idx = {aid: idx for idx, aid in enumerate(albums_df["album_id"])}

    # Playlist->Track edges
    playlist_tracks_df = data["playlist_tracks"]
    playlist_idx = [playlist_to_idx[pid] for pid in playlist_tracks_df["playlist_id"]]
    track_idx = [track_to_idx[tid] for tid in playlist_tracks_df["track_id"]]

    playlist_track_edge_index = torch.tensor([playlist_idx, track_idx], dtype=torch.long)
    
    # Add edge attributes if available
    playlist_track_edge_attr = None
    if "norm_position" in playlist_tracks_df.columns:
        playlist_track_edge_attr = torch.tensor(
            playlist_tracks_df["norm_position"].values.reshape(-1, 1), 
            dtype=torch.float32
        )

    # Track->Artist edges
    track_artists_df = data["track_artists"]
    track_idx = [track_to_idx[tid] for tid in track_artists_df["track_id"]]
    artist_idx = [artist_to_idx[aid] for aid in track_artists_df["artist_id"]]

    track_artist_edge_index = torch.tensor([track_idx, artist_idx], dtype=torch.long)

    # Track->Album edges
    track_albums_df = data["track_albums"]
    track_idx = [track_to_idx[tid] for tid in track_albums_df["track_id"]]
    album_idx = [album_to_idx[aid] for aid in track_albums_df["album_id"]]

    track_album_edge_index = torch.tensor([track_idx, album_idx], dtype=torch.long)

    logger.info(f"Created {playlist_track_edge_index.shape[1]:,} playlist-track edges")
    logger.info(f"Created {track_artist_edge_index.shape[1]:,} track-artist edges")
    logger.info(f"Created {track_album_edge_index.shape[1]:,} track-album edges")

    edges = {
        ("playlist", "contains", "track"): playlist_track_edge_index,
        ("track", "in_playlist", "playlist"): playlist_track_edge_index.flip(0),
        ("track", "by", "artist"): track_artist_edge_index,
        ("artist", "created", "track"): track_artist_edge_index.flip(0),
        ("track", "from_album", "album"): track_album_edge_index,
        ("album", "contains", "track"): track_album_edge_index.flip(0),
    }
    
    # Return edge attributes if available
    edge_attrs = {}
    if playlist_track_edge_attr is not None:
        edge_attrs[("playlist", "contains", "track")] = playlist_track_edge_attr
        edge_attrs[("track", "in_playlist", "playlist")] = playlist_track_edge_attr
        logger.info("Added position features to playlist-track edges")
    
    return edges, edge_attrs


def build_graph(data_dir: Path, output_path: Path):
    """Build and save the playlist graph."""
    # Load data
    data = load_processed_data(data_dir)

    # Create HeteroData object
    graph = HeteroData()

    # Add node counts
    graph["playlist"].num_nodes = len(data["playlists"])
    graph["track"].num_nodes = len(data["tracks"])
    graph["artist"].num_nodes = len(data["artists"])
    graph["album"].num_nodes = len(data["albums"])

    # Add node features
    features = create_node_features(data)
    for node_type, feat in features.items():
        graph[node_type].x = feat

    # Add edges and edge attributes
    edges, edge_attrs = create_edge_indices(data)
    for edge_type, edge_index in edges.items():
        graph[edge_type].edge_index = edge_index
        # Add edge attributes if available
        if edge_type in edge_attrs:
            graph[edge_type].edge_attr = edge_attrs[edge_type]

    # Save graph
    logger.info(f"Saving graph to {output_path}")
    torch.save(graph, output_path, _use_new_zipfile_serialization=True)

    # Print summary
    print("\nGraph Summary:")
    print("  Nodes:")
    for node_type in graph.node_types:
        num_nodes = graph[node_type].num_nodes
        feat_dim = graph[node_type].x.shape[1] if hasattr(graph[node_type], "x") else 0
        print(f"    {node_type}: {num_nodes:,} nodes, {feat_dim} features")

    print("  Edges:")
    for edge_type in graph.edge_types:
        num_edges = graph[edge_type].edge_index.shape[1]
        print(f"    {edge_type}: {num_edges:,} edges")

    return graph


def main():
    """Main function to build playlist graph."""
    parser = argparse.ArgumentParser(description="Build playlist graph from processed data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/kaggle/processed",
        help="Directory containing processed data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/kaggle/playlist_graph.pt",
        help="Output path for graph",
    )

    args = parser.parse_args()

    build_graph(Path(args.data_dir), Path(args.output))


if __name__ == "__main__":
    main()
