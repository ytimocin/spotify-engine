"""
Build heterogeneous graph from edge data for music recommendations.

Creates a PyTorch Geometric HeteroData object with:
- Node types: user, song, artist
- Edge types: user->song (listens), song->artist (belongs)
- Edge attributes: play_count, completion_ratio, edge_weight
"""

import argparse
import pandas as pd
import torch
from torch_geometric.data import HeteroData
import numpy as np
from pathlib import Path


def build_music_graph(edge_df: pd.DataFrame, songs_df: pd.DataFrame) -> HeteroData:
    """
    Build heterogeneous graph from edge data.
    
    Args:
        edge_df: DataFrame with user-song edges
        songs_df: DataFrame with song metadata (track_id, artist_id)
    
    Returns:
        HeteroData graph object
    """
    # Create node mappings
    users = edge_df['user_id'].unique()
    songs = edge_df['track_id'].unique()
    artists = edge_df['artist_id'].unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(users)}
    song_to_idx = {song: idx for idx, song in enumerate(songs)}
    artist_to_idx = {artist: idx for idx, artist in enumerate(artists)}
    
    print(f"Graph nodes: {len(users)} users, {len(songs)} songs, {len(artists)} artists")
    
    # Create graph
    graph = HeteroData()
    
    # Add node counts
    graph['user'].num_nodes = len(users)
    graph['song'].num_nodes = len(songs)
    graph['artist'].num_nodes = len(artists)
    
    # Create user->song edges
    edge_index = torch.tensor([
        [user_to_idx[row.user_id] for _, row in edge_df.iterrows()],
        [song_to_idx[row.track_id] for _, row in edge_df.iterrows()]
    ], dtype=torch.long)
    
    # Calculate edge weights (combining play frequency and completion)
    play_counts = edge_df['play_count'].values
    completion_ratios = edge_df['avg_completion_ratio'].values
    
    # Normalize play counts (log scale to handle outliers)
    normalized_plays = np.log1p(play_counts) / np.log1p(play_counts.max())
    
    # Combine: 70% completion ratio + 30% play frequency
    edge_weights = 0.7 * completion_ratios + 0.3 * normalized_plays
    
    # Store edges and attributes
    graph['user', 'listens', 'song'].edge_index = edge_index
    graph['user', 'listens', 'song'].edge_attr = torch.tensor(
        np.column_stack([play_counts, completion_ratios, edge_weights]),
        dtype=torch.float32
    )
    
    # Create song->artist edges
    song_artist_edges = []
    for _, row in songs_df[songs_df['track_id'].isin(songs)].iterrows():
        if row['track_id'] in song_to_idx and row['artist_id'] in artist_to_idx:
            song_artist_edges.append([
                song_to_idx[row['track_id']], 
                artist_to_idx[row['artist_id']]
            ])
    
    if song_artist_edges:
        graph['song', 'by', 'artist'].edge_index = torch.tensor(
            song_artist_edges, dtype=torch.long
        ).t()
    
    print(f"Graph edges: {edge_index.shape[1]} user-song, {len(song_artist_edges)} song-artist")
    
    return graph


def main():
    parser = argparse.ArgumentParser(description='Build graph from edge data')
    parser.add_argument('--edges', type=str, default='data/edge_list.parquet',
                        help='Input edge list Parquet file')
    parser.add_argument('--songs', type=str, default='data/synthetic_songs.csv',
                        help='Songs metadata CSV file')
    parser.add_argument('--output', type=str, default='data/graph.pt',
                        help='Output graph file')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading edge data from: {args.edges}")
    edge_df = pd.read_parquet(args.edges)
    
    print(f"Loading songs metadata from: {args.songs}")
    songs_df = pd.read_csv(args.songs)
    
    # Build graph
    print("\nBuilding graph...")
    graph = build_music_graph(edge_df, songs_df)
    
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
    print(f"- User-Song edges: {graph['user', 'listens', 'song'].edge_index.shape[1]}")
    print(f"- Edge attributes shape: {graph['user', 'listens', 'song'].edge_attr.shape}")


if __name__ == '__main__':
    main()