"""
Test script for Kaggle playlist-based model.

This script loads a trained PlaylistGAT model and demonstrates:
1. Making recommendations for a playlist
2. Explaining why tracks were recommended
3. Evaluating recommendation quality
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.kaggle.models import PlaylistGAT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def display_playlist_info(graph, playlist_idx: int):
    """Display information about a playlist."""
    print(f"\nPlaylist {playlist_idx} Information:")
    print("-" * 50)

    # Get playlist features
    playlist_features = graph["playlist"].x[playlist_idx]
    print(
        f"Features: track_count={playlist_features[0]:.2f}, "
        f"danceability={playlist_features[1]:.2f}, "
        f"energy={playlist_features[2]:.2f}"
    )

    # Get playlist tracks
    edge_index = graph[("playlist", "contains", "track")].edge_index
    playlist_edges = edge_index[0] == playlist_idx
    track_indices = edge_index[1][playlist_edges]

    print(f"Number of tracks: {len(track_indices)}")

    # Show some track info
    if len(track_indices) > 0:
        print("\nFirst 5 tracks in playlist:")
        for _i, track_idx in enumerate(track_indices[:5]):
            track_features = graph["track"].x[track_idx]
            print(
                f"  Track {track_idx}: energy={track_features[1]:.2f}, "
                f"valence={track_features[5]:.2f}"
            )

    return track_indices


def test_recommendations(model: PlaylistGAT, graph, playlist_idx: int, k: int = 10):
    """Test recommendations for a playlist."""
    print(f"\nGenerating {k} recommendations for playlist {playlist_idx}...")

    # Create node indices
    x_dict = {
        "playlist": torch.arange(graph["playlist"].num_nodes),
        "track": torch.arange(graph["track"].num_nodes),
        "artist": torch.arange(graph["artist"].num_nodes),
        "genre": torch.arange(graph["genre"].num_nodes),
    }

    # Get recommendations
    model.eval()
    with torch.no_grad():
        rec_tracks, rec_scores = model.get_playlist_recommendations(
            playlist_idx, x_dict, graph, k=k, exclude_known=True
        )

    print(f"\nTop {k} Recommended Tracks:")
    print("-" * 50)

    for i, (track_idx, score) in enumerate(zip(rec_tracks, rec_scores)):
        track_features = graph["track"].x[track_idx]
        print(f"{i + 1}. Track {track_idx} (score: {score:.3f})")
        print(
            f"   Features: energy={track_features[1]:.2f}, "
            f"valence={track_features[5]:.2f}, "
            f"danceability={track_features[0]:.2f}"
        )

        # Get artist
        artist_edges = graph[("track", "by", "artist")].edge_index
        track_artists = artist_edges[1][artist_edges[0] == track_idx]
        if len(track_artists) > 0:
            print(f"   Artist: {track_artists[0].item()}")

        # Get genres
        genre_edges = graph[("track", "has_genre", "genre")].edge_index
        track_genres = genre_edges[1][genre_edges[0] == track_idx]
        if len(track_genres) > 0:
            print(f"   Genres: {track_genres.tolist()}")

    return rec_tracks, rec_scores


def test_explanations(model: PlaylistGAT, graph, playlist_idx: int, track_idx: int):
    """Test recommendation explanations."""
    print(f"\nExplaining why Track {track_idx} was recommended for Playlist {playlist_idx}...")
    print("-" * 70)

    # Create node indices
    x_dict = {
        "playlist": torch.arange(graph["playlist"].num_nodes),
        "track": torch.arange(graph["track"].num_nodes),
        "artist": torch.arange(graph["artist"].num_nodes),
        "genre": torch.arange(graph["genre"].num_nodes),
    }

    # Get explanation
    model.eval()
    with torch.no_grad():
        explanation = model.explain_recommendation(playlist_idx, track_idx, x_dict, graph)

    print(f"Recommendation Score: {explanation['score']:.3f}")

    if explanation.get("genre_influence"):
        print("\nGenre Influence:")
        for genre_info in explanation["genre_influence"][:3]:
            print(
                f"  - Genre {genre_info['genre_idx']}: "
                f"similarity = {genre_info['similarity']:.3f}"
            )

    if explanation.get("artist_influence"):
        print("\nArtist Influence:")
        print(
            f"  - Artist {explanation['artist_influence']['artist_idx']}: "
            f"similarity = {explanation['artist_influence']['similarity']:.3f}"
        )
        if explanation["artist_influence"]["tracks_in_playlist"] > 0:
            print(
                f"    (Playlist already contains "
                f"{explanation['artist_influence']['tracks_in_playlist']} "
                f"tracks by this artist)"
            )

    if explanation.get("similar_tracks_in_playlist"):
        print("\nSimilar Tracks Already in Playlist:")
        for track_info in explanation["similar_tracks_in_playlist"]:
            print(
                f"  - Track {track_info['track_idx']}: "
                f"similarity = {track_info['similarity']:.3f}"
            )


def main():
    parser = argparse.ArgumentParser(description="Test Kaggle playlist model")
    parser.add_argument(
        "--graph", type=str, default="data/kaggle/playlist_graph.pt", help="Path to playlist graph"
    )
    parser.add_argument(
        "--model", type=str, default="models/kaggle/best_model.pt", help="Path to trained model"
    )
    parser.add_argument("--playlist", type=int, default=0, help="Playlist ID to test")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--explain", action="store_true", help="Show recommendation explanations")

    args = parser.parse_args()

    # Load graph
    logger.info(f"Loading graph from {args.graph}")
    graph = torch.load(args.graph, weights_only=False)

    # Display graph info
    print("\nPlaylist Graph Structure:")
    print("-" * 50)
    print(f"Node types: {graph.node_types}")
    print(f"Edge types: {graph.edge_types}")
    print()

    print("Node counts:")
    for node_type in graph.node_types:
        print(f"  {node_type}: {graph[node_type].num_nodes:,}")
    print()

    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\nNo trained model found at {model_path}")
        print("Please train a model first using: python -m src.kaggle.train")
        return

    # Load model
    logger.info(f"Loading model from {args.model}")

    # Get feature dimensions from graph
    playlist_feature_dim = graph["playlist"].x.shape[1]
    track_feature_dim = graph["track"].x.shape[1]
    artist_feature_dim = graph["artist"].x.shape[1]

    # Initialize model with same architecture as training
    model = PlaylistGAT(
        num_playlists=graph["playlist"].num_nodes,
        num_tracks=graph["track"].num_nodes,
        num_artists=graph["artist"].num_nodes,
        num_genres=graph["genre"].num_nodes,
        playlist_feature_dim=playlist_feature_dim,
        track_feature_dim=track_feature_dim,
        artist_feature_dim=artist_feature_dim,
        embedding_dim=64,
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.1,
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded successfully! Parameters: {total_params:,}")

    # Load metrics if available
    metrics_path = model_path.parent / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print("\nTraining Metrics:")
        print(f"  Best epoch: {metrics['best_epoch']}")
        print(f"  Best validation Recall@10: {metrics['best_val_recall@10']:.4f}")
        if "test_metrics" in metrics:
            print("\nTest Metrics:")
            for metric, value in metrics["test_metrics"].items():
                print(f"  {metric}: {value:.4f}")

    # Display playlist info
    display_playlist_info(graph, args.playlist)

    # Test recommendations
    rec_tracks, rec_scores = test_recommendations(model, graph, args.playlist, args.top_k)

    # Test explanations
    if args.explain and len(rec_tracks) > 0:
        # Explain top recommendation
        test_explanations(model, graph, args.playlist, rec_tracks[0].item())

        # Also explain a random recommendation
        if len(rec_tracks) > 3:
            print("\n" + "=" * 70)
            test_explanations(model, graph, args.playlist, rec_tracks[3].item())

    print("\n✅ Playlist-based recommendation system is working!")


if __name__ == "__main__":
    main()
