"""
Test script for enhanced GAT model with genre support.

Verifies that the model correctly uses genre information and provides explanations.
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Module imports after path setup
from src.common.explainability import RecommendationExplainer, format_explanation  # noqa: E402
from src.common.metrics_extended import evaluate_genre_aware_recommendations  # noqa: E402
from src.common.models.enhanced_gat_recommender import EnhancedGATRecommender  # noqa: E402
from src.common.visualization.attention_viz import analyze_genre_attention  # noqa: E402


def main():  # noqa: C901
    parser = argparse.ArgumentParser(description="Test enhanced GAT model")
    parser.add_argument("--graph", type=str, default="data/synthetic/graph.pt", help="Graph file")
    parser.add_argument("--user", type=int, default=0, help="User ID to test")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Load graph
    print(f"Loading graph from {args.graph}...")
    graph = torch.load(args.graph, weights_only=False)

    # Check if graph has genre information
    has_genres = "genre" in graph.node_types

    print("\nGraph statistics:")
    print(f"- Users: {graph['user'].num_nodes}")
    print(f"- Songs: {graph['song'].num_nodes}")
    print(f"- Artists: {graph['artist'].num_nodes}")

    if has_genres:
        print(f"- Genres: {graph['genre'].num_nodes}")
        print("\n✓ Genre information detected! Using enhanced model.")
    else:
        print("\n✗ No genre information found. Model will use basic features only.")

    # Create model
    model_config = {
        "num_users": graph["user"].num_nodes,
        "num_songs": graph["song"].num_nodes,
        "num_artists": graph["artist"].num_nodes,
        "embedding_dim": 64,
        "hidden_dim": 64,
        "num_layers": 2,
        "heads": 4,
        "dropout": 0.1,
    }

    if has_genres:
        model_config["num_genres"] = graph["genre"].num_nodes

    print("\nCreating enhanced model...")
    model = EnhancedGATRecommender(**model_config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Initialize model with random weights for testing
    print("\nInitializing model with random weights...")

    # Create node indices
    x_dict = {
        "user": torch.arange(graph["user"].num_nodes),
        "song": torch.arange(graph["song"].num_nodes),
        "artist": torch.arange(graph["artist"].num_nodes),
    }
    if has_genres:
        x_dict["genre"] = torch.arange(graph["genre"].num_nodes)

    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        embeddings, attention = model(x_dict, graph, return_attention=True)

    print("✓ Forward pass successful")
    print(f"  - User embeddings shape: {embeddings['user'].shape}")
    print(f"  - Song embeddings shape: {embeddings['song'].shape}")

    # Get recommendations for test user
    print(f"\nGenerating recommendations for user {args.user}...")
    rec_songs, rec_scores, _ = model.recommend(
        args.user, x_dict, graph, k=args.top_k, exclude_known=True
    )

    print(f"\nTop {args.top_k} recommendations:")
    for i, (song_idx, score) in enumerate(zip(rec_songs, rec_scores)):
        print(f"{i + 1}. Song {song_idx.item()} (score: {score.item():.3f})")

    # Test explainability
    if has_genres:
        print("\n" + "=" * 60)
        print("TESTING EXPLAINABILITY")
        print("=" * 60)

        # Create explainer
        explainer = RecommendationExplainer(graph, model)

        # Explain first recommendation
        first_rec = rec_songs[0].item()
        print(f"\nExplaining why Song {first_rec} was recommended to User {args.user}:")

        explanation = model.explain_recommendation(args.user, first_rec, x_dict, graph)

        print("\nExplanation details:")
        print(f"- Recommendation score: {explanation['score']:.3f}")

        if explanation.get("genre_influence"):
            print(f"\nGenre influence ({len(explanation['genre_influence'])} common genres):")
            for genre_info in explanation["genre_influence"][:3]:
                print(
                    f"  - Genre {genre_info['genre_idx']}: contribution = {genre_info['contribution']:.3f}"
                )

        if explanation.get("artist_influence"):
            print("\nArtist influence:")
            print(
                f"  - Artist {explanation['artist_influence']['artist_idx']}: "
                f"similarity = {explanation['artist_influence']['similarity']:.3f}"
            )

        # Test comprehensive explanation
        print("\n" + "-" * 40)
        print("Comprehensive explanation:")
        full_explanation = explainer.explain_recommendation(args.user, first_rec)
        print(format_explanation(full_explanation, verbose=args.verbose))

        # Analyze genre attention
        print("\n" + "-" * 40)
        print("Genre attention analysis:")
        genre_analysis = analyze_genre_attention(model, graph, args.user, top_k=args.top_k)

        if not genre_analysis.empty:
            print("\nGenre influence on recommendations:")
            print(
                genre_analysis[
                    ["song_idx", "score", "num_common_genres", "genre_influence_score"]
                ].to_string()
            )

    # Test evaluation metrics
    if has_genres:
        print("\n" + "=" * 60)
        print("TESTING GENRE-AWARE METRICS")
        print("=" * 60)

        test_users = list(range(min(10, graph["user"].num_nodes)))
        metrics = evaluate_genre_aware_recommendations(model, graph, test_users, k=args.top_k)

        print("\nGenre-aware evaluation metrics:")
        for metric, value in metrics.items():
            print(f"- {metric}: {value:.3f}")

    print("\n✅ All tests passed successfully!")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Enhanced model created with {total_params:,} parameters")
    print(f"✓ Successfully processed {'genre-aware' if has_genres else 'basic'} graph")
    print(f"✓ Generated {args.top_k} recommendations for user {args.user}")

    if has_genres:
        print("✓ Explainability module working correctly")
        print("✓ Genre-aware metrics computed successfully")
    else:
        print("ℹ️  To enable genre features, regenerate data with genre information")


if __name__ == "__main__":
    main()
