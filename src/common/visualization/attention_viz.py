"""
Attention visualization utilities for multi-hop explanations.

Provides tools to visualize attention weights and paths in the recommendation graph.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch


def visualize_attention_paths(
    graph,
    user_idx: int,
    song_idx: int,
    attention_weights: Dict,
    node_labels: Optional[Dict] = None,
    max_paths: int = 5,
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Visualize attention paths from user to recommended song.

    Args:
        graph: PyTorch Geometric HeteroData
        user_idx: User node index
        song_idx: Recommended song index
        attention_weights: Dictionary of attention weights by edge type
        node_labels: Optional labels for nodes
        max_paths: Maximum number of paths to show
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    # Create NetworkX graph for visualization
    G = nx.MultiDiGraph()

    # Add nodes with types
    G.add_node(f"user_{user_idx}", node_type="user", label=f"User {user_idx}")
    G.add_node(f"song_{song_idx}", node_type="song", label=f"Song {song_idx}")

    # Find paths through genres and artists
    paths = []

    # Path 1: User -> Genre -> Song
    if ("user", "prefers", "genre") in graph.edge_types and (
        "song",
        "has",
        "genre",
    ) in graph.edge_types:
        user_genre_edges = graph[("user", "prefers", "genre")].edge_index
        song_genre_edges = graph[("song", "has", "genre")].edge_index

        # Find user's genres
        user_genres = user_genre_edges[1][user_genre_edges[0] == user_idx]
        # Find song's genres
        song_genres = song_genre_edges[1][song_genre_edges[0] == song_idx]

        # Find common genres
        common_genres = set(user_genres.tolist()) & set(song_genres.tolist())

        for genre_idx in list(common_genres)[:max_paths]:
            G.add_node(f"genre_{genre_idx}", node_type="genre", label=f"Genre {genre_idx}")

            # Add edges with attention weights if available
            weight1 = 1.0  # Default weight
            weight2 = 1.0

            G.add_edge(
                f"user_{user_idx}", f"genre_{genre_idx}", edge_type="prefers", weight=weight1
            )
            G.add_edge(
                f"genre_{genre_idx}", f"song_{song_idx}", edge_type="influences", weight=weight2
            )

            paths.append(
                {
                    "path": [f"user_{user_idx}", f"genre_{genre_idx}", f"song_{song_idx}"],
                    "weight": weight1 * weight2,
                }
            )

    # Path 2: User -> Song (direct)
    if ("user", "listens", "song") in graph.edge_types:
        user_song_edges = graph[("user", "listens", "song")].edge_index
        direct_edge = (user_song_edges[0] == user_idx) & (user_song_edges[1] == song_idx)

        if direct_edge.any():
            G.add_edge(f"user_{user_idx}", f"song_{song_idx}", edge_type="listens", weight=1.0)
            paths.append({"path": [f"user_{user_idx}", f"song_{song_idx}"], "weight": 1.0})

    # Path 3: User -> Song -> Artist -> Song
    if ("song", "by", "artist") in graph.edge_types:
        song_artist_edges = graph[("song", "by", "artist")].edge_index

        # Find artist of recommended song
        song_artists = song_artist_edges[1][song_artist_edges[0] == song_idx]

        if len(song_artists) > 0:
            artist_idx = song_artists[0].item()
            G.add_node(f"artist_{artist_idx}", node_type="artist", label=f"Artist {artist_idx}")

            # Find other songs by same artist that user has listened to
            artist_songs = song_artist_edges[0][song_artist_edges[1] == artist_idx]
            user_songs = user_song_edges[1][user_song_edges[0] == user_idx]

            common_songs = set(artist_songs.tolist()) & set(user_songs.tolist())
            common_songs.discard(song_idx)  # Remove the recommended song

            for other_song in list(common_songs)[:2]:  # Show max 2 other songs
                G.add_node(f"song_{other_song}", node_type="song", label=f"Song {other_song}")
                G.add_edge(
                    f"user_{user_idx}", f"song_{other_song}", edge_type="listens", weight=0.8
                )
                G.add_edge(f"song_{other_song}", f"artist_{artist_idx}", edge_type="by", weight=1.0)
                G.add_edge(
                    f"artist_{artist_idx}", f"song_{song_idx}", edge_type="creates", weight=1.0
                )

    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes by type
    node_colors = {"user": "#ff6b6b", "song": "#4ecdc4", "artist": "#45b7d1", "genre": "#96ceb4"}

    for node_type, color in node_colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == node_type]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes, node_color=color, node_size=1000, alpha=0.9, ax=ax
        )

    # Draw edges with varying thickness based on attention
    edges = G.edges(data=True)
    for u, v, data in edges:
        weight = data.get("weight", 1.0)
        nx.draw_networkx_edges(
            G, pos, [(u, v)], width=weight * 3, alpha=0.6, edge_color="gray", ax=ax
        )

    # Draw labels
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

    # Add title and legend
    ax.set_title(
        f"Recommendation Paths: User {user_idx} → Song {song_idx}", fontsize=16, fontweight="bold"
    )

    # Create legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=node_colors["user"], label="User"),
        Patch(facecolor=node_colors["song"], label="Song"),
        Patch(facecolor=node_colors["artist"], label="Artist"),
        Patch(facecolor=node_colors["genre"], label="Genre"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    ax.axis("off")
    plt.tight_layout()

    return fig


def plot_attention_heatmap(
    attention_matrix: torch.Tensor,
    source_labels: List[str],
    target_labels: List[str],
    title: str = "Attention Weights",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = "YlOrRd",
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.

    Args:
        attention_matrix: Attention weights [num_sources, num_targets]
        source_labels: Labels for source nodes
        target_labels: Labels for target nodes
        title: Plot title
        figsize: Figure size
        cmap: Colormap

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Convert to numpy if needed
    if torch.is_tensor(attention_matrix):
        attention_matrix = attention_matrix.detach().cpu().numpy()

    # Create heatmap
    im = ax.imshow(attention_matrix, cmap=cmap, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(target_labels)))
    ax.set_yticks(np.arange(len(source_labels)))
    ax.set_xticklabels(target_labels, rotation=45, ha="right")
    ax.set_yticklabels(source_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", rotation=270, labelpad=20)

    # Add title
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Target Nodes")
    ax.set_ylabel("Source Nodes")

    # Add text annotations for values
    if attention_matrix.shape[0] <= 20 and attention_matrix.shape[1] <= 20:
        for i in range(attention_matrix.shape[0]):
            for j in range(attention_matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{attention_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    plt.tight_layout()
    return fig


def analyze_genre_attention(model, graph, user_idx: int, top_k: int = 10) -> pd.DataFrame:
    """
    Analyze how genres influence recommendations for a user.

    Args:
        model: Trained recommendation model
        graph: PyTorch Geometric HeteroData
        user_idx: User to analyze
        top_k: Number of top recommendations to analyze

    Returns:
        DataFrame with genre influence analysis
    """
    # Get node indices
    x_dict = {
        "user": torch.arange(graph["user"].num_nodes),
        "song": torch.arange(graph["song"].num_nodes),
        "artist": torch.arange(graph["artist"].num_nodes),
    }
    if "genre" in graph.node_types:
        x_dict["genre"] = torch.arange(graph["genre"].num_nodes)

    # Get recommendations
    with torch.no_grad():
        embeddings, _ = model(x_dict, graph)

        # Get user embedding
        user_emb = embeddings["user"][user_idx]

        # Score all songs
        song_scores = torch.matmul(embeddings["song"], user_emb)
        top_songs = torch.topk(song_scores, top_k).indices

        # Analyze genre influence
        results = []

        if "genre" in embeddings:
            # Get user's genre preferences
            user_genre_edges = graph[("user", "prefers", "genre")].edge_index
            user_genres = user_genre_edges[1][user_genre_edges[0] == user_idx]

            for song_idx in top_songs:
                song_idx = song_idx.item()

                # Get song's genres
                song_genre_edges = graph[("song", "has", "genre")].edge_index
                song_genres = song_genre_edges[1][song_genre_edges[0] == song_idx]

                # Calculate genre overlap and influence
                common_genres = set(user_genres.tolist()) & set(song_genres.tolist())

                genre_score = 0.0
                if len(common_genres) > 0:
                    # Average similarity between user and common genres
                    for genre_idx in common_genres:
                        genre_emb = embeddings["genre"][genre_idx]
                        genre_score += torch.cosine_similarity(
                            user_emb.unsqueeze(0), genre_emb.unsqueeze(0)
                        ).item()
                    genre_score /= len(common_genres)

                results.append(
                    {
                        "song_idx": song_idx,
                        "score": song_scores[song_idx].item(),
                        "num_common_genres": len(common_genres),
                        "genre_influence_score": genre_score,
                        "common_genre_ids": list(common_genres),
                    }
                )

        return pd.DataFrame(results)


def plot_multi_hop_explanation(explanation: Dict, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Create a visual explanation of why a song was recommended.

    Args:
        explanation: Output from model.explain_recommendation()
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Score breakdown
    components = []
    values = []

    if explanation.get("genre_influence"):
        total_genre_influence = sum(g["contribution"] for g in explanation["genre_influence"])
        components.append("Genre Match")
        values.append(total_genre_influence)

    if explanation.get("artist_influence"):
        components.append("Artist Similarity")
        values.append(explanation["artist_influence"]["similarity"])

    # Add base similarity
    base_score = explanation["score"] - sum(values)
    components.append("Base Similarity")
    values.append(base_score)

    # Create pie chart
    colors = plt.cm.Set3(range(len(components)))
    ax1.pie(values, labels=components, colors=colors, autopct="%1.1f%%", startangle=90)
    ax1.set_title("Recommendation Score Breakdown")

    # Plot 2: Genre contributions
    if explanation.get("genre_influence") and len(explanation["genre_influence"]) > 0:
        genres = [f"Genre {g['genre_idx']}" for g in explanation["genre_influence"]]
        contributions = [g["contribution"] for g in explanation["genre_influence"]]

        ax2.barh(genres, contributions, color=colors[0])
        ax2.set_xlabel("Contribution Score")
        ax2.set_title("Genre Contributions")
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.text(
            0.5, 0.5, "No genre data available", ha="center", va="center", transform=ax2.transAxes
        )
        ax2.set_title("Genre Contributions")
        ax2.axis("off")

    plt.suptitle(
        f"Why Song {explanation['song_idx']} was recommended to User {explanation['user_idx']}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    return fig
