"""
Extended metrics for genre-aware recommendations.

Includes diversity, coverage, and genre-specific metrics.
"""

from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import torch


def genre_diversity_score(
    recommendations: torch.Tensor, song_genre_mapping: Dict[int, Set[int]], k: int = 10
) -> float:
    """
    Calculate genre diversity in recommendations.

    Higher score means more diverse genre representation.

    Args:
        recommendations: Tensor of recommended song indices [num_users, num_recs]
        song_genre_mapping: Dict mapping song_id to set of genre_ids
        k: Number of recommendations to consider

    Returns:
        Average genre diversity score (0-1)
    """
    diversity_scores = []

    for user_recs in recommendations:
        # Get genres for user's recommendations
        user_genres = set()
        for song_idx in user_recs[:k]:
            song_idx = song_idx.item()
            if song_idx in song_genre_mapping:
                user_genres.update(song_genre_mapping[song_idx])

        # Calculate diversity as ratio of unique genres to recommendations
        if len(user_recs[:k]) > 0:
            diversity = len(user_genres) / min(k, len(user_recs))
            diversity_scores.append(diversity)

    return np.mean(diversity_scores) if diversity_scores else 0.0


def genre_precision_at_k(
    recommendations: torch.Tensor,
    user_genre_preferences: Dict[int, Set[int]],
    song_genre_mapping: Dict[int, Set[int]],
    k: int = 10,
) -> float:
    """
    Calculate precision of genre matching in recommendations.

    Measures how well recommendations match user's genre preferences.

    Args:
        recommendations: Tensor of recommended song indices [num_users, num_recs]
        user_genre_preferences: Dict mapping user_id to preferred genre_ids
        song_genre_mapping: Dict mapping song_id to genre_ids
        k: Number of recommendations to consider

    Returns:
        Average genre precision score
    """
    precision_scores = []

    for user_idx, user_recs in enumerate(recommendations):
        user_genres = user_genre_preferences.get(user_idx, set())

        if not user_genres:
            continue

        matches = 0
        for song_idx in user_recs[:k]:
            song_idx = song_idx.item()
            song_genres = song_genre_mapping.get(song_idx, set())

            # Check if any of the song's genres match user preferences
            if user_genres & song_genres:
                matches += 1

        precision = matches / min(k, len(user_recs))
        precision_scores.append(precision)

    return np.mean(precision_scores) if precision_scores else 0.0


def intra_list_distance(
    recommendations: torch.Tensor, song_embeddings: torch.Tensor, k: int = 10
) -> float:
    """
    Calculate average pairwise distance between recommended items.

    Higher values indicate more diverse recommendations.

    Args:
        recommendations: Tensor of recommended song indices [num_users, num_recs]
        song_embeddings: Song embedding matrix [num_songs, embedding_dim]
        k: Number of recommendations to consider

    Returns:
        Average intra-list distance
    """
    distances = []

    for user_recs in recommendations:
        user_recs = user_recs[:k]

        if len(user_recs) < 2:
            continue

        # Get embeddings for recommended songs
        rec_embeddings = song_embeddings[user_recs]

        # Calculate pairwise distances
        user_distances = []
        for i in range(len(rec_embeddings)):
            for j in range(i + 1, len(rec_embeddings)):
                dist = torch.norm(rec_embeddings[i] - rec_embeddings[j]).item()
                user_distances.append(dist)

        if user_distances:
            distances.append(np.mean(user_distances))

    return np.mean(distances) if distances else 0.0


def coverage_metrics(
    all_recommendations: List[torch.Tensor], num_items: int, k: int = 10
) -> Dict[str, float]:
    """
    Calculate catalog coverage metrics.

    Args:
        all_recommendations: List of recommendation tensors across all users
        num_items: Total number of items in catalog
        k: Number of recommendations per user

    Returns:
        Dict with coverage metrics
    """
    recommended_items = set()
    recommendation_counts = defaultdict(int)

    for recs in all_recommendations:
        for song_idx in recs[:k]:
            song_idx = song_idx.item()
            recommended_items.add(song_idx)
            recommendation_counts[song_idx] += 1

    # Simple coverage: percentage of catalog recommended
    simple_coverage = len(recommended_items) / num_items

    # Gini coefficient for recommendation distribution
    counts = list(recommendation_counts.values())
    counts.sort()
    n = len(counts)

    if n > 0 and sum(counts) > 0:
        cumsum = np.cumsum(counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    else:
        gini = 0.0

    # Entropy of recommendation distribution
    if sum(counts) > 0:
        probs = np.array(counts) / sum(counts)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(counts))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = 0.0

    return {
        "catalog_coverage": simple_coverage,
        "gini_coefficient": gini,
        "normalized_entropy": normalized_entropy,
        "unique_items_recommended": len(recommended_items),
    }


def novelty_score(
    recommendations: torch.Tensor, item_popularity: Dict[int, float], k: int = 10
) -> float:
    """
    Calculate novelty score based on item popularity.

    Higher scores indicate recommendations of less popular items.

    Args:
        recommendations: Tensor of recommended song indices
        item_popularity: Dict mapping item_id to popularity score (0-1)
        k: Number of recommendations to consider

    Returns:
        Average novelty score
    """
    novelty_scores = []

    for user_recs in recommendations:
        user_novelty = []

        for song_idx in user_recs[:k]:
            song_idx = song_idx.item()
            # Novelty is inverse of popularity
            popularity = item_popularity.get(song_idx, 0.5)
            novelty = 1 - popularity
            user_novelty.append(novelty)

        if user_novelty:
            novelty_scores.append(np.mean(user_novelty))

    return np.mean(novelty_scores) if novelty_scores else 0.0


def genre_affinity_recall(  # noqa: C901
    recommendations: torch.Tensor,
    ground_truth: Dict[int, Set[int]],
    user_genre_affinities: Dict[int, Dict[int, float]],
    song_genre_mapping: Dict[int, Set[int]],
    k: int = 10,
    affinity_threshold: float = 0.7,
) -> float:
    """
    Calculate recall weighted by genre affinity scores.

    Gives more weight to recommendations that match high-affinity genres.

    Args:
        recommendations: Recommended items
        ground_truth: True user-item interactions
        user_genre_affinities: User's affinity scores for each genre
        song_genre_mapping: Song to genre mapping
        k: Number of recommendations
        affinity_threshold: Minimum affinity to consider

    Returns:
        Weighted recall score
    """
    weighted_recalls = []

    for user_idx, user_recs in enumerate(recommendations):
        if user_idx not in ground_truth or not ground_truth[user_idx]:
            continue

        user_affinities = user_genre_affinities.get(user_idx, {})
        relevant_items = ground_truth[user_idx]

        weighted_hits = 0.0
        total_weight = 0.0

        for song_idx in user_recs[:k]:
            song_idx = song_idx.item()

            if song_idx in relevant_items:
                # Calculate weight based on genre affinity
                song_genres = song_genre_mapping.get(song_idx, set())

                if song_genres and user_affinities:
                    # Average affinity across song's genres
                    affinities = [
                        user_affinities.get(g, 0.0)
                        for g in song_genres
                        if user_affinities.get(g, 0.0) >= affinity_threshold
                    ]

                    if affinities:
                        weight = np.mean(affinities)
                        weighted_hits += weight
                    else:
                        # Default weight for songs without high-affinity genres
                        weighted_hits += 0.5
                else:
                    # Default weight if no genre info
                    weighted_hits += 1.0

        # Calculate maximum possible weight
        for item in list(relevant_items)[:k]:
            song_genres = song_genre_mapping.get(item, set())
            if song_genres and user_affinities:
                affinities = [user_affinities.get(g, 0.0) for g in song_genres]
                if affinities:
                    total_weight += max(affinities)
                else:
                    total_weight += 0.5
            else:
                total_weight += 1.0

        if total_weight > 0:
            weighted_recalls.append(weighted_hits / total_weight)

    return np.mean(weighted_recalls) if weighted_recalls else 0.0


def evaluate_genre_aware_recommendations(
    model, graph, test_users: List[int], k: int = 10
) -> Dict[str, float]:
    """
    Comprehensive evaluation of genre-aware recommendations.

    Args:
        model: Trained recommendation model
        graph: PyTorch Geometric HeteroData
        test_users: List of user indices to evaluate
        k: Number of recommendations

    Returns:
        Dict of evaluation metrics
    """
    # Prepare data structures
    song_genre_mapping = {}
    user_genre_preferences = {}
    user_genre_affinities = {}

    if ("song", "has", "genre") in graph.edge_types:
        edge_index = graph[("song", "has", "genre")].edge_index
        for i in range(edge_index.shape[1]):
            song_idx = edge_index[0, i].item()
            genre_idx = edge_index[1, i].item()
            if song_idx not in song_genre_mapping:
                song_genre_mapping[song_idx] = set()
            song_genre_mapping[song_idx].add(genre_idx)

    if ("user", "prefers", "genre") in graph.edge_types:
        edge_index = graph[("user", "prefers", "genre")].edge_index
        edge_attr = graph[("user", "prefers", "genre")].edge_attr

        for i in range(edge_index.shape[1]):
            user_idx = edge_index[0, i].item()
            genre_idx = edge_index[1, i].item()

            if user_idx not in user_genre_preferences:
                user_genre_preferences[user_idx] = set()
                user_genre_affinities[user_idx] = {}

            user_genre_preferences[user_idx].add(genre_idx)

            if edge_attr is not None and len(edge_attr) > i:
                affinity = edge_attr[i].item()
                user_genre_affinities[user_idx][genre_idx] = affinity

    # Get recommendations for test users
    all_recommendations = []
    x_dict = {
        "user": torch.arange(graph["user"].num_nodes),
        "song": torch.arange(graph["song"].num_nodes),
        "artist": torch.arange(graph["artist"].num_nodes),
    }
    if "genre" in graph.node_types:
        x_dict["genre"] = torch.arange(graph["genre"].num_nodes)

    model.eval()
    with torch.no_grad():
        embeddings, _ = model(x_dict, graph)

        for user_idx in test_users:
            user_emb = embeddings["user"][user_idx]
            scores = torch.matmul(embeddings["song"], user_emb)
            _, recs = torch.topk(scores, k)
            all_recommendations.append(recs)

    # Stack recommendations
    recommendations = torch.stack(all_recommendations)

    # Calculate metrics
    metrics = {
        "genre_diversity": genre_diversity_score(recommendations, song_genre_mapping, k),
        "genre_precision": genre_precision_at_k(
            recommendations, user_genre_preferences, song_genre_mapping, k
        ),
        "intra_list_distance": intra_list_distance(recommendations, embeddings["song"], k),
    }

    # Add coverage metrics
    coverage = coverage_metrics(all_recommendations, graph["song"].num_nodes, k)
    metrics.update({f"coverage_{k}": v for k, v in coverage.items()})

    return metrics
