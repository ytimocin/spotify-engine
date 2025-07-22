"""Spotify Engine - Graph-based music recommendation system."""

from .common.explainability import RecommendationExplainer, format_explanation
from .common.metrics import evaluate_batch, ndcg_at_k, recall_at_k
from .common.metrics_extended import (
    coverage_metrics,
    evaluate_genre_aware_recommendations,
    genre_diversity_score,
    genre_precision_at_k,
)
from .common.models import EnhancedGATRecommender, GATRecommender
from .common.trainers import AdvancedTrainer, SimpleTrainer

__version__ = "0.2.0"

__all__ = [
    # Models
    "GATRecommender",
    "EnhancedGATRecommender",
    # Trainers
    "SimpleTrainer",
    "AdvancedTrainer",
    # Explainability
    "RecommendationExplainer",
    "format_explanation",
    # Metrics
    "recall_at_k",
    "ndcg_at_k",
    "evaluate_batch",
    "genre_diversity_score",
    "genre_precision_at_k",
    "coverage_metrics",
    "evaluate_genre_aware_recommendations",
]
