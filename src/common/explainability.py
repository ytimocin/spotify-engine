"""
Explainability module for recommendation reasoning.

Provides human-readable explanations for recommendations based on:
- Genre preferences
- Artist connections
- Listening patterns
- Multi-hop graph paths
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch


class RecommendationExplainer:
    """Generate human-readable explanations for recommendations."""

    def __init__(self, graph, model, metadata: Optional[Dict] = None):
        """
        Initialize explainer.

        Args:
            graph: PyTorch Geometric HeteroData
            model: Trained recommendation model
            metadata: Optional metadata (genre names, artist names, etc.)
        """
        self.graph = graph
        self.model = model
        self.metadata = metadata or {}

        # Cache for efficiency
        self._edge_cache = {}
        self._build_edge_cache()

    def _build_edge_cache(self):
        """Build edge index cache for faster lookups."""
        for edge_type in self.graph.edge_types:
            edge_index = self.graph[edge_type].edge_index
            self._edge_cache[edge_type] = edge_index

    def explain_recommendation(
        self, user_idx: int, song_idx: int, top_k_reasons: int = 3
    ) -> Dict[str, any]:
        """
        Generate comprehensive explanation for why a song was recommended.

        Args:
            user_idx: User index
            song_idx: Song index
            top_k_reasons: Number of top reasons to include

        Returns:
            Dictionary with explanation details
        """
        explanation = {
            "user_id": user_idx,
            "song_id": song_idx,
            "reasons": [],
            "score": 0.0,
            "confidence": 0.0,
        }

        # Get embeddings
        x_dict = self._create_node_indices()
        with torch.no_grad():
            embeddings, _ = self.model(x_dict, self.graph)

        # Calculate recommendation score
        user_emb = embeddings["user"][user_idx]
        song_emb = embeddings["song"][song_idx]
        score = torch.cosine_similarity(user_emb, song_emb, dim=0).item()
        explanation["score"] = score

        # 1. Genre-based reasoning
        genre_reason = self._explain_genre_match(user_idx, song_idx, embeddings)
        if genre_reason["strength"] > 0:
            explanation["reasons"].append(genre_reason)

        # 2. Artist-based reasoning
        artist_reason = self._explain_artist_connection(user_idx, song_idx, embeddings)
        if artist_reason["strength"] > 0:
            explanation["reasons"].append(artist_reason)

        # 3. Collaborative filtering reasoning
        collab_reason = self._explain_collaborative(user_idx, song_idx, embeddings)
        if collab_reason["strength"] > 0:
            explanation["reasons"].append(collab_reason)

        # 4. Listening pattern reasoning
        pattern_reason = self._explain_listening_patterns(user_idx, song_idx)
        if pattern_reason["strength"] > 0:
            explanation["reasons"].append(pattern_reason)

        # Sort reasons by strength and keep top k
        explanation["reasons"].sort(key=lambda x: x["strength"], reverse=True)
        explanation["reasons"] = explanation["reasons"][:top_k_reasons]

        # Calculate confidence based on number and strength of reasons
        if explanation["reasons"]:
            explanation["confidence"] = np.mean([r["strength"] for r in explanation["reasons"]])

        return explanation

    def _explain_genre_match(
        self, user_idx: int, song_idx: int, embeddings: Dict[str, torch.Tensor]
    ) -> Dict:
        """Explain recommendation based on genre preferences."""
        reason = {"type": "genre_match", "description": "", "strength": 0.0, "details": {}}

        if "genre" not in self.graph.node_types:
            return reason

        # Get user's preferred genres
        user_genre_edge = self._edge_cache.get(("user", "prefers", "genre"))
        if user_genre_edge is None:
            return reason

        user_genres = user_genre_edge[1][user_genre_edge[0] == user_idx]

        # Get song's genres
        song_genre_edge = self._edge_cache.get(("song", "has", "genre"))
        if song_genre_edge is None:
            return reason

        song_genres = song_genre_edge[1][song_genre_edge[0] == song_idx]

        # Find common genres
        common_genres = set(user_genres.tolist()) & set(song_genres.tolist())

        if common_genres:
            # Calculate strength based on genre affinity
            total_affinity = 0.0
            genre_names = []

            for genre_idx in common_genres:
                # Get user's affinity for this genre
                if hasattr(self.graph[("user", "prefers", "genre")], "edge_attr"):
                    edge_mask = (user_genre_edge[0] == user_idx) & (user_genre_edge[1] == genre_idx)
                    if edge_mask.any():
                        affinity = (
                            self.graph[("user", "prefers", "genre")].edge_attr[edge_mask][0].item()
                        )
                        total_affinity += affinity

                # Get genre name
                genre_name = self._get_genre_name(genre_idx)
                genre_names.append(genre_name)

            avg_affinity = total_affinity / len(common_genres) if common_genres else 0

            reason["strength"] = min(avg_affinity, 1.0)
            reason["description"] = (
                f"This song matches your preferred genres: {', '.join(genre_names[:3])}"
            )
            reason["details"] = {
                "common_genres": list(common_genres),
                "genre_names": genre_names,
                "average_affinity": avg_affinity,
            }

        return reason

    def _explain_artist_connection(
        self, user_idx: int, song_idx: int, embeddings: Dict[str, torch.Tensor]
    ) -> Dict:
        """Explain recommendation based on artist connections."""
        reason = {"type": "artist_connection", "description": "", "strength": 0.0, "details": {}}

        # Get song's artist
        song_artist_edge = self._edge_cache.get(("song", "by", "artist"))
        if song_artist_edge is None:
            return reason

        song_artists = song_artist_edge[1][song_artist_edge[0] == song_idx]
        if len(song_artists) == 0:
            return reason

        artist_idx = song_artists[0].item()

        # Check if user has listened to other songs by this artist
        user_song_edge = self._edge_cache.get(("user", "listens", "song"))
        if user_song_edge is None:
            return reason

        user_songs = user_song_edge[1][user_song_edge[0] == user_idx]

        # Find other songs by the same artist
        artist_songs = song_artist_edge[0][song_artist_edge[1] == artist_idx]
        other_songs = set(artist_songs.tolist()) & set(user_songs.tolist())
        other_songs.discard(song_idx)  # Remove the recommended song

        if other_songs:
            # Calculate strength based on listening history
            listen_count = len(other_songs)
            strength = min(listen_count / 5.0, 1.0)  # Normalize to [0, 1]

            artist_name = self._get_artist_name(artist_idx)

            reason["strength"] = strength
            reason["description"] = f"You've enjoyed {listen_count} other songs by {artist_name}"
            reason["details"] = {
                "artist_id": artist_idx,
                "artist_name": artist_name,
                "other_songs_count": listen_count,
                "other_song_ids": list(other_songs)[:5],  # Limit to 5
            }

        return reason

    def _explain_collaborative(
        self, user_idx: int, song_idx: int, embeddings: Dict[str, torch.Tensor]
    ) -> Dict:
        """Explain based on similar users' preferences."""
        reason = {
            "type": "collaborative_filtering",
            "description": "",
            "strength": 0.0,
            "details": {},
        }

        # Find similar users
        user_emb = embeddings["user"][user_idx]
        all_user_embs = embeddings["user"]

        # Calculate similarities
        similarities = torch.cosine_similarity(user_emb.unsqueeze(0), all_user_embs)
        similarities[user_idx] = -1  # Exclude self

        # Get top similar users
        top_k = 10
        top_similarities, top_users = torch.topk(similarities, min(top_k, len(similarities)))

        # Check how many of these users listened to the recommended song
        user_song_edge = self._edge_cache.get(("user", "listens", "song"))
        if user_song_edge is None:
            return reason

        similar_listeners = 0
        for similar_user in top_users:
            if similar_user == user_idx:
                continue
            user_songs = user_song_edge[1][user_song_edge[0] == similar_user]
            if song_idx in user_songs:
                similar_listeners += 1

        if similar_listeners > 0:
            percentage = (similar_listeners / top_k) * 100
            strength = min(similar_listeners / 5.0, 1.0)

            reason["strength"] = strength
            reason["description"] = (
                f"{percentage:.0f}% of users with similar taste enjoyed this song"
            )
            reason["details"] = {
                "similar_users_count": similar_listeners,
                "total_similar_users": top_k,
                "average_similarity": top_similarities.mean().item(),
            }

        return reason

    def _explain_listening_patterns(self, user_idx: int, song_idx: int) -> Dict:
        """Explain based on user's listening patterns."""
        reason = {"type": "listening_pattern", "description": "", "strength": 0.0, "details": {}}

        # Get user's listening history
        user_song_edge = self._edge_cache.get(("user", "listens", "song"))
        if user_song_edge is None:
            return reason

        # Check if user has already listened to this song
        edge_mask = (user_song_edge[0] == user_idx) & (user_song_edge[1] == song_idx)

        if edge_mask.any() and hasattr(self.graph[("user", "listens", "song")], "edge_attr"):
            # User has listened to this song before
            edge_attrs = self.graph[("user", "listens", "song")].edge_attr[edge_mask][0]

            if len(edge_attrs) >= 3:  # play_count, completion_ratio, edge_weight
                play_count = int(edge_attrs[0].item())
                completion_ratio = edge_attrs[1].item()

                if play_count > 5:
                    strength = min(play_count / 20.0, 1.0)
                    reason["strength"] = strength
                    reason["description"] = (
                        f"You've played this song {play_count} times with {completion_ratio:.0%} average completion"
                    )
                    reason["details"] = {
                        "play_count": play_count,
                        "completion_ratio": completion_ratio,
                    }

        return reason

    def _create_node_indices(self) -> Dict[str, torch.Tensor]:
        """Create node indices for model input."""
        indices = {
            "user": torch.arange(self.graph["user"].num_nodes),
            "song": torch.arange(self.graph["song"].num_nodes),
            "artist": torch.arange(self.graph["artist"].num_nodes),
        }
        if "genre" in self.graph.node_types:
            indices["genre"] = torch.arange(self.graph["genre"].num_nodes)
        return indices

    def _get_genre_name(self, genre_idx: int) -> str:
        """Get genre name from metadata or default."""
        if "genres" in self.metadata and genre_idx < len(self.metadata["genres"]):
            return self.metadata["genres"][genre_idx]
        return f"Genre {genre_idx}"

    def _get_artist_name(self, artist_idx: int) -> str:
        """Get artist name from metadata or default."""
        if "artists" in self.metadata and artist_idx < len(self.metadata["artists"]):
            return self.metadata["artists"][artist_idx]
        return f"Artist {artist_idx}"

    def _get_song_name(self, song_idx: int) -> str:
        """Get song name from metadata or default."""
        if "songs" in self.metadata and song_idx < len(self.metadata["songs"]):
            return self.metadata["songs"][song_idx]
        return f"Song {song_idx}"

    def generate_recommendation_report(self, user_idx: int, top_k: int = 10) -> pd.DataFrame:
        """
        Generate a detailed report of recommendations with explanations.

        Args:
            user_idx: User to generate recommendations for
            top_k: Number of recommendations

        Returns:
            DataFrame with recommendations and explanations
        """
        # Get recommendations
        x_dict = self._create_node_indices()

        with torch.no_grad():
            if hasattr(self.model, "recommend"):
                # Use model's recommend method
                rec_songs, rec_scores, _ = self.model.recommend(
                    user_idx, x_dict, self.graph, k=top_k
                )
            else:
                # Fallback to manual recommendation
                embeddings, _ = self.model(x_dict, self.graph)
                user_emb = embeddings["user"][user_idx]
                song_scores = torch.matmul(embeddings["song"], user_emb)
                rec_scores, rec_songs = torch.topk(song_scores, top_k)

        # Generate explanations for each recommendation
        results = []

        for rank, (song_idx, score) in enumerate(zip(rec_songs, rec_scores)):
            song_idx = song_idx.item()
            score = score.item()

            # Get explanation
            explanation = self.explain_recommendation(user_idx, song_idx)

            # Format main reason
            main_reason = ""
            if explanation["reasons"]:
                main_reason = explanation["reasons"][0]["description"]

            results.append(
                {
                    "rank": rank + 1,
                    "song_id": song_idx,
                    "song_name": self._get_song_name(song_idx),
                    "score": score,
                    "confidence": explanation["confidence"],
                    "main_reason": main_reason,
                    "num_reasons": len(explanation["reasons"]),
                }
            )

        return pd.DataFrame(results)


def format_explanation(explanation: Dict, verbose: bool = False) -> str:
    """
    Format explanation dictionary into human-readable text.

    Args:
        explanation: Output from explain_recommendation
        verbose: Whether to include detailed information

    Returns:
        Formatted explanation string
    """
    lines = []

    lines.append(f"Recommendation: Song {explanation['song_id']} for User {explanation['user_id']}")
    lines.append(f"Score: {explanation['score']:.3f} (Confidence: {explanation['confidence']:.1%})")
    lines.append("\nReasons:")

    for i, reason in enumerate(explanation["reasons"], 1):
        lines.append(f"\n{i}. {reason['description']}")
        lines.append(
            f"   Strength: {'█' * int(reason['strength'] * 10)}{'░' * (10 - int(reason['strength'] * 10))} ({reason['strength']:.1%})"
        )

        if verbose and reason.get("details"):
            lines.append("   Details:")
            for key, value in reason["details"].items():
                if isinstance(value, list) and len(value) > 5:
                    value = value[:5] + ["..."]
                lines.append(f"   - {key}: {value}")

    return "\n".join(lines)
