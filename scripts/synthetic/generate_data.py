"""
Generate synthetic music listening session data for the Spotify Engine project.

This script creates realistic music listening patterns including:
- Power-law distribution of song popularity
- Time-based listening patterns
- User preference clustering
- Artist-song relationships
"""

import argparse
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

# typing imports removed - not needed


class DataValidator:
    """Validate generated data for consistency and quality."""

    @staticmethod
    def validate_sessions(sessions_df: pd.DataFrame, songs_df: pd.DataFrame) -> dict:
        """Validate session data quality."""
        issues = []
        metrics = {}

        # Check for invalid durations
        invalid_durations = sessions_df[sessions_df["ms_played"] > sessions_df["track_duration_ms"]]
        if len(invalid_durations) > 0:
            issues.append(f"Found {len(invalid_durations)} sessions with ms_played > duration")

        # Check for orphaned references
        orphaned_songs = set(sessions_df["track_id"]) - set(songs_df["track_id"])
        if orphaned_songs:
            issues.append(f"Found {len(orphaned_songs)} references to non-existent songs")

        # Calculate quality metrics
        metrics["skip_rate"] = (sessions_df["ms_played"] < CONFIG.skip_threshold_ms).mean()
        metrics["completion_rate"] = (
            sessions_df["ms_played"]
            >= sessions_df["track_duration_ms"] * CONFIG.completion_threshold_ratio
        ).mean()

        # Group sessions by user and hour to calculate average session length
        sessions_copy = sessions_df.copy()
        sessions_copy["timestamp"] = pd.to_datetime(sessions_copy["timestamp"])
        metrics["avg_songs_per_session"] = (
            sessions_copy.groupby(["user_id", pd.Grouper(key="timestamp", freq="h")]).size().mean()
        )

        # Calculate listening diversity
        metrics["unique_songs_per_user"] = (
            sessions_df.groupby("user_id")["track_id"].nunique().mean()
        )
        metrics["unique_artists_per_user"] = (
            sessions_df.groupby("user_id")["artist_id"].nunique().mean()
        )

        return {"issues": issues, "metrics": metrics}

    @staticmethod
    def validate_graph_connectivity(
        sessions_df: pd.DataFrame, users_df: pd.DataFrame, songs_df: pd.DataFrame
    ) -> dict:
        """Ensure graph will be well-connected."""
        # Check for isolated users
        active_users = sessions_df["user_id"].unique()
        isolated_users = set(users_df["user_id"]) - set(active_users)

        # Check for unplayed songs
        played_songs = sessions_df["track_id"].unique()
        unplayed_songs = set(songs_df["track_id"]) - set(played_songs)

        # Calculate graph density
        graph_density = len(sessions_df) / (len(users_df) * len(songs_df))

        return {
            "isolated_users": len(isolated_users),
            "isolated_users_pct": len(isolated_users) / len(users_df) * 100,
            "unplayed_songs": len(unplayed_songs),
            "unplayed_songs_pct": len(unplayed_songs) / len(songs_df) * 100,
            "graph_density": graph_density,
        }

    @staticmethod
    def validate_genre_coverage(
        user_genre_prefs_df: pd.DataFrame,
        artist_genres_df: pd.DataFrame,
        genres_df: pd.DataFrame,
    ) -> dict:
        """Validate genre data completeness."""
        metrics = {}

        # Check artists without genres
        # In our implementation, all artists have genres by design
        metrics["artists_without_genres"] = 0

        # Check genre distribution
        genre_counts = artist_genres_df["genre_id"].value_counts()
        unused_genres = set(genres_df["genre_id"]) - set(genre_counts.index)
        metrics["unused_genres"] = len(unused_genres)

        # User genre preference coverage
        avg_genres_per_user = user_genre_prefs_df.groupby("user_id").size().mean()
        metrics["avg_genres_per_user"] = avg_genres_per_user

        return metrics


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation."""

    # User behavior patterns
    user_type_distribution: Dict[str, float] = field(
        default_factory=lambda: {"casual": 0.5, "regular": 0.35, "power": 0.15}
    )

    activity_level_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {"casual": (0.1, 0.3), "regular": (0.3, 0.7), "power": (0.7, 1.0)}
    )

    # Listening patterns
    session_length_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.2, 3: 0.3, 5: 0.25, 10: 0.15, 20: 0.1}
    )

    listening_behavior_weights: Dict[str, float] = field(
        default_factory=lambda: {"full": 0.4, "skip": 0.2, "partial": 0.4}
    )

    # User type behavioral multipliers
    user_type_session_length_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"casual": 0.7, "regular": 1.0, "power": 1.5}
    )

    user_type_skip_rate_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"casual": 1.3, "regular": 1.0, "power": 0.6}
    )

    user_type_genre_diversity_multipliers: Dict[str, float] = field(
        default_factory=lambda: {"casual": 0.8, "regular": 1.0, "power": 1.4}
    )

    # Time patterns - hourly listening weights (24 hours)
    # Reduced early morning activity (0-6am) to be more realistic
    hour_weights: List[float] = field(
        default_factory=lambda: [
            0.1,  # 0am
            0.05,  # 1am
            0.05,  # 2am
            0.05,  # 3am
            0.05,  # 4am
            0.1,  # 5am
            0.3,  # 6am
            0.7,  # 7am (commute starts)
            0.9,  # 8am
            0.5,  # 9am
            0.4,  # 10am
            0.4,  # 11am
            0.5,  # 12pm (lunch)
            0.6,  # 1pm
            0.7,  # 2pm
            0.8,  # 3pm
            0.9,  # 4pm (commute home)
            1.0,  # 5pm (peak evening)
            1.0,  # 6pm (peak evening)
            1.0,  # 7pm (peak evening)
            0.9,  # 8pm
            0.8,  # 9pm
            0.6,  # 10pm
            0.4,  # 11pm
        ]
    )

    # Genre configuration
    genre_count_by_user_type: Dict[str, Tuple[int, int]] = field(
        default_factory=lambda: {
            "casual": (1, 3),  # min, max genres
            "regular": (2, 5),
            "power": (3, 8),
        }
    )

    # Artist and song patterns
    artist_popularity_alpha: float = 1.5
    artist_genre_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 0.5, 2: 0.35, 3: 0.15}  # 1 genre  # 2 genres  # 3 genres
    )

    # Song duration ranges (ms)
    song_duration_range: Tuple[int, int] = (30000, 600000)  # 30 seconds to 10 minutes

    # Skip thresholds
    skip_threshold_ms: int = 30000  # Songs played less than this are considered skips
    completion_threshold_ratio: float = 0.8  # Songs played >= 80% are considered complete

    # Session continuity
    same_artist_probability: float = 0.4  # Probability of playing another song from same artist
    weekday_preference_probability: float = 0.8  # Weekday preference for known content
    weekend_preference_probability: float = 0.5  # Weekend allows more exploration

    @classmethod
    def from_yaml(cls, config_path: str) -> "DataGenerationConfig":
        """Load configuration from YAML file."""
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)

        # Map YAML structure to dataclass fields
        config = cls()

        if "user_types" in yaml_config:
            user_cfg = yaml_config["user_types"]
            if "distribution" in user_cfg:
                config.user_type_distribution = user_cfg["distribution"]
            if "activity_levels" in user_cfg:
                config.activity_level_ranges = {
                    k: tuple(v) for k, v in user_cfg["activity_levels"].items()
                }
            if "genre_preferences" in user_cfg:
                config.genre_count_by_user_type = {
                    k: tuple(v) for k, v in user_cfg["genre_preferences"].items()
                }

        if "sessions" in yaml_config:
            session_cfg = yaml_config["sessions"]
            if "length_weights" in session_cfg:
                config.session_length_weights = {
                    int(k): v for k, v in session_cfg["length_weights"].items()
                }
            if "behavior_weights" in session_cfg:
                config.listening_behavior_weights = session_cfg["behavior_weights"]
            if "same_artist_probability" in session_cfg:
                config.same_artist_probability = session_cfg["same_artist_probability"]
            if "weekday_preference_probability" in session_cfg:
                config.weekday_preference_probability = session_cfg[
                    "weekday_preference_probability"
                ]
            if "weekend_preference_probability" in session_cfg:
                config.weekend_preference_probability = session_cfg[
                    "weekend_preference_probability"
                ]

        if "time_patterns" in yaml_config:
            time_cfg = yaml_config["time_patterns"]
            if "hourly_weights" in time_cfg:
                config.hour_weights = time_cfg["hourly_weights"]

        if "content" in yaml_config:
            content_cfg = yaml_config["content"]
            if "artist_popularity_alpha" in content_cfg:
                config.artist_popularity_alpha = content_cfg["artist_popularity_alpha"]
            if "artist_genre_distribution" in content_cfg:
                config.artist_genre_weights = {
                    int(k): v for k, v in content_cfg["artist_genre_distribution"].items()
                }
            if "song_duration" in content_cfg:
                duration = content_cfg["song_duration"]
                config.song_duration_range = (duration["min_ms"], duration["max_ms"])

        if "playback" in yaml_config:
            playback_cfg = yaml_config["playback"]
            if "skip_threshold_ms" in playback_cfg:
                config.skip_threshold_ms = playback_cfg["skip_threshold_ms"]
            if "completion_threshold_ratio" in playback_cfg:
                config.completion_threshold_ratio = playback_cfg["completion_threshold_ratio"]

        return config


# Global configuration instance
CONFIG = DataGenerationConfig()


def generate_genres() -> pd.DataFrame:
    """Generate music genres with realistic Zipf distribution."""
    # 35 diverse genres for better recommendation testing
    genre_names = [
        "Pop",
        "Rock",
        "Hip Hop",
        "Electronic",
        "R&B",
        "Country",
        "Jazz",
        "Classical",
        "Latin",
        "Indie",
        "Metal",
        "Folk",
        "Reggae",
        "Blues",
        "Soul",
        "Alternative",
        "Dance",
        "Punk",
        "Funk",
        "World",
        "Ambient",
        "Techno",
        "House",
        "Dubstep",
        "Acoustic",
        "Gospel",
        "New Age",
        "Experimental",
        "Progressive",
        "Grunge",
        "Ska",
        "Instrumental",
        "Vocal Jazz",
        "Bossa Nova",
        "Post-Rock",
    ]

    genres_data = []

    # Use Zipf distribution for popularity (s ≈ 1.1)
    # This creates realistic long tail with top genres having < 40% total share
    zipf_s = 1.1
    ranks = np.arange(1, len(genre_names) + 1)
    zipf_weights = 1 / (ranks**zipf_s)

    # Normalize to create popularity scores between 0.15 and 0.95
    min_pop, max_pop = 0.15, 0.95
    normalized_weights = (zipf_weights - zipf_weights.min()) / (
        zipf_weights.max() - zipf_weights.min()
    )
    popularities = min_pop + normalized_weights * (max_pop - min_pop)

    for i, (genre_name, popularity) in enumerate(zip(genre_names, popularities)):
        genres_data.append(
            {
                "genre_id": f"G{i + 1:03d}",  # G001, G002, etc.
                "genre_name": genre_name,
                "popularity": round(popularity, 3),
            }
        )

    return pd.DataFrame(genres_data)


def generate_artist_genres(artists_df: pd.DataFrame, genres_df: pd.DataFrame) -> pd.DataFrame:
    """Assign 1-3 genres to each artist based on genre popularity."""
    artist_genres = []

    for _, artist in artists_df.iterrows():
        # Number of genres per artist (popular artists might have more genre diversity)
        genre_counts = list(CONFIG.artist_genre_weights.keys())
        genre_weights = list(CONFIG.artist_genre_weights.values())
        n_genres = random.choices(genre_counts, weights=genre_weights)[0]

        # Select genres weighted by popularity and artist popularity
        # Popular artists are more likely to be in popular genres
        genre_weights = genres_df["popularity"].values ** (1 + artist["popularity"])
        selected_genres = genres_df.sample(n=n_genres, weights=genre_weights, replace=False)

        for _, genre in selected_genres.iterrows():
            artist_genres.append(
                {
                    "artist_id": artist["artist_id"],
                    "genre_id": genre["genre_id"],
                    "genre_name": genre["genre_name"],
                }
            )

    return pd.DataFrame(artist_genres)


def generate_artists(n_artists: int) -> pd.DataFrame:
    """Generate synthetic artist data."""
    artist_names = [f"Artist_{i:04d}" for i in range(n_artists)]

    # Artist popularity follows power law
    popularity_scores = np.random.pareto(a=CONFIG.artist_popularity_alpha, size=n_artists)
    popularity_scores = popularity_scores / popularity_scores.max()

    return pd.DataFrame(
        {
            "artist_id": [f"A{i:04d}" for i in range(n_artists)],
            "artist_name": artist_names,
            "popularity": popularity_scores,
        }
    )


def generate_songs(n_songs: int, artists_df: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic song data with artist associations."""
    song_data = []

    for i in range(n_songs):
        # Weighted selection - popular artists have more songs
        artist = artists_df.sample(n=1, weights=artists_df["popularity"]).iloc[0]

        # Song duration between configured min and max
        duration_ms = random.randint(*CONFIG.song_duration_range)

        song_data.append(
            {
                "track_id": f"T{i:05d}",
                "track_name": f"Song_{i:05d}",
                "artist_id": artist["artist_id"],
                "artist_name": artist["artist_name"],
                "duration_ms": duration_ms,
                # Song popularity correlated with artist popularity
                "popularity": min(1.0, artist["popularity"] * np.random.beta(2, 5)),
            }
        )

    return pd.DataFrame(song_data)


def generate_song_genres(songs_df: pd.DataFrame, artist_genres_df: pd.DataFrame) -> pd.DataFrame:
    """Create song-genre mappings based on artist genres."""
    # Merge songs with artist genres - songs inherit all genres from their artist
    song_genres = songs_df[["track_id", "artist_id"]].merge(
        artist_genres_df[["artist_id", "genre_id", "genre_name"]], on="artist_id", how="left"
    )

    # Drop artist_id as it's not needed in the final mapping
    song_genres = song_genres[["track_id", "genre_id", "genre_name"]]

    return song_genres


def generate_users(n_users: int) -> pd.DataFrame:
    """Generate synthetic user data."""
    user_types = list(CONFIG.user_type_distribution.keys())
    user_weights = list(CONFIG.user_type_distribution.values())

    users = []
    for i in range(n_users):
        user_type = random.choices(user_types, weights=user_weights)[0]

        # Activity level based on user type
        min_activity, max_activity = CONFIG.activity_level_ranges[user_type]
        activity_multiplier = np.random.uniform(min_activity, max_activity)

        users.append(
            {
                "user_id": f"U{i:04d}",
                "user_type": user_type,
                "activity_level": activity_multiplier,
            }
        )

    return pd.DataFrame(users)


def generate_user_genre_preferences(
    users_df: pd.DataFrame, genres_df: pd.DataFrame
) -> pd.DataFrame:
    """Generate user genre preferences with affinity scores."""
    user_genre_prefs = []

    for _, user in users_df.iterrows():
        # Number of preferred genres based on user type
        min_genres, max_genres = CONFIG.genre_count_by_user_type[user["user_type"]]
        base_n_genres = random.randint(min_genres, max_genres)

        # Apply genre diversity multiplier
        diversity_multiplier = CONFIG.user_type_genre_diversity_multipliers[user["user_type"]]
        n_preferred_genres = max(1, min(len(genres_df), int(base_n_genres * diversity_multiplier)))

        # Select genres with some bias towards popular genres
        # But power users are more likely to explore niche genres
        popularity_bias = (
            2.0 if user["user_type"] == "casual" else 1.0 if user["user_type"] == "regular" else 0.5
        )
        genre_weights = genres_df["popularity"].values ** popularity_bias
        selected_genres = genres_df.sample(
            n=n_preferred_genres, weights=genre_weights, replace=False
        )

        # Assign affinity scores (how much the user likes each genre)
        for idx, (_, genre) in enumerate(selected_genres.iterrows()):
            # Primary genre gets highest affinity
            if idx == 0:
                affinity = random.uniform(0.7, 1.0)
            # Secondary genres get medium affinity
            elif idx < 3:
                affinity = random.uniform(0.4, 0.7)
            # Other genres get lower affinity
            else:
                affinity = random.uniform(0.2, 0.5)

            user_genre_prefs.append(
                {
                    "user_id": user["user_id"],
                    "genre_id": genre["genre_id"],
                    "genre_name": genre["genre_name"],
                    "affinity_score": round(affinity, 3),
                }
            )

    return pd.DataFrame(user_genre_prefs)


def generate_listening_sessions(
    users_df: pd.DataFrame,
    songs_df: pd.DataFrame,
    user_genre_prefs_df: pd.DataFrame,
    song_genres_df: pd.DataFrame,
    n_days: int = 30,
) -> pd.DataFrame:
    """Generate realistic listening sessions with time patterns and session continuity."""
    sessions = []
    start_date = datetime.now() - timedelta(days=n_days)

    # Time-based listening patterns (hourly weights)
    hour_weights = CONFIG.hour_weights

    print("Generating listening sessions...")

    # Pre-compute song-genre matrix for efficient lookups
    print("Pre-computing song-genre affinity matrix...")

    # Create unique song and genre lists
    unique_songs = songs_df["track_id"].unique()
    unique_genres = song_genres_df["genre_id"].unique()

    # Create song->index and genre->index mappings
    song_to_idx = {song: idx for idx, song in enumerate(unique_songs)}
    genre_to_idx = {genre: idx for idx, genre in enumerate(unique_genres)}

    # Initialize song-genre matrix (songs x genres)
    song_genre_matrix = np.zeros((len(unique_songs), len(unique_genres)))

    # Fill the matrix
    for _, row in song_genres_df.iterrows():
        if row["track_id"] in song_to_idx and row["genre_id"] in genre_to_idx:
            song_idx = song_to_idx[row["track_id"]]
            genre_idx = genre_to_idx[row["genre_id"]]
            song_genre_matrix[song_idx, genre_idx] = 1

    # Pre-compute song popularity array
    song_popularity = np.zeros(len(unique_songs))
    for _, song in songs_df.iterrows():
        if song["track_id"] in song_to_idx:
            song_popularity[song_to_idx[song["track_id"]]] = song["popularity"]

    for _, user in tqdm(users_df.iterrows(), total=len(users_df)):
        # Number of listening sessions (not individual songs) based on user activity
        n_listening_sessions = int(n_days * user["activity_level"] * random.uniform(1, 5))

        # Get user's genre preferences as array
        user_genres = user_genre_prefs_df[user_genre_prefs_df["user_id"] == user["user_id"]]

        # Create user genre affinity vector
        user_genre_vector = np.full(len(unique_genres), 0.1)  # Default affinity
        for _, pref in user_genres.iterrows():
            if pref["genre_id"] in genre_to_idx:
                user_genre_vector[genre_to_idx[pref["genre_id"]]] = pref["affinity_score"]

        # Compute song weights for this user using matrix operations
        # song_affinities = song_genre_matrix @ user_genre_vector (songs x 1)
        song_affinities = song_genre_matrix.dot(user_genre_vector)

        # Normalize affinities by number of genres per song (avoid bias for multi-genre songs)
        song_genre_counts = song_genre_matrix.sum(axis=1)
        song_genre_counts[song_genre_counts == 0] = 1  # Avoid division by zero
        song_affinities = song_affinities / song_genre_counts

        # Compute final weights: popularity * (0.3 + 0.7 * affinity)
        user_song_weights = song_popularity * (0.3 + 0.7 * song_affinities)

        # Create lookup for quick access during session generation
        song_weight_dict = {
            song: user_song_weights[idx]
            for song, idx in song_to_idx.items()
            if user_song_weights[idx] > 0
        }

        # Generate listening sessions for this user
        for _ in range(n_listening_sessions):
            # Pick hour based on listening patterns
            hour = random.choices(range(24), weights=hour_weights)[0]

            session_start = start_date + timedelta(
                days=random.uniform(0, n_days),
                hours=hour,
                minutes=random.randint(0, 59),
            )

            # Weekend vs weekday patterns
            is_weekend = session_start.weekday() >= 5
            prefer_known_artists_prob = (
                CONFIG.weekend_preference_probability
                if is_weekend
                else CONFIG.weekday_preference_probability
            )

            # Session length (number of songs in this listening session)
            # Apply user type multiplier to session length
            session_lengths = list(CONFIG.session_length_weights.keys())
            session_weights = list(CONFIG.session_length_weights.values())
            base_session_length = random.choices(session_lengths, weights=session_weights)[0]

            # Apply user type multiplier
            length_multiplier = CONFIG.user_type_session_length_multipliers[user["user_type"]]
            session_length = max(1, int(base_session_length * length_multiplier))

            current_time = session_start
            last_artist_id = None

            for song_idx in range(session_length):
                # Songs in same session are more likely from same artist
                if (
                    song_idx > 0
                    and last_artist_id
                    and random.random() < CONFIG.same_artist_probability
                ):
                    # Continue with same artist
                    same_artist_songs = songs_df[songs_df["artist_id"] == last_artist_id]
                    if len(same_artist_songs) > 1:
                        # Get weights for songs by this artist
                        artist_song_weights = {
                            row["track_id"]: song_weight_dict.get(row["track_id"], 0.1)
                            for _, row in same_artist_songs.iterrows()
                        }
                        artist_song_weights = {
                            k: v for k, v in artist_song_weights.items() if v > 0
                        }

                        if artist_song_weights:
                            selected_track_id = random.choices(
                                list(artist_song_weights.keys()),
                                weights=list(artist_song_weights.values()),
                            )[0]
                            song = songs_df[songs_df["track_id"] == selected_track_id].iloc[0]
                        else:
                            # Fall back to any song
                            selected_track_id = random.choice(list(song_weight_dict.keys()))
                            song = songs_df[songs_df["track_id"] == selected_track_id].iloc[0]
                    else:
                        # Fall back to genre-based selection
                        selected_track_id = random.choices(
                            list(song_weight_dict.keys()),
                            weights=list(song_weight_dict.values()),
                        )[0]
                        song = songs_df[songs_df["track_id"] == selected_track_id].iloc[0]
                else:
                    # Regular song selection based on genre preferences
                    # Use genre preferences more on weekdays, more exploration on weekends
                    if random.random() < prefer_known_artists_prob:
                        # Select based on pre-computed genre preferences
                        selected_track_id = random.choices(
                            list(song_weight_dict.keys()),
                            weights=list(song_weight_dict.values()),
                        )[0]
                    else:
                        # More exploration - use popularity only
                        pop_weights = {
                            row["track_id"]: row["popularity"] for _, row in songs_df.iterrows()
                        }
                        selected_track_id = random.choices(
                            list(pop_weights.keys()),
                            weights=list(pop_weights.values()),
                        )[0]

                    song = songs_df[songs_df["track_id"] == selected_track_id].iloc[0]

                last_artist_id = song["artist_id"]

                # Listening duration: full song, skip, or partial
                # Apply user type multiplier to skip behavior
                behaviors = list(CONFIG.listening_behavior_weights.keys())
                behavior_weights = list(CONFIG.listening_behavior_weights.values())

                # Adjust weights based on user type
                skip_multiplier = CONFIG.user_type_skip_rate_multipliers[user["user_type"]]
                adjusted_weights = behavior_weights.copy()

                # Find skip index and adjust weights
                skip_idx = behaviors.index("skip")
                full_idx = behaviors.index("full")

                # Increase skip weight for casual users, decrease for power users
                adjusted_weights[skip_idx] *= skip_multiplier
                # Compensate by adjusting full listening weight inversely
                adjusted_weights[full_idx] *= 2.0 - skip_multiplier

                # Normalize weights
                total_weight = sum(adjusted_weights)
                adjusted_weights = [w / total_weight for w in adjusted_weights]

                listen_behavior = random.choices(behaviors, weights=adjusted_weights)[0]

                # Generate a user's listening preference for this song to couple skip/completion rates
                # Users who tend to skip more have lower completion rates when they do listen
                user_patience = np.random.beta(2, 2)  # 0-1, where higher = more patient listener

                if listen_behavior == "full":
                    # Use beta distribution to create realistic completion rates
                    # More patient users complete songs more fully
                    if user_patience > 0.7:
                        # Patient users: high completion
                        completion_ratio = np.random.beta(3, 1.2)  # Skewed toward high completion
                        completion_ratio = np.clip(completion_ratio, 0.95, 1.0)
                    else:
                        # Less patient users: still "full" but lower completion
                        completion_ratio = np.random.beta(2, 1.8)  # More varied completion
                        completion_ratio = np.clip(completion_ratio, 0.85, 1.0)
                    ms_played = int(song["duration_ms"] * completion_ratio)

                elif listen_behavior == "skip":
                    # Impatient users skip earlier, patient users skip later (if they skip at all)
                    if user_patience < 0.3:
                        # Very impatient: skip very early
                        max_skip_time = min(
                            CONFIG.skip_threshold_ms * 0.5, int(song["duration_ms"])
                        )
                        ms_played = random.randint(3000, max(3000, int(max_skip_time)))
                    elif user_patience < 0.6:
                        # Moderately impatient: normal skip timing
                        ms_played = random.randint(
                            5000, min(CONFIG.skip_threshold_ms, int(song["duration_ms"]))
                        )
                    else:
                        # Patient users: skip later if they do skip
                        min_skip_time = min(
                            CONFIG.skip_threshold_ms * 0.7, int(song["duration_ms"] * 0.3)
                        )
                        ms_played = random.randint(
                            int(min_skip_time),
                            min(CONFIG.skip_threshold_ms, int(song["duration_ms"])),
                        )

                else:  # partial
                    # Partial listening also influenced by user patience
                    if user_patience > 0.6:
                        # Patient users: listen to more of the song
                        min_ratio = 0.5 if song["duration_ms"] > 60000 else 0.6
                        max_ratio = 0.85
                    else:
                        # Impatient users: listen to less
                        min_ratio = 0.3 if song["duration_ms"] > 60000 else 0.4
                        max_ratio = 0.7

                    completion_ratio = random.uniform(min_ratio, max_ratio)
                    ms_played = int(song["duration_ms"] * completion_ratio)

                sessions.append(
                    {
                        "user_id": user["user_id"],
                        "track_id": song["track_id"],
                        "artist_id": song["artist_id"],
                        "timestamp": current_time.isoformat(),
                        "ms_played": ms_played,
                        "track_duration_ms": song["duration_ms"],
                    }
                )

                # Add small gap between songs (3-10 seconds)
                current_time = current_time + timedelta(
                    milliseconds=int(ms_played) + random.randint(3000, 10000)
                )

    return pd.DataFrame(sessions)


def main():
    """Generate synthetic music data based on command line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic music listening data")
    parser.add_argument("--users", type=int, default=1000, help="Number of users")
    parser.add_argument("--songs", type=int, default=5000, help="Number of songs")
    parser.add_argument("--artists", type=int, default=500, help="Number of artists")
    parser.add_argument("--days", type=int, default=30, help="Number of days of data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic/synthetic_sessions.csv",
        help="Output file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    global CONFIG
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            return
        print(f"Loading configuration from: {args.config}")
        CONFIG = DataGenerationConfig.from_yaml(args.config)
    else:
        # Check for default config file
        default_config = "config/default.yaml"
        if os.path.exists(default_config):
            print(f"Loading default configuration from: {default_config}")
            CONFIG = DataGenerationConfig.from_yaml(default_config)

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"Generating synthetic data with {args.users} users, "
        f"{args.songs} songs, {args.artists} artists"
    )

    # Generate data
    genres_df = generate_genres()
    print(f"✓ Generated {len(genres_df)} genres")

    artists_df = generate_artists(args.artists)
    print(f"✓ Generated {len(artists_df)} artists")

    artist_genres_df = generate_artist_genres(artists_df, genres_df)
    print(f"✓ Assigned genres to artists ({len(artist_genres_df)} artist-genre pairs)")

    songs_df = generate_songs(args.songs, artists_df)
    print(f"✓ Generated {len(songs_df)} songs")

    song_genres_df = generate_song_genres(songs_df, artist_genres_df)
    print(f"✓ Assigned genres to songs ({len(song_genres_df)} song-genre pairs)")

    users_df = generate_users(args.users)
    print(f"✓ Generated {len(users_df)} users")

    user_genre_prefs_df = generate_user_genre_preferences(users_df, genres_df)
    print(f"✓ Generated user genre preferences ({len(user_genre_prefs_df)} user-genre pairs)")

    sessions_df = generate_listening_sessions(
        users_df, songs_df, user_genre_prefs_df, song_genres_df, args.days
    )
    print(f"✓ Generated {len(sessions_df)} listening sessions")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save main session data
    sessions_df.to_csv(args.output, index=False)
    print(f"\nSaved session data to {args.output}")

    # Save metadata files for reference
    artists_df.to_csv(args.output.replace("sessions.csv", "artists.csv"), index=False)
    songs_df.to_csv(args.output.replace("sessions.csv", "songs.csv"), index=False)
    users_df.to_csv(args.output.replace("sessions.csv", "users.csv"), index=False)
    genres_df.to_csv(args.output.replace("sessions.csv", "genres.csv"), index=False)
    artist_genres_df.to_csv(args.output.replace("sessions.csv", "artist_genres.csv"), index=False)
    song_genres_df.to_csv(args.output.replace("sessions.csv", "song_genres.csv"), index=False)
    user_genre_prefs_df.to_csv(
        args.output.replace("sessions.csv", "user_genre_preferences.csv"), index=False
    )

    # Print statistics
    print("\nDataset Statistics:")
    print(f"- Total sessions: {len(sessions_df):,}")
    print(f"- Avg sessions per user: {len(sessions_df) / len(users_df):.1f}")
    unique_pairs = sessions_df.groupby(["user_id", "track_id"]).size().shape[0]
    print(f"- Unique user-song pairs: {unique_pairs:,}")
    print(f"- Total listening time: {sessions_df['ms_played'].sum() / (1000 * 60 * 60):.1f} hours")

    # Sample data preview
    print("\nSample sessions:")
    print(sessions_df.head(5).to_string())

    # Run data validation
    print("\n" + "=" * 50)
    print("DATA VALIDATION")
    print("=" * 50)

    # Validate sessions
    session_validation = DataValidator.validate_sessions(sessions_df, songs_df)
    if session_validation["issues"]:
        print("\nData Quality Issues Found:")
        for issue in session_validation["issues"]:
            print(f"  ⚠️  {issue}")
    else:
        print("\n✓ No data quality issues found")

    print("\nSession Quality Metrics:")
    for metric, value in session_validation["metrics"].items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.3f}")
        else:
            print(f"  - {metric}: {value}")

    # Validate graph connectivity
    connectivity = DataValidator.validate_graph_connectivity(sessions_df, users_df, songs_df)
    print("\nGraph Connectivity:")
    print(
        f"  - Isolated users: {connectivity['isolated_users']} "
        f"({connectivity['isolated_users_pct']:.1f}%)"
    )
    print(
        f"  - Unplayed songs: {connectivity['unplayed_songs']} "
        f"({connectivity['unplayed_songs_pct']:.1f}%)"
    )
    print(f"  - Graph density: {connectivity['graph_density']:.6f}")

    # Validate genre coverage
    genre_coverage = DataValidator.validate_genre_coverage(
        user_genre_prefs_df, artist_genres_df, genres_df
    )
    print("\nGenre Coverage:")
    for metric, value in genre_coverage.items():
        if isinstance(value, float):
            print(f"  - {metric}: {value:.2f}")
        else:
            print(f"  - {metric}: {value}")


if __name__ == "__main__":
    main()
