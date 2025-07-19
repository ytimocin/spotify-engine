"""
Generate synthetic music listening session data for the Spotify Engine project.

This script creates realistic music listening patterns including:
- Power-law distribution of song popularity
- Time-based listening patterns
- User preference clustering
- Artist-song relationships
"""

import argparse
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import os
from tqdm import tqdm


def generate_artists(n_artists: int) -> pd.DataFrame:
    """Generate synthetic artist data."""
    artist_names = [f"Artist_{i:04d}" for i in range(n_artists)]

    # Artist popularity follows power law
    popularity_scores = np.random.pareto(a=1.5, size=n_artists)
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

        # Song duration between 30 seconds and 10 minutes
        duration_ms = random.randint(30000, 600000)

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


def generate_users(n_users: int) -> pd.DataFrame:
    """Generate synthetic user data."""
    user_types = ["casual", "regular", "power"]
    user_weights = [0.5, 0.35, 0.15]  # Most users are casual

    users = []
    for i in range(n_users):
        user_type = random.choices(user_types, weights=user_weights)[0]

        # Activity level based on user type
        activity_multiplier = {
            "casual": np.random.uniform(0.1, 0.3),
            "regular": np.random.uniform(0.3, 0.7),
            "power": np.random.uniform(0.7, 1.0),
        }[user_type]

        users.append(
            {
                "user_id": f"U{i:04d}",
                "user_type": user_type,
                "activity_level": activity_multiplier,
            }
        )

    return pd.DataFrame(users)


def generate_listening_sessions(
    users_df: pd.DataFrame, songs_df: pd.DataFrame, n_days: int = 30
) -> pd.DataFrame:
    """Generate realistic listening sessions with time patterns and session continuity."""

    sessions = []
    start_date = datetime.now() - timedelta(days=n_days)

    # Time-based listening patterns (hourly weights)
    hour_weights = [
        0.3,
        0.3,
        0.3,
        0.3,
        0.4,
        0.6,
        0.8,
        1.0,  # 0-7 (early morning to commute)
        0.9,
        0.5,
        0.4,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,  # 8-15 (work hours)
        0.9,
        1.0,
        1.0,
        1.0,
        0.9,
        0.8,
        0.6,
        0.4,  # 16-23 (evening to night)
    ]

    print("Generating listening sessions...")

    for _, user in tqdm(users_df.iterrows(), total=len(users_df)):
        # Number of listening sessions (not individual songs) based on user activity
        n_listening_sessions = int(
            n_days * user["activity_level"] * random.uniform(1, 5)
        )

        # User has genre preferences (simplified as preference for certain artists)
        n_preferred_artists = random.randint(3, 10)
        preferred_artists = (
            songs_df["artist_id"]
            .drop_duplicates()
            .sample(n=n_preferred_artists)
            .tolist()
        )

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
            prefer_known_artists_prob = 0.5 if is_weekend else 0.8

            # Session length (number of songs in this listening session)
            session_length = random.choices(
                [1, 3, 5, 10, 20], weights=[0.2, 0.3, 0.25, 0.15, 0.1]
            )[0]

            current_time = session_start
            last_artist_id = None

            for song_idx in range(session_length):
                # Songs in same session are more likely from same artist
                if song_idx > 0 and last_artist_id and random.random() < 0.4:
                    # Continue with same artist
                    same_artist_songs = songs_df[
                        songs_df["artist_id"] == last_artist_id
                    ]
                    if len(same_artist_songs) > 1:
                        song = same_artist_songs.sample(
                            n=1, weights=same_artist_songs["popularity"]
                        ).iloc[0]
                    else:
                        # Fall back to regular selection
                        if (
                            random.random() < prefer_known_artists_prob
                            and preferred_artists
                        ):
                            artist_songs = songs_df[
                                songs_df["artist_id"].isin(preferred_artists)
                            ]
                            if len(artist_songs) > 0:
                                song = artist_songs.sample(
                                    n=1, weights=artist_songs["popularity"]
                                ).iloc[0]
                            else:
                                song = songs_df.sample(
                                    n=1, weights=songs_df["popularity"]
                                ).iloc[0]
                        else:
                            song = songs_df.sample(
                                n=1, weights=songs_df["popularity"]
                            ).iloc[0]
                else:
                    # Regular song selection
                    if (
                        random.random() < prefer_known_artists_prob
                        and preferred_artists
                    ):
                        artist_songs = songs_df[
                            songs_df["artist_id"].isin(preferred_artists)
                        ]
                        if len(artist_songs) > 0:
                            song = artist_songs.sample(
                                n=1, weights=artist_songs["popularity"]
                            ).iloc[0]
                        else:
                            song = songs_df.sample(
                                n=1, weights=songs_df["popularity"]
                            ).iloc[0]
                    else:
                        song = songs_df.sample(
                            n=1, weights=songs_df["popularity"]
                        ).iloc[0]

                last_artist_id = song["artist_id"]

                # Listening duration: full song, skip, or partial
                listen_behavior = random.choices(
                    ["full", "skip", "partial"], weights=[0.4, 0.2, 0.4]
                )[0]

                if listen_behavior == "full":
                    ms_played = song["duration_ms"]
                elif listen_behavior == "skip":
                    ms_played = random.randint(3000, min(30000, int(song["duration_ms"])))  # 3-30 seconds or full duration
                else:  # partial
                    # For short songs, play at least 50%, for longer songs play 30-80%
                    min_play = int(song["duration_ms"] * 0.3) if song["duration_ms"] > 60000 else int(song["duration_ms"] * 0.5)
                    max_play = int(song["duration_ms"] * 0.8)
                    ms_played = random.randint(min_play, max_play)

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
    parser = argparse.ArgumentParser(
        description="Generate synthetic music listening data"
    )
    parser.add_argument("--users", type=int, default=1000, help="Number of users")
    parser.add_argument("--songs", type=int, default=5000, help="Number of songs")
    parser.add_argument("--artists", type=int, default=500, help="Number of artists")
    parser.add_argument("--days", type=int, default=30, help="Number of days of data")
    parser.add_argument(
        "--output",
        type=str,
        default="data/synthetic_sessions.csv",
        help="Output file path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(
        f"Generating synthetic data with {args.users} users, {args.songs} songs, {args.artists} artists"
    )

    # Generate data
    artists_df = generate_artists(args.artists)
    print(f"✓ Generated {len(artists_df)} artists")

    songs_df = generate_songs(args.songs, artists_df)
    print(f"✓ Generated {len(songs_df)} songs")

    users_df = generate_users(args.users)
    print(f"✓ Generated {len(users_df)} users")

    sessions_df = generate_listening_sessions(users_df, songs_df, args.days)
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

    # Print statistics
    print("\nDataset Statistics:")
    print(f"- Total sessions: {len(sessions_df):,}")
    print(f"- Avg sessions per user: {len(sessions_df) / len(users_df):.1f}")
    print(
        f"- Unique user-song pairs: {sessions_df.groupby(['user_id', 'track_id']).size().shape[0]:,}"
    )
    print(
        f"- Total listening time: {sessions_df['ms_played'].sum() / (1000 * 60 * 60):.1f} hours"
    )

    # Sample data preview
    print("\nSample sessions:")
    print(sessions_df.head(5).to_string())


if __name__ == "__main__":
    main()
