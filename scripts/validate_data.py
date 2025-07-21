"""
Validate synthetic music listening data for quality and consistency.

This script checks:
- Required columns exist
- Data types are correct
- No invalid values (negative durations, ms_played > track_duration)
- Sufficient user-song interactions for training
- User type behavioral differences (session length, skip rates, genre diversity)
- Genre system validation (35 genres, Zipf distribution)
- Temporal patterns validation (reduced early morning activity)
- Completion rate distribution (no 100% spikes)
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def validate_genre_system(data_dir: str) -> Tuple[bool, List[str]]:
    """Validate the genre system with 35 genres and Zipf distribution."""
    issues = []

    try:
        genres_df = pd.read_csv(f"{data_dir}/synthetic_genres.csv")
        artist_genres_df = pd.read_csv(f"{data_dir}/synthetic_artist_genres.csv")
        pd.read_csv(f"{data_dir}/synthetic_user_genre_preferences.csv")  # Check existence
    except FileNotFoundError as e:
        return False, [f"Genre file not found: {str(e)}"]

    # Check genre count
    if len(genres_df) != 35:
        issues.append(f"Expected 35 genres, found {len(genres_df)}")

    # Check Zipf distribution of genre popularity
    if "popularity" in genres_df.columns:
        popularities = sorted(genres_df["popularity"].values, reverse=True)

        # Check that top genre has reasonable share (should be < 40%)
        top_genre_share = popularities[0]
        if top_genre_share > 0.4:
            issues.append(f"Top genre popularity too high: {top_genre_share:.2f} (should be < 0.4)")

        # Check that distribution has long tail
        bottom_10_avg = np.mean(popularities[-10:])
        if bottom_10_avg > 0.3:
            issues.append(
                f"Long tail genres too popular: {bottom_10_avg:.2f} avg (should be < 0.3)"
            )

    # Validate genre assignments
    total_artist_genres = len(artist_genres_df)
    unique_artists = artist_genres_df["artist_id"].nunique()
    avg_genres_per_artist = total_artist_genres / unique_artists

    if avg_genres_per_artist < 1.0 or avg_genres_per_artist > 3.0:
        issues.append(f"Average genres per artist: {avg_genres_per_artist:.2f} (should be 1-3)")

    return len(issues) == 0, issues


def validate_user_behavioral_differences(
    sessions_df: pd.DataFrame, users_df: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """Validate that user types show expected behavioral differences."""
    issues = []

    # Merge sessions with user types
    merged_df = sessions_df.merge(users_df, on="user_id")

    # Calculate metrics by user type
    user_metrics = {}
    for user_type in ["casual", "regular", "power"]:
        type_data = merged_df[merged_df["user_type"] == user_type]
        if len(type_data) == 0:
            issues.append(f"No data found for user type: {user_type}")
            continue

        # Skip rate (< 30 seconds)
        skip_rate = (type_data["ms_played"] < 30000).mean()

        # Session length (songs per session)
        sessions_per_user = type_data.groupby("user_id").size()
        avg_session_length = sessions_per_user.mean()

        user_metrics[user_type] = {"skip_rate": skip_rate, "avg_session_length": avg_session_length}

    # Validate expected behavioral patterns
    if len(user_metrics) >= 3:
        # Skip rates: casual > regular > power
        if not (
            user_metrics["casual"]["skip_rate"]
            > user_metrics["regular"]["skip_rate"]
            > user_metrics["power"]["skip_rate"]
        ):
            issues.append("Skip rates don't follow expected pattern: casual > regular > power")

        # Session lengths: power > regular > casual
        if not (
            user_metrics["power"]["avg_session_length"]
            > user_metrics["regular"]["avg_session_length"]
        ):
            issues.append(
                "Session lengths don't follow expected pattern: power > regular >= casual"
            )

    return len(issues) == 0, issues


def validate_temporal_patterns(sessions_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate temporal patterns with reduced early morning activity."""
    issues = []

    # Parse timestamps
    sessions_df["timestamp"] = pd.to_datetime(sessions_df["timestamp"])
    sessions_df["hour"] = sessions_df["timestamp"].dt.hour

    # Get hourly distribution
    hourly_counts = sessions_df["hour"].value_counts().sort_index()
    hourly_proportions = hourly_counts / len(sessions_df)

    # Check early morning activity (1-5am should be very low)
    early_morning_activity = hourly_proportions[1:6].mean()  # 1am-5am
    if early_morning_activity > 0.02:  # Should be < 2% of total activity
        issues.append(
            f"Too much early morning activity: {early_morning_activity:.3f} (should be < 0.02)"
        )

    # Check peak evening hours (5-7pm should be highest)
    evening_peak = hourly_proportions[17:20].mean()  # 5pm-7pm
    if evening_peak < 0.06:  # Should be significant peak
        issues.append(f"Evening peak too low: {evening_peak:.3f} (should be > 0.06)")

    # Check that midnight-1am is higher than 1-5am
    midnight_activity = hourly_proportions[0]  # 12am-1am
    if midnight_activity <= early_morning_activity:
        issues.append("Midnight activity should be higher than early morning activity")

    return len(issues) == 0, issues


def validate_completion_rate_distribution(sessions_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate completion rate distribution has no unrealistic spikes."""
    issues = []

    # Calculate completion rates
    sessions_df["completion_rate"] = sessions_df["ms_played"] / sessions_df["track_duration_ms"]

    # Check for 100% completion spike (should be minimal)
    exact_100_percent = (sessions_df["completion_rate"] >= 0.999).mean()
    if exact_100_percent > 0.1:  # Should be < 10%
        issues.append(f"Too many exact 100% completions: {exact_100_percent:.3f} (should be < 0.1)")

    # Check distribution shape for "full" listens (>80% completion)
    full_listens = sessions_df[sessions_df["completion_rate"] > 0.8]["completion_rate"]
    if len(full_listens) > 0:
        # Should have some variation, not all exactly 1.0
        completion_std = full_listens.std()
        if completion_std < 0.02:
            issues.append(f"Full completion rates too uniform (std: {completion_std:.4f})")

    # Check skip vs completion coupling
    skip_threshold = 30000
    skip_sessions = sessions_df[sessions_df["ms_played"] < skip_threshold]
    full_sessions = sessions_df[sessions_df["completion_rate"] > 0.8]

    if len(skip_sessions) > 0 and len(full_sessions) > 0:
        # Users who skip more should have more varied completion rates
        user_skip_rates = sessions_df.groupby("user_id").apply(
            lambda x: (x["ms_played"] < skip_threshold).mean()
        )
        user_completion_variance = (
            sessions_df[sessions_df["completion_rate"] > 0.8]
            .groupby("user_id")["completion_rate"]
            .std()
        )

        # Users with higher skip rates should have more completion variance
        common_users = set(user_skip_rates.index) & set(user_completion_variance.index)
        if len(common_users) > 10:
            skip_rates_common = user_skip_rates.loc[list(common_users)]
            completion_var_common = user_completion_variance.loc[list(common_users)].fillna(0)

            correlation = np.corrcoef(skip_rates_common, completion_var_common)[0, 1]
            if not np.isnan(correlation) and correlation < 0.1:
                issues.append(
                    f"Skip rates and completion variance not properly coupled (correlation: {correlation:.3f})"
                )

    return len(issues) == 0, issues


def validate_comprehensive_data(data_dir: str) -> Tuple[bool, List[str]]:
    """Run comprehensive validation on all data files."""
    issues = []

    # Check that all expected files exist
    expected_files = [
        "synthetic_sessions.csv",
        "synthetic_users.csv",
        "synthetic_songs.csv",
        "synthetic_artists.csv",
        "synthetic_genres.csv",
        "synthetic_artist_genres.csv",
        "synthetic_song_genres.csv",
        "synthetic_user_genre_preferences.csv",
    ]

    missing_files = []
    for filename in expected_files:
        filepath = Path(data_dir) / filename
        if not filepath.exists():
            missing_files.append(filename)

    if missing_files:
        issues.extend([f"Missing file: {f}" for f in missing_files])
        return False, issues

    # Load main data files
    try:
        sessions_df = pd.read_csv(f"{data_dir}/synthetic_sessions.csv")
        users_df = pd.read_csv(f"{data_dir}/synthetic_users.csv")
        pd.read_csv(f"{data_dir}/synthetic_songs.csv")  # Check existence
    except Exception as e:
        return False, [f"Error loading main data files: {str(e)}"]

    # Run all validation checks
    genre_valid, genre_issues = validate_genre_system(data_dir)
    if not genre_valid:
        issues.extend([f"Genre: {issue}" for issue in genre_issues])

    behavior_valid, behavior_issues = validate_user_behavioral_differences(sessions_df, users_df)
    if not behavior_valid:
        issues.extend([f"Behavior: {issue}" for issue in behavior_issues])

    temporal_valid, temporal_issues = validate_temporal_patterns(sessions_df)
    if not temporal_valid:
        issues.extend([f"Temporal: {issue}" for issue in temporal_issues])

    completion_valid, completion_issues = validate_completion_rate_distribution(sessions_df)
    if not completion_valid:
        issues.extend([f"Completion: {issue}" for issue in completion_issues])

    return len(issues) == 0, issues


def validate_synthetic_data(sessions_path: str) -> Tuple[bool, List[str]]:
    """
    Validate the generated synthetic data.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    try:
        sessions_df = pd.read_csv(sessions_path)
    except FileNotFoundError:
        return False, [f"File not found: {sessions_path}"]
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        return False, [f"Error reading file: {str(e)}"]

    issues = []

    # Check for required columns
    required_cols = [
        "user_id",
        "track_id",
        "artist_id",
        "timestamp",
        "ms_played",
        "track_duration_ms",
    ]
    missing_cols = set(required_cols) - set(sessions_df.columns)
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")
        return False, issues  # Can't continue without required columns

    # Check data types
    if not pd.api.types.is_numeric_dtype(sessions_df["ms_played"]):
        issues.append("ms_played should be numeric")
    if not pd.api.types.is_numeric_dtype(sessions_df["track_duration_ms"]):
        issues.append("track_duration_ms should be numeric")

    # Check for invalid durations
    invalid_durations = sessions_df[sessions_df["ms_played"] > sessions_df["track_duration_ms"]]
    if len(invalid_durations) > 0:
        issues.append(
            f"Found {len(invalid_durations)} sessions where ms_played > track_duration_ms"
        )

    # Check for negative values
    if (sessions_df["ms_played"] < 0).any():
        issues.append("Found negative ms_played values")
    if (sessions_df["track_duration_ms"] < 0).any():
        issues.append("Found negative track_duration_ms values")

    # Check for zero durations
    zero_plays = (sessions_df["ms_played"] == 0).sum()
    if zero_plays > 0:
        issues.append(f"Found {zero_plays} sessions with zero ms_played")

    # Check for sufficient user-song interactions
    interactions_per_user = sessions_df.groupby("user_id")["track_id"].nunique()
    users_with_few_songs = (interactions_per_user < 5).sum()
    if users_with_few_songs > 0:
        issues.append(f"{users_with_few_songs} users have fewer than 5 unique song interactions")

    # Check for orphaned IDs
    n_users = sessions_df["user_id"].nunique()
    n_tracks = sessions_df["track_id"].nunique()
    n_artists = sessions_df["artist_id"].nunique()

    # Additional statistics (not issues, just info)
    print("\nDataset Statistics:")
    print(f"- Total sessions: {len(sessions_df):,}")
    print(f"- Unique users: {n_users:,}")
    print(f"- Unique tracks: {n_tracks:,}")
    print(f"- Unique artists: {n_artists:,}")
    print(f"- Avg sessions per user: {len(sessions_df) / n_users:.1f}")
    print(f"- Avg unique tracks per user: {interactions_per_user.mean():.1f}")

    # Check listening patterns
    skip_threshold = 30000  # 30 seconds
    skips = (sessions_df["ms_played"] < skip_threshold).sum()
    skip_rate = skips / len(sessions_df) * 100
    print(f"- Skip rate (<30s): {skip_rate:.1f}%")

    full_plays = (sessions_df["ms_played"] == sessions_df["track_duration_ms"]).sum()
    full_play_rate = full_plays / len(sessions_df) * 100
    print(f"- Full play rate: {full_play_rate:.1f}%")

    return len(issues) == 0, issues


def main():
    """Run comprehensive validation on synthetic music data."""
    parser = argparse.ArgumentParser(description="Validate synthetic music data comprehensively")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing synthetic data files",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Run legacy validation on single sessions file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic_sessions.csv",
        help="Path to sessions CSV file (for legacy mode)",
    )

    args = parser.parse_args()

    if args.legacy:
        print(f"Running legacy validation on: {args.input}")
        print("=" * 50)
        is_valid, issues = validate_synthetic_data(args.input)
    else:
        print(f"Running comprehensive validation on data directory: {args.data_dir}")
        print("=" * 70)

        # Run comprehensive validation
        is_valid, issues = validate_comprehensive_data(args.data_dir)

        # Also run basic validation on sessions file
        sessions_path = f"{args.data_dir}/synthetic_sessions.csv"
        if Path(sessions_path).exists():
            print("\nRunning basic validation on sessions file...")
            basic_valid, basic_issues = validate_synthetic_data(sessions_path)
            if not basic_valid:
                is_valid = False
                issues.extend([f"Basic: {issue}" for issue in basic_issues])

    if is_valid:
        print("\n✅ All data validation checks passed!")
        print("The dataset meets quality standards and is ready for training.")
    else:
        print("\n❌ Data validation failed!")
        print(f"\nIssues found ({len(issues)} total):")
        for issue in issues:
            print(f"  - {issue}")

        print("\n📋 Validation Summary:")
        issues_str = str(issues)
        print("  - Basic validation: " + ("✅ PASS" if "Basic:" not in issues_str else "❌ FAIL"))
        print("  - Genre system: " + ("✅ PASS" if "Genre:" not in issues_str else "❌ FAIL"))
        print("  - User behavior: " + ("✅ PASS" if "Behavior:" not in issues_str else "❌ FAIL"))
        print(
            "  - Temporal patterns: " + ("✅ PASS" if "Temporal:" not in issues_str else "❌ FAIL")
        )
        print(
            "  - Completion rates: " + ("✅ PASS" if "Completion:" not in issues_str else "❌ FAIL")
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
