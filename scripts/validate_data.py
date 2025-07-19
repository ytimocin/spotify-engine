"""
Validate synthetic music listening data for quality and consistency.

This script checks:
- Required columns exist
- Data types are correct
- No invalid values (negative durations, ms_played > track_duration)
- Sufficient user-song interactions for training
"""

import pandas as pd
import sys
from typing import List, Tuple


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
    except Exception as e:
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
    invalid_durations = sessions_df[
        sessions_df["ms_played"] > sessions_df["track_duration_ms"]
    ]
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
        issues.append(
            f"{users_with_few_songs} users have fewer than 5 unique song interactions"
        )

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
    """Main validation function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate synthetic music data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic_sessions.csv",
        help="Path to sessions CSV file",
    )

    args = parser.parse_args()

    print(f"Validating data from: {args.input}")
    print("=" * 50)

    is_valid, issues = validate_synthetic_data(args.input)

    if is_valid:
        print("\n✅ Data validation passed!")
        print("The dataset is ready for training.")
    else:
        print("\n❌ Data validation failed!")
        print("\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()
