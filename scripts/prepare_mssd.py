"""
ETL pipeline to prepare music session data for graph construction.

Aggregates raw listening sessions into user-song edges with:
- play_count: how many times played
- total_ms: total milliseconds listened
- avg_completion_ratio: average listening completion
"""

import argparse
import pandas as pd
from pathlib import Path


def prepare_edge_data(sessions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sessions into user-song edges.

    Returns DataFrame with columns:
    - user_id, track_id, artist_id
    - play_count, total_ms_played, avg_completion_ratio
    """
    # Group by user-song pairs
    edge_data = (
        sessions_df.groupby(["user_id", "track_id", "artist_id"])
        .agg({"ms_played": ["count", "sum"], "track_duration_ms": "first"})
        .reset_index()
    )

    # Flatten column names
    edge_data.columns = [
        "user_id",
        "track_id",
        "artist_id",
        "play_count",
        "total_ms_played",
        "track_duration_ms",
    ]

    # Calculate average completion ratio
    edge_data["avg_ms_per_play"] = (
        edge_data["total_ms_played"] / edge_data["play_count"]
    )
    edge_data["avg_completion_ratio"] = (
        edge_data["avg_ms_per_play"] / edge_data["track_duration_ms"]
    )

    # Cap completion ratio at 1.0 (in case of rounding errors)
    edge_data["avg_completion_ratio"] = edge_data["avg_completion_ratio"].clip(
        upper=1.0
    )

    # Drop intermediate columns
    edge_data = edge_data.drop(columns=["avg_ms_per_play", "track_duration_ms"])

    return edge_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare music session data for graph construction"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/synthetic_sessions.csv",
        help="Input sessions CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/edge_list.parquet",
        help="Output edge list Parquet file",
    )

    args = parser.parse_args()

    print(f"Loading sessions from: {args.input}")
    sessions_df = pd.read_csv(args.input)
    print(f"Loaded {len(sessions_df):,} sessions")

    print("\nAggregating into edges...")
    edge_data = prepare_edge_data(sessions_df)
    print(f"Created {len(edge_data):,} unique user-song edges")

    # Show statistics
    print("\nEdge Statistics:")
    print(f"- Users: {edge_data['user_id'].nunique():,}")
    print(f"- Songs: {edge_data['track_id'].nunique():,}")
    print(f"- Artists: {edge_data['artist_id'].nunique():,}")
    print(f"- Avg plays per edge: {edge_data['play_count'].mean():.1f}")
    print(f"- Avg completion ratio: {edge_data['avg_completion_ratio'].mean():.1%}")

    # Save to parquet
    output_path = Path(args.output)
    output_path.parent.mkdir(exist_ok=True)
    edge_data.to_parquet(output_path, index=False)
    print(f"\nSaved edge data to: {args.output}")

    # Preview
    print("\nSample edges:")
    print(edge_data.head())


if __name__ == "__main__":
    main()
