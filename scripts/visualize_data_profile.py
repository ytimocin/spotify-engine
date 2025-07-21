"""Generate visual data profile report for synthetic data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def create_data_profile_report(data_dir: str, output_dir: str):
    """Generate comprehensive visual report of the synthetic data."""
    # Load data
    sessions_df = pd.read_csv(f"{data_dir}/synthetic_sessions.csv")
    users_df = pd.read_csv(f"{data_dir}/synthetic_users.csv")
    songs_df = pd.read_csv(f"{data_dir}/synthetic_songs.csv")

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    # 1. User activity distribution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    user_sessions = sessions_df.groupby("user_id").size()
    ax1.hist(user_sessions, bins=50, edgecolor="black", alpha=0.7)
    ax1.set_xlabel("Number of Sessions")
    ax1.set_ylabel("Number of Users")
    ax1.set_title("User Activity Distribution")

    # User type distribution
    user_type_sessions = sessions_df.merge(users_df, on="user_id")
    user_type_counts = user_type_sessions.groupby("user_type").size()
    ax2.bar(user_type_counts.index, user_type_counts.values)
    ax2.set_xlabel("User Type")
    ax2.set_ylabel("Number of Sessions")
    ax2.set_title("Sessions by User Type")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/user_activity.png", dpi=150)
    plt.close()

    # 2. Temporal patterns
    sessions_df["timestamp"] = pd.to_datetime(sessions_df["timestamp"])
    sessions_df["hour"] = sessions_df["timestamp"].dt.hour
    sessions_df["weekday"] = sessions_df["timestamp"].dt.weekday

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Hourly distribution
    hourly_sessions = sessions_df["hour"].value_counts().sort_index()
    ax1.bar(hourly_sessions.index, hourly_sessions.values, color="skyblue")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Number of Sessions")
    ax1.set_title("Sessions by Hour of Day")
    ax1.set_xticks(range(24))

    # Weekday distribution
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_sessions = sessions_df["weekday"].value_counts().sort_index()
    ax2.bar(range(7), weekday_sessions.values)
    ax2.set_xticks(range(7))
    ax2.set_xticklabels(weekday_names)
    ax2.set_xlabel("Day of Week")
    ax2.set_ylabel("Number of Sessions")
    ax2.set_title("Sessions by Day of Week")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/temporal_patterns.png", dpi=150)
    plt.close()

    # 3. Listening behavior
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Skip rate distribution
    sessions_df["completion_rate"] = sessions_df["ms_played"] / sessions_df["track_duration_ms"]
    sessions_df["is_skip"] = sessions_df["ms_played"] < 30000

    completion_rates = sessions_df["completion_rate"].clip(0, 1)
    ax1.hist(completion_rates, bins=50, edgecolor="black", alpha=0.7, color="green")
    ax1.axvline(x=0.8, color="red", linestyle="--", label="80% threshold")
    ax1.set_xlabel("Completion Rate")
    ax1.set_ylabel("Number of Sessions")
    ax1.set_title("Song Completion Distribution")
    ax1.legend()

    # Skip rate by user type
    skip_by_type = user_type_sessions.groupby("user_type")["is_skip"].mean()
    ax2.bar(skip_by_type.index, skip_by_type.values, color="coral")
    ax2.set_xlabel("User Type")
    ax2.set_ylabel("Skip Rate")
    ax2.set_title("Skip Rate by User Type")
    ax2.set_ylim(0, 0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/listening_behavior.png", dpi=150)
    plt.close()

    # 4. Genre analysis (if available)
    try:
        user_genre_prefs_df = pd.read_csv(f"{data_dir}/synthetic_user_genre_preferences.csv")
        artist_genres_df = pd.read_csv(f"{data_dir}/synthetic_artist_genres.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Genre popularity distribution
        genre_counts = artist_genres_df["genre_name"].value_counts()
        ax1.barh(genre_counts.index[:10], genre_counts.values[:10])
        ax1.set_xlabel("Number of Artists")
        ax1.set_ylabel("Genre")
        ax1.set_title("Top 10 Genres by Artist Count")

        # User genre diversity
        genres_per_user = user_genre_prefs_df.groupby("user_id").size()
        user_types_genres = users_df.merge(
            genres_per_user.rename("genre_count"), left_on="user_id", right_index=True
        )
        genre_diversity = user_types_genres.groupby("user_type")["genre_count"].mean()

        ax2.bar(genre_diversity.index, genre_diversity.values, color="purple")
        ax2.set_xlabel("User Type")
        ax2.set_ylabel("Average Number of Genres")
        ax2.set_title("Genre Diversity by User Type")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/genre_analysis.png", dpi=150)
        plt.close()

        print("  ✓ Generated genre analysis visualization")
    except FileNotFoundError:
        print("  - Genre data not found, skipping genre analysis")

    # 5. Summary statistics
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    # Calculate summary stats
    total_sessions = len(sessions_df)
    total_users = len(users_df)
    total_songs = len(songs_df)
    total_listening_hours = sessions_df["ms_played"].sum() / (1000 * 60 * 60)
    avg_session_length = sessions_df["ms_played"].mean() / 1000  # in seconds
    skip_rate = (sessions_df["ms_played"] < 30000).mean()
    completion_rate = (sessions_df["ms_played"] >= sessions_df["track_duration_ms"] * 0.8).mean()

    summary_data = [
        ["Total Sessions", f"{total_sessions:,}"],
        ["Total Users", f"{total_users:,}"],
        ["Total Songs", f"{total_songs:,}"],
        ["Total Listening Time", f"{total_listening_hours:.1f} hours"],
        ["Average Session Length", f"{avg_session_length:.1f} seconds"],
        ["Overall Skip Rate", f"{skip_rate:.1%}"],
        ["Overall Completion Rate", f"{completion_rate:.1%}"],
    ]

    table = ax.table(
        cellText=summary_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.6, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)

    # Style the header
    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title("Dataset Summary Statistics", fontsize=16, fontweight="bold", pad=20)

    plt.savefig(f"{output_dir}/summary_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nData profile visualizations saved to {output_dir}/")
    print("  ✓ User activity distribution")
    print("  ✓ Temporal patterns")
    print("  ✓ Listening behavior analysis")
    print("  ✓ Summary statistics")


def main():
    """Main function to run data profiling."""
    parser = argparse.ArgumentParser(description="Generate data profile visualizations")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing the synthetic data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/profile_report",
        help="Directory to save visualization outputs",
    )

    args = parser.parse_args()

    print(f"Loading data from: {args.data_dir}")
    print(f"Saving visualizations to: {args.output_dir}")

    create_data_profile_report(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
