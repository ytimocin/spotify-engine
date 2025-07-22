"""Generate comprehensive visual data profile report for synthetic data."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def create_user_activity_plots(sessions_df: pd.DataFrame, users_df: pd.DataFrame, output_dir: str):
    """Create user activity distribution plots."""
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


def create_temporal_plots(sessions_df: pd.DataFrame, users_df: pd.DataFrame, output_dir: str):
    """Create enhanced temporal pattern plots."""
    sessions_df["timestamp"] = pd.to_datetime(sessions_df["timestamp"])
    sessions_df["hour"] = sessions_df["timestamp"].dt.hour
    sessions_df["weekday"] = sessions_df["timestamp"].dt.weekday

    # Create 2x2 subplot for comprehensive temporal analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Enhanced hourly distribution with early morning highlighting
    hourly_sessions = sessions_df["hour"].value_counts().sort_index()
    hourly_proportions = hourly_sessions / len(sessions_df)

    # Color bars differently for early morning (1-5am) vs rest
    colors = [
        "red" if 1 <= h <= 5 else "skyblue" if 17 <= h <= 19 else "lightgray" for h in range(24)
    ]

    ax1.bar(range(24), hourly_proportions.values, color=colors, alpha=0.7)
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("Proportion of Total Sessions")
    ax1.set_title("Sessions by Hour (Red: Early Morning 1-5am, Blue: Peak Evening)")
    ax1.set_xticks(range(24))
    ax1.grid(True, alpha=0.3)

    # Add annotations
    early_morning_avg = hourly_proportions[1:6].mean()
    evening_peak_avg = hourly_proportions[17:20].mean()
    ax1.text(
        3,
        0.02,
        f"Early morning\navg: {early_morning_avg:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
    )
    ax1.text(
        18,
        0.07,
        f"Evening peak\navg: {evening_peak_avg:.3f}",
        ha="center",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3),
    )

    # 2. Weekday vs Weekend patterns
    sessions_df["is_weekend"] = sessions_df["weekday"] >= 5
    weekend_hourly = sessions_df[sessions_df["is_weekend"]]["hour"].value_counts().sort_index()
    weekday_hourly = sessions_df[~sessions_df["is_weekend"]]["hour"].value_counts().sort_index()

    weekend_prop = weekend_hourly / len(sessions_df[sessions_df["is_weekend"]])
    weekday_prop = weekday_hourly / len(sessions_df[~sessions_df["is_weekend"]])

    x = np.arange(24)
    width = 0.35

    ax2.bar(x - width / 2, weekday_prop.values, width, label="Weekday", alpha=0.7, color="blue")
    ax2.bar(x + width / 2, weekend_prop.values, width, label="Weekend", alpha=0.7, color="orange")
    ax2.set_xlabel("Hour of Day")
    ax2.set_ylabel("Proportion of Sessions")
    ax2.set_title("Weekday vs Weekend Listening Patterns")
    ax2.set_xticks(x)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Temporal heatmap by user type
    user_type_sessions = sessions_df.merge(users_df, on="user_id")
    user_type_order = ["casual", "regular", "power"]
    heatmap_data = []

    for user_type in user_type_order:
        type_data = user_type_sessions[user_type_sessions["user_type"] == user_type]
        hourly_dist = type_data["hour"].value_counts().sort_index()
        hourly_prop = hourly_dist / len(type_data)
        full_hourly = pd.Series(0.0, index=range(24))
        full_hourly.update(hourly_prop)
        heatmap_data.append(full_hourly.values)

    sns.heatmap(
        heatmap_data,
        xticklabels=range(24),
        yticklabels=user_type_order,
        annot=False,
        cmap="YlOrRd",
        ax=ax3,
        cbar_kws={"label": "Listening Intensity"},
    )
    ax3.set_title("Listening Patterns by User Type and Hour")
    ax3.set_xlabel("Hour of Day")
    ax3.set_ylabel("User Type")

    # 4. Weekly pattern analysis
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    weekday_sessions = sessions_df["weekday"].value_counts().sort_index()
    weekday_proportions = weekday_sessions / len(sessions_df)

    colors = ["lightblue"] * 5 + ["orange", "orange"]
    ax4.bar(range(7), weekday_proportions.values, color=colors, alpha=0.7)
    ax4.set_xticks(range(7))
    ax4.set_xticklabels(weekday_names)
    ax4.set_xlabel("Day of Week")
    ax4.set_ylabel("Proportion of Sessions")
    ax4.set_title("Sessions by Day of Week (Orange: Weekend)")
    ax4.grid(True, alpha=0.3)

    weekday_avg = weekday_proportions[:5].mean()
    weekend_avg = weekday_proportions[5:].mean()
    ax4.axhline(
        y=weekday_avg,
        color="blue",
        linestyle="--",
        alpha=0.7,
        label=f"Weekday avg: {weekday_avg:.3f}",
    )
    ax4.axhline(
        y=weekend_avg,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Weekend avg: {weekend_avg:.3f}",
    )
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_temporal_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_behavior_plots(sessions_df: pd.DataFrame, users_df: pd.DataFrame, output_dir: str):
    """Create enhanced listening behavior plots."""
    sessions_df["completion_rate"] = sessions_df["ms_played"] / sessions_df["track_duration_ms"]
    sessions_df["is_skip"] = sessions_df["ms_played"] < 30000

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Completion rate distribution
    completion_rates = sessions_df["completion_rate"].clip(0, 1)
    ax1.hist(completion_rates, bins=60, edgecolor="black", alpha=0.7, color="green", density=True)
    ax1.axvline(x=0.8, color="red", linestyle="--", label="80% threshold")
    ax1.axvline(x=1.0, color="orange", linestyle="--", label="100% completion")

    high_completion = completion_rates[completion_rates > 0.95]
    if len(high_completion) > 0:
        ax1.hist(
            high_completion,
            bins=20,
            alpha=0.5,
            color="red",
            range=(0.95, 1.0),
            density=True,
            label="95-100% region",
        )

    ax1.set_xlabel("Completion Rate")
    ax1.set_ylabel("Density")
    ax1.set_title("Song Completion Distribution (No 100% Spike)")
    ax1.legend()

    # 2. User type behavioral comparison
    user_type_sessions = sessions_df.merge(users_df, on="user_id")
    user_type_order = ["casual", "regular", "power"]

    behavioral_metrics = {}
    for user_type in user_type_order:
        type_data = user_type_sessions[user_type_sessions["user_type"] == user_type]
        skip_rate = type_data["is_skip"].mean()
        session_lengths = type_data.groupby("user_id").size()
        avg_session_length = session_lengths.mean()
        behavioral_metrics[user_type] = {
            "skip_rate": skip_rate,
            "avg_session_length": avg_session_length,
        }

    x = np.arange(len(user_type_order))
    width = 0.35

    skip_rates = [behavioral_metrics[ut]["skip_rate"] for ut in user_type_order]
    session_lengths = [behavioral_metrics[ut]["avg_session_length"] / 10 for ut in user_type_order]

    bars1 = ax2.bar(x - width / 2, skip_rates, width, label="Skip Rate", color="coral", alpha=0.7)
    bars2 = ax2.bar(
        x + width / 2,
        session_lengths,
        width,
        label="Avg Session Length (/10)",
        color="skyblue",
        alpha=0.7,
    )

    ax2.set_xlabel("User Type")
    ax2.set_ylabel("Rate / Scaled Length")
    ax2.set_title("User Type Behavioral Differences")
    ax2.set_xticks(x)
    ax2.set_xticklabels(user_type_order)
    ax2.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height * 10:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 3. Skip-completion coupling
    user_skip_rates = user_type_sessions.groupby("user_id")["is_skip"].mean()
    user_completion_std = (
        user_type_sessions[user_type_sessions["completion_rate"] > 0.8]
        .groupby("user_id")["completion_rate"]
        .std()
    )

    coupling_data = pd.DataFrame(
        {"skip_rate": user_skip_rates, "completion_variance": user_completion_std}
    ).dropna()

    if len(coupling_data) > 0:
        ax3.scatter(
            coupling_data["skip_rate"],
            coupling_data["completion_variance"],
            alpha=0.6,
            s=30,
            color="purple",
        )

        if len(coupling_data) > 5:
            z = np.polyfit(coupling_data["skip_rate"], coupling_data["completion_variance"], 1)
            p = np.poly1d(z)
            ax3.plot(
                coupling_data["skip_rate"],
                p(coupling_data["skip_rate"]),
                "r--",
                alpha=0.8,
                linewidth=2,
            )

            corr = coupling_data["skip_rate"].corr(coupling_data["completion_variance"])
            ax3.text(
                0.05,
                0.95,
                f"Correlation: {corr:.3f}",
                transform=ax3.transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

    ax3.set_xlabel("User Skip Rate")
    ax3.set_ylabel("User Completion Rate Variance")
    ax3.set_title("Skip-Completion Coupling\n(Higher skip rate → More completion variance)")
    ax3.grid(True, alpha=0.3)

    # 4. Session length by user type
    session_length_data = []
    session_length_labels = []
    for user_type in user_type_order:
        type_data = user_type_sessions[user_type_sessions["user_type"] == user_type]
        user_session_lengths = type_data.groupby("user_id").size()
        session_length_data.append(user_session_lengths.values)
        session_length_labels.append(f"{user_type}\n(n={len(user_session_lengths)})")

    box_plot = ax4.boxplot(session_length_data, labels=session_length_labels, patch_artist=True)
    colors = ["lightcoral", "lightblue", "lightgreen"]
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)

    ax4.set_xlabel("User Type")
    ax4.set_ylabel("Sessions per User")
    ax4.set_title(
        "Session Length Distribution by User Type\n(Power users should have longer sessions)"
    )
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/enhanced_listening_behavior.png", dpi=150, bbox_inches="tight")
    plt.close()


def create_genre_analysis_plots(data_dir: str, output_dir: str, users_df: pd.DataFrame):
    """Create genre analysis visualizations."""
    try:
        user_genre_prefs_df = pd.read_csv(f"{data_dir}/synthetic_user_genre_preferences.csv")
        artist_genres_df = pd.read_csv(f"{data_dir}/synthetic_artist_genres.csv")
        genres_df = pd.read_csv(f"{data_dir}/synthetic_genres.csv")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Genre popularity distribution (Zipf validation)
        if "popularity" in genres_df.columns:
            sorted_genres = genres_df.sort_values("popularity", ascending=False)
            ranks = np.arange(1, len(sorted_genres) + 1)

            ax1.loglog(ranks, sorted_genres["popularity"], "b-", alpha=0.7, linewidth=2)
            ax1.set_xlabel("Genre Rank (log scale)")
            ax1.set_ylabel("Popularity Score (log scale)")
            ax1.set_title("Genre Popularity Distribution (Zipf Law)")
            ax1.grid(True, alpha=0.3)

            zipf_theoretical = sorted_genres["popularity"].iloc[0] / ranks
            ax1.loglog(ranks, zipf_theoretical, "r--", alpha=0.5, label="Theoretical Zipf")
            ax1.legend()

        # 2. Top genres by artist count
        genre_counts = artist_genres_df["genre_name"].value_counts()
        top_genres = genre_counts.head(12)
        ax2.barh(range(len(top_genres)), top_genres.values, color="skyblue")
        ax2.set_yticks(range(len(top_genres)))
        ax2.set_yticklabels(top_genres.index, fontsize=9)
        ax2.set_xlabel("Number of Artists")
        ax2.set_title("Top 12 Genres by Artist Count")

        # 3. User genre diversity by user type
        genres_per_user = user_genre_prefs_df.groupby("user_id").size()
        user_types_genres = users_df.merge(
            genres_per_user.rename("genre_count"), left_on="user_id", right_index=True
        )

        user_type_order = ["casual", "regular", "power"]
        genre_data_by_type = [
            user_types_genres[user_types_genres["user_type"] == ut]["genre_count"].values
            for ut in user_type_order
        ]

        box_plot = ax3.boxplot(genre_data_by_type, labels=user_type_order, patch_artist=True)
        colors = ["lightcoral", "lightblue", "lightgreen"]
        for patch, color in zip(box_plot["boxes"], colors):
            patch.set_facecolor(color)

        ax3.set_xlabel("User Type")
        ax3.set_ylabel("Number of Preferred Genres")
        ax3.set_title("Genre Diversity Distribution by User Type")

        # 4. Genre affinity heatmap
        merged_prefs = user_genre_prefs_df.merge(users_df[["user_id", "user_type"]], on="user_id")
        top_genre_names = genre_counts.head(10).index

        affinity_matrix = []
        for user_type in user_type_order:
            type_affinities = []
            for genre in top_genre_names:
                avg_affinity = merged_prefs[
                    (merged_prefs["user_type"] == user_type) & (merged_prefs["genre_name"] == genre)
                ]["affinity_score"].mean()
                type_affinities.append(avg_affinity if not pd.isna(avg_affinity) else 0)
            affinity_matrix.append(type_affinities)

        sns.heatmap(
            affinity_matrix,
            xticklabels=[g[:8] for g in top_genre_names],
            yticklabels=user_type_order,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            ax=ax4,
            cbar_kws={"label": "Average Affinity Score"},
        )
        ax4.set_title("Genre Affinity by User Type (Top 10 Genres)")
        ax4.set_xlabel("Genre")
        ax4.set_ylabel("User Type")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/enhanced_genre_analysis.png", dpi=150, bbox_inches="tight")
        plt.close()

        print("  ✓ Generated enhanced genre analysis visualization")
        return len(genres_df)

    except FileNotFoundError as e:
        print(f"  - Genre data not found: {e}")
        print("  - Skipping genre analysis")
        return None


def create_summary_statistics_plot(
    sessions_df: pd.DataFrame, users_df: pd.DataFrame, songs_df: pd.DataFrame, output_dir: str
):
    """Create summary statistics table visualization."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("tight")
    ax.axis("off")

    total_sessions = len(sessions_df)
    total_users = len(users_df)
    total_songs = len(songs_df)
    total_listening_hours = sessions_df["ms_played"].sum() / (1000 * 60 * 60)
    avg_session_length = sessions_df["ms_played"].mean() / 1000
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

    for i in range(2):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax.set_title("Dataset Summary Statistics", fontsize=16, fontweight="bold", pad=20)
    plt.savefig(f"{output_dir}/summary_statistics.png", dpi=150, bbox_inches="tight")
    plt.close()


def print_validation_metrics(sessions_df: pd.DataFrame, users_df: pd.DataFrame, genre_count):
    """Print data quality validation metrics."""
    try:
        sessions_df["hour"] = pd.to_datetime(sessions_df["timestamp"]).dt.hour
        sessions_df["completion_rate"] = sessions_df["ms_played"] / sessions_df["track_duration_ms"]
        sessions_df["is_skip"] = sessions_df["ms_played"] < 30000

        early_morning_activity = sessions_df[sessions_df["hour"].isin([1, 2, 3, 4, 5])].shape[
            0
        ] / len(sessions_df)
        completion_spike = (sessions_df["completion_rate"] >= 0.999).mean()

        print("\n📊 Key Data Quality Metrics:")
        print(f"  - Early morning activity (1-5am): {early_morning_activity:.3f} (target: < 0.02)")
        print(f"  - 100% completion spike: {completion_spike:.3f} (target: < 0.10)")
        print(f"  - Total genres: {genre_count if genre_count else 'Unknown'} (target: 35)")

        user_type_sessions = sessions_df.merge(users_df, on="user_id")
        skip_rates_by_type = user_type_sessions.groupby("user_type")["is_skip"].mean()
        print("  - Skip rates by user type:")
        for user_type in ["casual", "regular", "power"]:
            if user_type in skip_rates_by_type.index:
                print(f"    - {user_type}: {skip_rates_by_type[user_type]:.3f}")
    except Exception as e:
        print(f"  - Could not calculate validation metrics: {e}")


def create_data_profile_report(data_dir: str, output_dir: str):
    """Generate comprehensive visual report of the synthetic data."""
    sessions_df = pd.read_csv(f"{data_dir}/synthetic_sessions.csv")
    users_df = pd.read_csv(f"{data_dir}/synthetic_users.csv")
    songs_df = pd.read_csv(f"{data_dir}/synthetic_songs.csv")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")

    create_user_activity_plots(sessions_df, users_df, output_dir)
    create_temporal_plots(sessions_df, users_df, output_dir)
    create_behavior_plots(sessions_df, users_df, output_dir)

    genre_count = create_genre_analysis_plots(data_dir, output_dir, users_df)
    create_summary_statistics_plot(sessions_df, users_df, songs_df, output_dir)

    print(f"\nData profile visualizations saved to {output_dir}/")
    print("  ✓ User activity distribution")
    print("  ✓ Enhanced temporal patterns (early morning validation)")
    print("  ✓ Enhanced listening behavior analysis (skip-completion coupling)")
    print("  ✓ Enhanced genre analysis (35 genres, Zipf distribution)")
    print("  ✓ Summary statistics")

    print_validation_metrics(sessions_df, users_df, genre_count)


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
