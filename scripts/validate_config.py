"""Validate configuration files for consistency and completeness."""

import sys
from typing import List

import yaml


def validate_config_file(config_path: str) -> List[str]:  # noqa: C901
    """Validate a configuration file against the DataGenerationConfig schema."""
    issues = []

    # Load config
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        return [f"Configuration file not found: {config_path}"]
    except yaml.YAMLError as e:
        return [f"Invalid YAML syntax: {e}"]

    # Check required sections
    required_sections = ["user_types", "sessions", "time_patterns", "content", "playback"]
    for section in required_sections:
        if section not in config_data:
            issues.append(f"Missing required section: {section}")

    # Validate user type distribution sums to 1.0
    if "user_types" in config_data and "distribution" in config_data["user_types"]:
        dist = config_data["user_types"]["distribution"]
        total = sum(dist.values())
        if abs(total - 1.0) > 0.001:
            issues.append(f"User type distribution sums to {total}, should be 1.0")

        # Check all user types have activity levels
        if "activity_levels" in config_data["user_types"]:
            activity_levels = config_data["user_types"]["activity_levels"]
            for user_type in dist.keys():
                if user_type not in activity_levels:
                    issues.append(f"Missing activity level for user type: {user_type}")
                elif (
                    not isinstance(activity_levels[user_type], list)
                    or len(activity_levels[user_type]) != 2
                ):
                    issues.append(f"Activity level for {user_type} should be a list of [min, max]")

    # Validate hourly weights has 24 values
    if "time_patterns" in config_data and "hourly_weights" in config_data["time_patterns"]:
        weights = config_data["time_patterns"]["hourly_weights"]
        if len(weights) != 24:
            issues.append(f"Hourly weights has {len(weights)} values, should be 24")

        # Check all weights are non-negative
        if any(w < 0 for w in weights):
            issues.append("Hourly weights should all be non-negative")

    # Validate session configuration
    if "sessions" in config_data:
        session_cfg = config_data["sessions"]

        # Validate session length weights sum to 1.0
        if "length_weights" in session_cfg:
            weights = session_cfg["length_weights"]
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                issues.append(f"Session length weights sum to {total}, should be 1.0")

        # Validate behavior weights sum to 1.0
        if "behavior_weights" in session_cfg:
            weights = session_cfg["behavior_weights"]
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                issues.append(f"Behavior weights sum to {total}, should be 1.0")

            # Check required behaviors
            required_behaviors = {"full", "skip", "partial"}
            missing_behaviors = required_behaviors - set(weights.keys())
            if missing_behaviors:
                issues.append(f"Missing behavior weights: {missing_behaviors}")

        # Validate probabilities are between 0 and 1
        probability_fields = [
            "same_artist_probability",
            "weekday_preference_probability",
            "weekend_preference_probability",
        ]
        for field in probability_fields:
            if field in session_cfg:
                value = session_cfg[field]
                if not (0 <= value <= 1):
                    issues.append(f"{field} should be between 0 and 1, got {value}")

    # Validate content configuration
    if "content" in config_data:
        content_cfg = config_data["content"]

        # Validate artist genre distribution
        if "artist_genre_distribution" in content_cfg:
            dist = content_cfg["artist_genre_distribution"]
            total = sum(dist.values())
            if abs(total - 1.0) > 0.001:
                issues.append(f"Artist genre distribution sums to {total}, should be 1.0")

        # Validate song duration
        if "song_duration" in content_cfg:
            duration = content_cfg["song_duration"]
            if "min_ms" not in duration or "max_ms" not in duration:
                issues.append("Song duration must have min_ms and max_ms")
            elif duration["min_ms"] >= duration["max_ms"]:
                issues.append("Song duration min_ms must be less than max_ms")
            elif duration["min_ms"] < 0:
                issues.append("Song duration values must be non-negative")

    # Validate playback configuration
    if "playback" in config_data:
        playback_cfg = config_data["playback"]

        if "skip_threshold_ms" in playback_cfg and playback_cfg["skip_threshold_ms"] < 0:
            issues.append("skip_threshold_ms must be non-negative")

        if "completion_threshold_ratio" in playback_cfg:
            ratio = playback_cfg["completion_threshold_ratio"]
            if not (0 <= ratio <= 1):
                issues.append(f"completion_threshold_ratio should be between 0 and 1, got {ratio}")

    return issues


def main():
    """Main function to validate configuration files."""
    if len(sys.argv) < 2:
        print("Usage: python validate_config.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    print(f"Validating configuration: {config_file}")

    issues = validate_config_file(config_file)

    if issues:
        print("\n❌ Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("\n✓ Configuration is valid")


if __name__ == "__main__":
    main()
