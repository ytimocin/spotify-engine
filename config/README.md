# Configuration Files

This directory contains YAML configuration files for customizing the synthetic data generation process.

## Available Configurations

### default.yaml
The default configuration with balanced user behavior patterns and standard listening preferences.

### weekend_heavy.yaml
A configuration optimized for weekend-heavy listening patterns with:
- More power users (30% vs 15%)
- Longer listening sessions
- Lower skip rates
- Higher activity during weekend hours

## Creating Custom Configurations

You can create your own configuration files by copying and modifying `default.yaml`. Key sections include:

### User Types
```yaml
user_types:
  distribution:
    casual: 0.5    # 50% of users are casual listeners
    regular: 0.35  # 35% are regular listeners
    power: 0.15    # 15% are power users
```

### Session Behavior
```yaml
sessions:
  behavior_weights:
    full: 0.4      # 40% chance to listen to full song
    skip: 0.2      # 20% chance to skip early
    partial: 0.4   # 40% chance to listen partially
```

### Time Patterns
```yaml
time_patterns:
  hourly_weights:  # 24 values for each hour of the day
    - 0.3   # 00:00 - Low activity at midnight
    - 0.3   # 01:00
    # ... etc
```

## Usage

To use a custom configuration:

```bash
python scripts/generate_synthetic_data.py --config config/your_config.yaml
```

If no config is specified, the script will automatically use `config/default.yaml` if it exists.