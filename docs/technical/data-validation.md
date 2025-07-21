# Data Validation System

The spotify-engine includes a comprehensive data validation system that ensures synthetic data quality and realistic behavioral patterns.

## Overview

The validation system performs multi-level quality checks on synthetic music listening data:

1. **Basic Data Integrity**: Column existence, data types, value ranges
2. **Genre System Validation**: 35-genre system with Zipf distribution
3. **User Behavioral Patterns**: Realistic differences between user types
4. **Temporal Pattern Validation**: Realistic listening time patterns
5. **Completion Rate Distribution**: Beta distribution without unrealistic spikes

## Architecture

### Core Components

1. **scripts/validate_data.py** - Main validation script with comprehensive checks
2. **scripts/visualize_data_profile.py** - Visual validation and data profiling
3. **Makefile targets** - Convenient commands for validation workflows

## Validation Functions

### 1. Genre System Validation

Ensures the 35-genre system follows realistic patterns:

```python
def validate_genre_system(data_dir: str) -> Tuple[bool, List[str]]:
    """Validate the genre system with 35 genres and Zipf distribution."""
    
    # Load genre data files
    genres_df = pd.read_csv(f"{data_dir}/synthetic_genres.csv")
    artist_genres_df = pd.read_csv(f"{data_dir}/synthetic_artist_genres.csv")
    
    issues = []
    
    # Check genre count (should be exactly 35)
    if len(genres_df) != 35:
        issues.append(f"Expected 35 genres, found {len(genres_df)}")
    
    # Validate Zipf distribution of genre popularity
    if "popularity" in genres_df.columns:
        popularities = sorted(genres_df["popularity"].values, reverse=True)
        
        # Top genre should not dominate (< 40% share)
        top_genre_share = popularities[0]
        if top_genre_share > 0.4:
            issues.append(f"Top genre too popular: {top_genre_share:.2f}")
        
        # Long tail should exist (bottom 10 genres < 30% avg)
        bottom_10_avg = np.mean(popularities[-10:])
        if bottom_10_avg > 0.3:
            issues.append(f"Long tail genres too popular: {bottom_10_avg:.2f}")
    
    # Validate artist-genre assignments (1-3 genres per artist)
    avg_genres_per_artist = len(artist_genres_df) / artist_genres_df["artist_id"].nunique()
    if not (1.0 <= avg_genres_per_artist <= 3.0):
        issues.append(f"Invalid genres per artist: {avg_genres_per_artist:.2f}")
    
    return len(issues) == 0, issues
```

### 2. User Behavioral Differences Validation

Validates that different user types show expected behavioral patterns:

```python
def validate_user_behavioral_differences(
    sessions_df: pd.DataFrame, users_df: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """Validate that user types show expected behavioral differences."""
    
    # Merge sessions with user types
    merged_df = sessions_df.merge(users_df, on="user_id")
    
    # Calculate metrics by user type
    user_metrics = {}
    for user_type in ["casual", "regular", "power"]:
        type_data = merged_df[merged_df["user_type"] == user_type]
        
        # Skip rate (< 30 seconds listening)
        skip_rate = (type_data["ms_played"] < 30000).mean()
        
        # Average session length (songs per user)
        avg_session_length = type_data.groupby("user_id").size().mean()
        
        user_metrics[user_type] = {
            "skip_rate": skip_rate,
            "avg_session_length": avg_session_length
        }
    
    issues = []
    
    # Validate expected patterns:
    # Skip rates: casual > regular > power
    if not (user_metrics["casual"]["skip_rate"] > 
            user_metrics["regular"]["skip_rate"] > 
            user_metrics["power"]["skip_rate"]):
        issues.append("Skip rates don't follow expected pattern")
    
    # Session lengths: power > regular >= casual  
    if not (user_metrics["power"]["avg_session_length"] > 
            user_metrics["regular"]["avg_session_length"]):
        issues.append("Session lengths don't follow expected pattern")
    
    return len(issues) == 0, issues
```

### 3. Temporal Pattern Validation

Ensures realistic listening patterns throughout the day:

```python
def validate_temporal_patterns(sessions_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate temporal patterns with reduced early morning activity."""
    
    # Parse timestamps and extract hours
    sessions_df["timestamp"] = pd.to_datetime(sessions_df["timestamp"])
    sessions_df["hour"] = sessions_df["timestamp"].dt.hour
    
    # Calculate hourly activity proportions
    hourly_counts = sessions_df["hour"].value_counts().sort_index()
    hourly_proportions = hourly_counts / len(sessions_df)
    
    issues = []
    
    # Early morning activity (1-5am) should be very low (< 2%)
    early_morning_activity = hourly_proportions[1:6].mean()
    if early_morning_activity > 0.02:
        issues.append(f"Too much early morning activity: {early_morning_activity:.3f}")
    
    # Evening peak (5-7pm) should be significant (> 6%)
    evening_peak = hourly_proportions[17:20].mean()
    if evening_peak < 0.06:
        issues.append(f"Evening peak too low: {evening_peak:.3f}")
    
    # Midnight should be higher than deep early morning
    midnight_activity = hourly_proportions[0]
    if midnight_activity <= early_morning_activity:
        issues.append("Midnight activity should exceed early morning")
    
    return len(issues) == 0, issues
```

### 4. Completion Rate Distribution Validation

Validates realistic song completion patterns using beta distribution:

```python
def validate_completion_rate_distribution(sessions_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate completion rate distribution has no unrealistic spikes."""
    
    # Calculate completion rates
    sessions_df["completion_rate"] = sessions_df["ms_played"] / sessions_df["track_duration_ms"]
    
    issues = []
    
    # Check for unrealistic 100% completion spike (should be < 10%)
    exact_100_percent = (sessions_df["completion_rate"] >= 0.999).mean()
    if exact_100_percent > 0.1:
        issues.append(f"Too many exact 100% completions: {exact_100_percent:.3f}")
    
    # Validate realistic variation in "full" listens (>80% completion)
    full_listens = sessions_df[sessions_df["completion_rate"] > 0.8]["completion_rate"]
    if len(full_listens) > 0:
        completion_std = full_listens.std()
        if completion_std < 0.02:
            issues.append(f"Full completions too uniform (std: {completion_std:.4f})")
    
    # Validate skip-completion coupling
    skip_threshold = 30000
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
            issues.append(f"Skip-completion coupling weak (correlation: {correlation:.3f})")
    
    return len(issues) == 0, issues
```

## Usage

### Command Line Interface

```bash
# Run comprehensive validation on data directory
python scripts/validate_data.py --data-dir data

# Run legacy validation on single sessions file
python scripts/validate_data.py --legacy --input data/synthetic_sessions.csv

# Quick validation via Makefile
make validate
```

### Programmatic Usage

```python
from scripts.validate_data import validate_comprehensive_data

# Validate entire dataset
is_valid, issues = validate_comprehensive_data("data/")

if is_valid:
    print("✅ All validation checks passed!")
else:
    print("❌ Validation failed:")
    for issue in issues:
        print(f"  - {issue}")
```

## Validation Output

### Successful Validation
```
Running comprehensive validation on data directory: data
======================================================================

Dataset Statistics:
- Total sessions: 125,847
- Unique users: 1,000
- Unique tracks: 5,000
- Unique artists: 1,500
- Avg sessions per user: 125.8
- Avg unique tracks per user: 47.3
- Skip rate (<30s): 31.2%
- Full play rate: 23.8%

✅ All data validation checks passed!
The dataset meets quality standards and is ready for training.
```

### Failed Validation
```
❌ Data validation failed!

Issues found (4 total):
  - Genre: Expected 35 genres, found 32
  - Behavior: Skip rates don't follow expected pattern: casual > regular > power
  - Temporal: Too much early morning activity: 0.045 (should be < 0.02)
  - Completion: Too many exact 100% completions: 0.156 (should be < 0.10)

📋 Validation Summary:
  - Basic validation: ✅ PASS
  - Genre system: ❌ FAIL
  - User behavior: ❌ FAIL
  - Temporal patterns: ❌ FAIL
  - Completion rates: ❌ FAIL
```

## Data Profiling and Visualization

### Visual Validation Reports

The system generates comprehensive visual reports for data quality assessment:

```python
# Generate visual data profile
python scripts/visualize_data_profile.py --data-dir data --output-dir data/profile_report

# Quick profiling via Makefile
make profile
```

**Generated Visualizations:**

1. **User Activity Distribution**
   - Session count distribution across users
   - User type activity patterns

2. **Enhanced Temporal Patterns**
   - Hourly listening distribution with early morning highlighting
   - Weekday vs weekend patterns
   - User type temporal heatmaps

3. **Enhanced Listening Behavior**
   - Completion rate distribution (beta distribution validation)
   - User type behavioral differences
   - Skip-completion coupling analysis

4. **Enhanced Genre Analysis** 
   - Genre popularity Zipf distribution validation
   - Top genres by artist count
   - User genre diversity by user type
   - Genre affinity heatmaps

5. **Summary Statistics**
   - Dataset overview table
   - Key quality metrics

### Data Quality Metrics Output

```
📊 Key Data Quality Metrics:
  - Early morning activity (1-5am): 0.018 (target: < 0.02)
  - 100% completion spike: 0.087 (target: < 0.10)
  - Total genres: 35 (target: 35)
  - Skip rates by user type:
    - casual: 0.423
    - regular: 0.298
    - power: 0.187
```

## Integration with Training Pipeline

### Automated Validation

The validation system integrates with the training pipeline:

```bash
# Full pipeline with validation
make all-with-validation

# This runs:
# 1. make generate     # Generate synthetic data
# 2. make validate     # Validate data quality
# 3. make data         # Process data if validation passes
# 4. make train        # Train model on validated data
```

### Validation Triggers

```python
# Validation automatically triggered by:
# 1. Data generation completion
# 2. Before graph construction
# 3. Before model training (optional)

def build_graph_with_validation(data_dir: str, validate: bool = True):
    if validate:
        is_valid, issues = validate_comprehensive_data(data_dir)
        if not is_valid:
            raise ValueError(f"Data validation failed: {issues}")
    
    # Proceed with graph construction
    return build_graph(data_dir)
```

## Configuration

### Validation Thresholds

Validation thresholds can be customized:

```python
VALIDATION_CONFIG = {
    'genre_count': 35,
    'max_top_genre_share': 0.4,
    'max_long_tail_avg': 0.3,
    'min_genres_per_artist': 1.0,
    'max_genres_per_artist': 3.0,
    'max_early_morning_activity': 0.02,
    'min_evening_peak': 0.06,
    'max_exact_completion_rate': 0.1,
    'min_completion_std': 0.02,
    'min_skip_completion_correlation': 0.1
}
```

### Custom Validation Rules

```python
def add_custom_validation(sessions_df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Add domain-specific validation rules."""
    issues = []
    
    # Example: Validate average track duration
    avg_duration = sessions_df["track_duration_ms"].mean() / 1000  # seconds
    if not (120 <= avg_duration <= 300):  # 2-5 minutes
        issues.append(f"Unrealistic average track duration: {avg_duration:.1f}s")
    
    # Example: Validate user listening span
    user_spans = sessions_df.groupby("user_id")["timestamp"].apply(
        lambda x: (pd.to_datetime(x.max()) - pd.to_datetime(x.min())).days
    )
    if user_spans.mean() < 7:  # At least a week of data per user
        issues.append("Insufficient temporal span for users")
    
    return len(issues) == 0, issues
```

## Future Enhancements

### Planned Features

1. **Real-time Validation**: Continuous monitoring during data generation
2. **Statistical Testing**: Formal hypothesis tests for distribution validation
3. **Anomaly Detection**: ML-based detection of unusual patterns
4. **Validation Reporting**: Automated reports with recommendations

### Research Directions

1. **Domain-Aware Validation**: Music-specific quality metrics
2. **Cross-Dataset Validation**: Comparison with real music data patterns
3. **Temporal Validation**: Evolution of listening patterns over time
4. **Personalization Validation**: Individual user pattern validation

## Related Files

- `scripts/validate_data.py` - Main validation implementation
- `scripts/visualize_data_profile.py` - Visual validation and profiling
- `scripts/generate_synthetic_data.py` - Data generation with validation hooks
- `Makefile` - Validation workflow automation
- `src/build_graph.py` - Graph construction with validation integration