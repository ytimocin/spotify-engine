# Synthetic Data Patterns

## Overview

The synthetic data generator creates realistic music listening patterns that mimic real-world behavior.

## Key Patterns

### 1. Time-Based Listening

- **Peak Hours**: 7-9 AM (commute), 5-7 PM (evening commute), 8-10 PM (relaxation)
- **Low Activity**: 2-5 AM (night), 10 AM-12 PM (work focus)
- **Weekend Patterns**: More exploratory listening, less routine

### 2. Session Continuity

Users listen in sessions of multiple songs:

- 20% single songs (quick checks)
- 30% short sessions (3 songs)
- 25% medium sessions (5 songs)
- 15% long sessions (10 songs)
- 10% extended sessions (20+ songs)

Within sessions:

- 40% chance of continuing with same artist
- Small gaps (3-10 seconds) between songs

### 3. User Types

- **Casual** (50%): 10-30% activity level
  - 1-3 preferred genres (focused preferences)
  - Higher skip rates (40-50%)
  - Shorter sessions (1-5 songs)
- **Regular** (35%): 30-70% activity level
  - 2-5 preferred genres (moderate diversity)
  - Moderate skip rates (25-35%)
  - Medium sessions (3-10 songs)
- **Power** (15%): 70-100% activity level
  - 3-8 preferred genres (exploratory)
  - Lower skip rates (15-25%)
  - Longer sessions (5-20 songs)

**Behavioral Multipliers**: Each user type has distinct behavioral patterns that affect session length, skip rates, and genre exploration. Power users listen 3-5x more than casual users.

### 4. Listening Behaviors

- **Full plays** (40%): Complete song duration
  - Uses beta distribution (α=8, β=2) for realistic variation
  - Eliminates unrealistic 100% completion spikes
  - Most "full" plays are 85-95% complete
- **Skips** (20%): 3-30 seconds
  - Higher for casual users, lower for power users
  - Coupled with completion variance (skip-prone users have more varied completions)
- **Partial** (40%): 30-80% of song duration
  - Beta distribution ensures smooth distribution
  - No artificial clustering at specific percentages

### 5. Preference Patterns

- Each user prefers 3-10 artists
- 70-80% of listens are preferred artists (weekday)
- 50% preferred artists on weekends (more exploration)

### 6. Popularity Distribution

- Artists follow power-law popularity (Pareto α=1.5)
- Songs inherit correlated popularity from artists
- Popular songs/artists get exponentially more plays

### 7. Genre Patterns

- **35 Genres**: Comprehensive music taxonomy
- **Zipf Distribution**: Top genres popular but not dominant
  - Top genre: <40% of total popularity
  - Long tail: Bottom 10 genres average <3% each
- **Artist-Genre Mapping**:
  - 50% of artists have 1 genre
  - 35% have 2 genres
  - 15% have 3 genres (typically popular artists)
- **User-Genre Preferences**:
  - Affinity scores (0-1) based on user type
  - Genre diversity increases with user activity level
  - Weekend listening shows more genre exploration

### 8. Enhanced Temporal Patterns

- **Reduced Early Morning Activity**: 1-5 AM activity <2% (realistic sleep patterns)
- **Peak Hours**: 
  - Morning commute: 7-9 AM
  - Evening peak: 5-7 PM (highest activity)
  - Late evening: 8-10 PM (relaxation)
- **Midnight Activity**: Higher than 3-5 AM but lower than evening
- **User Type Variations**:
  - Power users maintain activity throughout the day
  - Casual users concentrate in peak hours
  - Regular users show moderate variation
