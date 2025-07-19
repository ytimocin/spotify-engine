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
- **Regular** (35%): 30-70% activity level
- **Power** (15%): 70-100% activity level

### 4. Listening Behaviors

- **Full plays** (40%): Complete song duration
- **Skips** (20%): 3-30 seconds
- **Partial** (40%): 30-80% of song duration

### 5. Preference Patterns

- Each user prefers 3-10 artists
- 70-80% of listens are preferred artists (weekday)
- 50% preferred artists on weekends (more exploration)

### 6. Popularity Distribution

- Artists follow power-law popularity (Pareto α=1.5)
- Songs inherit correlated popularity from artists
- Popular songs/artists get exponentially more plays