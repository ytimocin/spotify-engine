# Kaggle Flow - Playlist-Based Recommendations

## What is it?
The Kaggle flow helps **complete your playlists** by suggesting songs that fit well with what's already there. It's like having a friend who knows your playlist vibe and suggests "Hey, this song would fit perfectly!"

## The Big Picture
```
Your Playlist:
  ✓ Song A (energetic rock)
  ✓ Song B (energetic pop)  
  ✓ Song C (upbeat indie)
  ? _________________ ← Model suggests songs that fit!
```

## How Does it Work?

### 1. Prepare Real Spotify Data
```bash
python scripts/kaggle/prepare_data.py
```
- Loads actual Spotify playlists from Kaggle
- Extracts song features (energy, danceability, mood)
- Identifies artists and genres
- Cleans and organizes everything

### 2. Build a Graph
```bash
python scripts/kaggle/build_graph.py
```
Creates a network connecting:
- **Playlists** → Songs they contain
- **Songs** → Artists who made them
- **Songs/Artists** → Genres they belong to

Like mapping out musical relationships!

### 3. Train the AI Model
```bash
python -m src.kaggle.train
```
The model learns what makes playlists cohesive:
- "Workout playlists have high-energy songs"
- "Chill playlists share similar moods"
- "This playlist likes alternative rock"

**Training trick**: We hide the last 5 songs from each playlist and see if the model can predict them!

### 4. Get Playlist Recommendations
```bash
python scripts/kaggle/test_model.py --playlist 100 --explain
```
Ask: "What songs should I add to Playlist 100?"

The model returns:
```
Playlist 100: "Summer Vibes"
Current: 25 songs (avg energy: 0.75)

Recommended additions:
1. "Good Life" by OneRepublic (score: 0.92)
   - Matches playlist energy level
   - Similar genre (pop rock)
   - 3 songs in playlist have same vibe

2. "Pump It" by The Black Eyed Peas (score: 0.88)
   - High danceability like your playlist
   - You have 2 other songs by this artist
```

## Example Explanation
```
Why "Good Life" fits your playlist:
- Genre match: Your playlist is 60% pop rock
- Energy match: Song energy (0.73) ≈ playlist average (0.75)
- Similar tracks: Close to "Hey Soul Sister" already in playlist
- Artist style: Fits with your upbeat theme
```

## One-Line Summary
**"Smart playlist assistant: Suggests songs that match your playlist's vibe and mood."**

## Quick Run
```bash
# First: Download data from Kaggle (see data/kaggle/README.md)
make kaggle-all  # Does everything automatically!
```

## Key Difference from Synthetic
- **Synthetic**: "What song comes NEXT in time?"
- **Kaggle**: "What song BELONGS in this collection?"