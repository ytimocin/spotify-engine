# Synthetic Flow - Session-Based Recommendations

## What is it?
The synthetic flow predicts **what song you'll listen to next** based on your listening history. It's like Spotify's "autoplay" feature that keeps the music going after your playlist ends.

## The Big Picture
```
Your listening session: Song A → Song B → Song C → ???
                                                    ↑
                                          Model predicts this!
```

## How Does it Work?

### 1. Generate Fake Music Data
```bash
python scripts/synthetic/generate_data.py
```
- Creates fake users (casual, regular, power users)
- Generates fake listening sessions
- Adds 35 music genres
- Simulates realistic behavior (people skip songs they don't like!)

### 2. Build a Graph
```bash
python scripts/synthetic/prepare_edges.py
python -m src.synthetic.build_graph
```
Creates a network connecting:
- **Users** → Songs they listened to
- **Songs** → Artists who made them
- **Songs/Artists** → Genres they belong to

Think of it like a social network, but for music!

### 3. Train the AI Model
```bash
python -m src.synthetic.train_improved
```
The model learns patterns like:
- "Users who like Song A often listen to Song B next"
- "Rock fans tend to listen to more rock"
- "Morning listeners prefer calm music"

### 4. Get Recommendations
```bash
python -m src.synthetic.test_model --user 0
```
Ask: "What should User 0 listen to next?"

The model returns:
```
Top 5 Recommendations:
1. Song 42 (score: 0.89) - Similar genre & artist
2. Song 17 (score: 0.85) - You liked similar songs
3. Song 99 (score: 0.82) - Popular with similar users
...
```

## Example Explanation
```
Why was Song 42 recommended to User 0?
- Genre match: You love "indie rock" (genre similarity: 0.75)
- Artist connection: You've played 5 songs by this artist
- Similar users: 89% of users like you enjoyed this song
```

## One-Line Summary
**"Netflix for music: Predicts your next song based on what you and similar users listened to before."**

## Quick Run
```bash
make synthetic-all  # Does everything automatically!
```