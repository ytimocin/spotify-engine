# Getting Started

This guide will walk you through setting up and running the Spotify Engine recommendation system.

## Prerequisites

### System Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: Version 3.8 to 3.12 (3.12 recommended)
- **Memory**: At least 2GB RAM for training
- **Disk Space**: ~500MB for dependencies and data

### Check Your Python Version

```bash
python --version
# or
python3 --version
```

If you need to install Python, visit [python.org](https://www.python.org/downloads/).

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd spotify-engine
```

### 2. Create Virtual Environment

```bash
# Using Python 3.12 (recommended)
python3.12 -m venv .venv

# Or with any Python 3.8+
python -m venv .venv
```

### 3. Activate Virtual Environment

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```cmd
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### 4. Install PyTorch (CRITICAL - Do This First!)

PyTorch must be installed before other dependencies.

**macOS/Linux:**

```bash
pip install torch torchvision torchaudio
```

**Windows:**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**With CUDA (for GPU acceleration):**
Visit [pytorch.org](https://pytorch.org/get-started/locally/) for the correct command.

### 5. Install Other Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Graph neural network library
pip install torch-geometric
```

## Running the System

You have two options for running the recommendation system:

### Option A: Synthetic Data Pipeline (Session-based)

This pipeline generates synthetic listening sessions and predicts next-song recommendations.

```bash
# Run complete pipeline
make synthetic-all
```

This automatically:
1. Generates synthetic data (1000 users, 5000 songs)
2. Prepares session-based edges
3. Builds the graph
4. Trains the model
5. Tests recommendations

### Option B: Kaggle Data Pipeline (Playlist-based)

This pipeline uses real Spotify playlist data for playlist completion recommendations.

```bash
# First, download Kaggle data (see data/kaggle/README.md for URLs)
# Place CSV files in data/kaggle/

# Run complete pipeline
make kaggle-all
```

This automatically:
1. Processes Kaggle playlist data
2. Builds playlist-track-artist-genre graph
3. Trains playlist completion model
4. Tests playlist recommendations

**Training Modes** (edit Makefile to change):
- Mini mode (~5 min): Quick testing
- Quick mode (~15 min): Demo quality
- Balanced mode (~45 min): Better quality
- Full mode (~3-4 hours): Best quality

## Manual Steps (for both pipelines)

If you prefer to run steps individually:

### Synthetic Pipeline Steps

```bash
# 1. Generate synthetic data
python scripts/synthetic/generate_data.py

# 2. Prepare edges
python scripts/synthetic/prepare_edges.py

# 3. Build graph
python -m src.synthetic.build_graph

# 4. Train model
python -m src.synthetic.train_improved --epochs 50

# 5. Test model
python -m src.synthetic.test_model
```

### Kaggle Pipeline Steps

```bash
# 1. Prepare data
python scripts/kaggle/prepare_data.py

# 2. Build graph
python scripts/kaggle/build_graph.py

# 3. Train model
python -m src.kaggle.train --epochs 5 --max-playlists 1000

# 4. Test model
python scripts/kaggle/test_model.py
```

### View Interactive Demo

For synthetic data recommendations:

```bash
jupyter notebook notebooks/quick_demo.ipynb
```

## Verifying Your Setup

### Test Data Generation

```bash
python scripts/validate_data.py
```

Should show:

- ✓ Files exist
- ✓ Data statistics
- ✓ No errors

### Test Model Loading

```python
import torch

# For synthetic model
checkpoint = torch.load('models/synthetic/model_improved.ckpt')
print(f"Model trained for {len(checkpoint['metrics']['train_loss'])} epochs")
print(f"Final Recall@10: {checkpoint['metrics']['recall@10'][-1]:.2%}")

# For Kaggle model
model_state = torch.load('models/kaggle/best_model.pt')
print(f"Kaggle model loaded successfully")
```

## Common Issues

### ImportError: No module named 'torch_geometric'

**Solution**: Install PyTorch first, then torch-geometric:

```bash
pip uninstall torch-geometric
pip install torch torchvision torchaudio
pip install torch-geometric
```

### CUDA/GPU Errors

**Solution**: Use CPU-only PyTorch:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Out of Memory During Training

**Solution**: Reduce batch size or training data:

```bash
# Synthetic pipeline
python -m src.synthetic.train --batch-size 128

# Kaggle pipeline
python -m src.kaggle.train --batch-size 64 --max-playlists 1000
```

### Jupyter Notebook Not Found

**Solution**: Install Jupyter:

```bash
pip install notebook
# or
pip install jupyterlab
```

## Next Steps

Now that you have the system running:

1. **Try Both Pipelines**: Compare session-based vs playlist-based recommendations
2. **Experiment with Training**: Adjust epochs, batch sizes, and data sizes
3. **Explore the Models**: 
   - Synthetic: `src/synthetic/` and `src/common/models/`
   - Kaggle: `src/kaggle/models.py`
4. **Read the Docs**: 
   - [Training Guide](technical/training.md) - Training strategies
   - [Kaggle Pipeline](kaggle-pipeline.md) - Playlist recommendations
   - [Architecture](technical/architecture.md) - System design

## Getting Help

- [Technical Architecture](technical/architecture.md) - System design
- [Training Guide](technical/training.md) - Training strategies
- [Trainer Architecture](technical/trainers.md) - Custom trainers
- Open an issue if you encounter problems!
