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

### Step 1: Generate Synthetic Data
Since we're using synthetic data (to avoid licensing issues), first generate the dataset:

```bash
python scripts/generate_synthetic_data.py
```

This creates:
- `data/raw_sessions.csv`: Synthetic listening sessions
- `data/tracks.csv`: Song metadata
- 1000 users, 5000 songs, 500 artists
- ~100K listening sessions

### Step 2: Prepare the Data
Convert raw sessions into aggregated edge data:

```bash
python scripts/prepare_mssd.py
```

This creates:
- `data/edge_list.parquet`: Aggregated user-song interactions
- Calculates play counts and completion ratios

### Step 3: Build the Graph
Construct the heterogeneous graph structure:

```bash
python -m src.build_graph
```

This creates:
- `data/graph.pt`: PyTorch Geometric HeteroData object
- Contains user, song, and artist nodes
- Weighted edges based on listening behavior

### Step 4: Train the Model
Train the Graph Attention Network:

```bash
python -m src.train
```

Default training runs for 10 epochs. You can customize:
```bash
# More epochs
python -m src.train --epochs 20

# Different learning rate
python -m src.train --lr 0.001

# Smaller batches (if low memory)
python -m src.train --batch-size 256
```

This creates:
- `models/model.ckpt`: Trained model checkpoint
- `models/model.json`: Training metrics

### Step 5: View Recommendations
Launch the interactive demo:

```bash
jupyter notebook notebooks/quick_demo.ipynb
```

Then:
1. Open the notebook in your browser
2. Run all cells (Cell → Run All)
3. See personalized recommendations with explanations!

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
from src.models.gat_recommender import GATRecommender

# Load checkpoint
checkpoint = torch.load('models/model.ckpt')
print(f"Model trained for {len(checkpoint['metrics']['train_loss'])} epochs")
print(f"Final Recall@10: {checkpoint['metrics']['recall@10'][-1]:.2%}")
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
**Solution**: Reduce batch size:
```bash
python -m src.train --batch-size 128
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

1. **Experiment with Training**: Try different hyperparameters
2. **Explore the Code**: Check `src/models/gat_recommender.py`
3. **Understand the Data**: Look at `data/tracks.csv` 
4. **Modify the Model**: Try deeper architectures
5. **Add Features**: Incorporate genre, tempo, etc.

## Getting Help

- Check [Technical Architecture](technical/architecture.md) for system design
- See [Training Guide](technical/training.md) for model details
- Review [Future Enhancements](future/) for improvement ideas
- Open an issue if you encounter problems!