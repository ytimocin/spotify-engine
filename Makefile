.PHONY: help setup install clean synthetic-all kaggle-all lint format

# Default target
help:
	@echo "Spotify Engine - Music Recommendation System"
	@echo ""
	@echo "Setup:"
	@echo "  make setup          - Create virtual environment"
	@echo "  make install        - Install dependencies"
	@echo ""
	@echo "Pipelines:"
	@echo "  make synthetic-all  - Run synthetic data pipeline (session-based)"
	@echo "  make kaggle-all     - Run Kaggle data pipeline (playlist-based)"
	@echo "  make kaggle-fast    - Run FAST Kaggle training (10-100x speedup)"
	@echo "  make benchmark      - Compare training speeds"
	@echo ""
	@echo "Development:"
	@echo "  make lint           - Check code quality"
	@echo "  make format         - Auto-format code"
	@echo "  make clean          - Remove generated files"

# Setup virtual environment
setup:
	python3 -m venv .venv
	@echo "Run: source .venv/bin/activate"

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install torch torch-geometric
	@echo "✅ Dependencies installed"

# Synthetic pipeline (session-based recommendations)
synthetic-all:
	@echo "🎵 Running synthetic data pipeline..."
	python scripts/synthetic/generate_data.py
	python scripts/synthetic/prepare_edges.py
	python -m src.synthetic.build_graph
	python -m src.synthetic.train_improved --epochs 20
	python -m src.synthetic.test_model
	@echo "✅ Synthetic pipeline complete!"

# Kaggle pipeline (playlist-based recommendations)
kaggle-all:
	@echo "📋 Running Kaggle playlist pipeline..."
	python scripts/kaggle/prepare_data.py
	python scripts/kaggle/build_graph.py
	# Training modes (uncomment one):
	# Mini mode (~5 min): Quick testing, lower quality
	python -m src.kaggle.train --epochs 3 --max-playlists 500 --batch-size 256
	# Quick mode (~15 min): Demo quality
	# python -m src.kaggle.train --epochs 5 --max-playlists 1000 --batch-size 128
	# Balanced mode (~45 min): Better quality
	# python -m src.kaggle.train --epochs 8 --max-playlists 5000 --batch-size 96
	# Full mode (~3-4 hours): Best quality
	# python -m src.kaggle.train --epochs 20 --max-playlists 50000 --batch-size 64
	python scripts/kaggle/test_model.py
	@echo "✅ Kaggle pipeline complete!"

# Fast Kaggle training (using sparse operations)
kaggle-fast:
	@echo "⚡ Running FAST Kaggle training..."
	python scripts/kaggle/prepare_data.py
	python scripts/kaggle/build_graph.py
	# Fast training with sparse operations
	python -m src.kaggle.fast_train --epochs 10 --max-playlists 5000 --batch-size 512
	python scripts/kaggle/test_model.py --model models/kaggle_fast/best_model.pt
	@echo "✅ Fast Kaggle training complete!"

# Benchmark training speed
benchmark:
	@echo "⏱️  Running training speed benchmark..."
	python scripts/kaggle/benchmark_training.py --max-playlists 500 --epochs 3
	@echo "✅ Benchmark complete!"

# Code quality
lint:
	flake8 src/ scripts/

format:
	black src/ scripts/
	isort src/ scripts/

# Clean generated files
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf data/synthetic/*.csv data/synthetic/*.parquet data/synthetic/*.pt
	rm -rf data/kaggle/*.parquet data/kaggle/*.pt
	rm -rf models/synthetic/* models/kaggle/*
	@echo "🧹 Cleaned generated files"