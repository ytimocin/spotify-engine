.PHONY: help setup install clean data train demo test lint format all

# Default target
help:
	@echo "Spotify Engine - Available commands:"
	@echo "  make setup      - Create virtual environment and install dependencies"
	@echo "  make install    - Install dependencies (requires activated venv)"
	@echo "  make data       - Generate synthetic data and build graph"
	@echo "  make train      - Train the GAT model"
	@echo "  make demo       - Launch Jupyter notebook demo"
	@echo "  make test       - Run tests (when implemented)"
	@echo "  make lint       - Run code linters"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Remove generated files and cache"
	@echo "  make all        - Run full pipeline (data + train)"

# Setup virtual environment
setup:
	python3 -m venv .venv
	@echo "Virtual environment created. Run:"
	@echo "  source .venv/bin/activate  # On macOS/Linux"
	@echo "  .venv\\Scripts\\activate     # On Windows"
	@echo "Then run: make install"

# Install dependencies
install:
	pip install --upgrade pip
	pip install torch torchvision torchaudio
	pip install -r requirements.txt
	pip install torch-geometric
	@echo "Dependencies installed successfully!"

# Generate data and build graph
data:
	@echo "Generating synthetic data..."
	python scripts/generate_synthetic_data.py
	@echo "Validating data..."
	python scripts/validate_data.py
	@echo "Preparing edge list..."
	python scripts/prepare_mssd.py
	@echo "Building graph..."
	python -m src.build_graph
	@echo "Data pipeline complete!"

# Train model
train:
	@echo "Training GAT model..."
	python -m src.train --epochs 20
	@echo "Training complete! Check models/model.ckpt"

# Quick training (fewer epochs)
train-quick:
	python -m src.train --epochs 5

# Launch demo
demo:
	jupyter notebook notebooks/quick_demo.ipynb

# Run tests (placeholder)
test:
	@echo "Tests not yet implemented"
	# pytest tests/

# Lint code
lint:
	@echo "Running flake8..."
	flake8 src/ scripts/ --max-line-length=100 --ignore=E501,W503 || true
	@echo "Running pylint..."
	pylint src/ scripts/ --disable=C0103,C0114,C0115,C0116,R0913 || true

# Format code
format:
	black src/ scripts/ --line-length=100
	isort src/ scripts/

# Clean generated files
clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache .coverage htmlcov/
	rm -rf *.egg-info dist/ build/
	rm -f data/*.csv data/*.parquet data/*.pt data/*.json
	rm -f models/*.ckpt models/*.pt models/*.json
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + || true
	@echo "Cleaned generated files and cache"

# Clean everything including venv
clean-all: clean
	rm -rf .venv/
	@echo "Removed virtual environment"

# Run full pipeline
all: data train
	@echo "Full pipeline complete!"

# Development shortcuts
dev-install:
	pip install black flake8 pylint isort pytest ipykernel

run: data train demo

# Check environment
check-env:
	@python --version
	@pip --version
	@echo "PyTorch: $$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "PyG: $$(python -c 'import torch_geometric; print(torch_geometric.__version__)' 2>/dev/null || echo 'Not installed')"

# Generate requirements
freeze:
	pip freeze > requirements-full.txt
	@echo "Full requirements saved to requirements-full.txt"