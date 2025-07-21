.PHONY: help setup install clean data train demo test lint format all

# Default target
help:
	@echo "Spotify Engine - Available commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup          - Create virtual environment"
	@echo "  make install        - Install core dependencies"
	@echo "  make dev-install    - Install development tools (formatters, linters)"
	@echo ""
	@echo "Main Pipeline:"
	@echo "  make data           - Generate synthetic data and build graph"
	@echo "  make train          - Train the GAT model (basic)"
	@echo "  make train-improved - Train with validation & early stopping (recommended)"
	@echo "  make test-model     - Test the trained model and show recommendations"
	@echo "  make compare-models - Compare all available models"
	@echo "  make demo           - Launch Jupyter notebook demo"
	@echo "  make all            - Run full pipeline (data + train)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format         - Auto-format code with black & isort"
	@echo "  make format-check   - Check formatting without changes"
	@echo "  make lint           - Run flake8 and pylint"
	@echo "  make type-check     - Run mypy type checking"
	@echo "  make quality        - Run all quality checks"
	@echo "  make fix            - Auto-fix code issues"
	@echo ""
	@echo "Other:"
	@echo "  make test           - Run tests (when implemented)"
	@echo "  make clean          - Remove generated files and cache"
	@echo "  make clean-all      - Remove everything including venv"

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

# Train model with SimpleTrainer
train:
	@echo "Training GAT model with SimpleTrainer..."
	python -m src.train --epochs 20 --output-dir models
	@echo "Training complete! Check models/simple/"

# Train with AdvancedTrainer (recommended)
train-improved:
	@echo "Training GAT model with AdvancedTrainer..."
	python -m src.train_improved --epochs 50 --patience 5 --use-scheduler --output-dir models
	@echo "Training complete! Check models/advanced/"

# Quick training (fewer epochs)
train-quick:
	python -m src.train --epochs 5

# Launch demo
demo:
	jupyter notebook notebooks/quick_demo.ipynb

# Test trained model
test-model:
	@echo "Testing trained model..."
	python -m src.test_model --num-users 3 --num-recs 5

# Compare models
compare-models:
	@echo "Comparing all trained models..."
	python -m src.test_model --compare

# Run unit tests
test:
	@echo "Running unit tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Install development dependencies
dev-install:
	pip install -r requirements-dev.txt
	@echo "Development tools installed!"

# Format code
format:
	@echo "Formatting with black..."
	black src/ scripts/ 
	@echo "Sorting imports with isort..."
	isort src/ scripts/
	@echo "Code formatting complete!"

# Check code formatting (without changing files)
format-check:
	@echo "Checking code format..."
	black src/ scripts/ --check --diff
	isort src/ scripts/ --check-only --diff

# Lint code
lint:
	@echo "Running flake8..."
	flake8 src/ scripts/
	@echo "Running pylint..."
	pylint src/ scripts/ || true
	@echo "Linting complete!"

# Type checking
type-check:
	@echo "Running mypy type checks..."
	mypy src/

# Run all quality checks
quality: format-check lint type-check
	@echo "All quality checks complete!"

# Auto-fix code issues
fix: format
	@echo "Running auto-fixes..."
	# Add any additional auto-fix commands here
	@echo "Auto-fix complete!"

# Pre-commit setup
pre-commit-install:
	pre-commit install
	@echo "Pre-commit hooks installed!"

pre-commit-run:
	pre-commit run --all-files

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

# Development shortcuts (removed - duplicate of dev-install above)

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