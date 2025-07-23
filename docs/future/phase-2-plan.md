# Phase 2 Implementation Plan: Model Management & Advanced Training

This document provides a detailed implementation roadmap for Phase 2 of the spotify-engine project, focusing on model versioning, experiment tracking, and advanced training features.

## Overview

**Goal**: Transform the system from a research prototype to a production-ready recommendation engine with systematic experiment management and advanced training capabilities.

**Timeline**: 4-6 weeks  
**Priority**: High (Next phase after Phase 1 completion)

## Phase 2 Features

### 2.1 Model Versioning & Experiment Tracking

#### 2.1.1 Experiment Management System

**Implementation Steps:**

1. **Create Experiment Framework**

   ```python
   # src/experiment_manager.py
   class ExperimentManager:
       def __init__(self, base_dir="experiments"):
           self.base_dir = Path(base_dir)
           self.current_experiment = None

       def start_experiment(self, name: str, config: dict):
           """Start new experiment with timestamp and configuration."""
           timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
           exp_id = f"{name}_{timestamp}"

           exp_dir = self.base_dir / exp_id
           exp_dir.mkdir(parents=True, exist_ok=True)

           # Save configuration
           with open(exp_dir / "config.yaml", "w") as f:
               yaml.dump(config, f)

           self.current_experiment = {
               'id': exp_id,
               'dir': exp_dir,
               'config': config,
               'start_time': datetime.now()
           }

           return exp_id
   ```

2. **Model Versioning**

   ```python
   class ModelVersion:
       def __init__(self, experiment_id: str, model_path: str, metrics: dict):
           self.experiment_id = experiment_id
           self.model_path = model_path
           self.metrics = metrics
           self.created_at = datetime.now()

       def save_metadata(self):
           """Save version metadata for comparison."""
           metadata = {
               'experiment_id': self.experiment_id,
               'model_path': str(self.model_path),
               'metrics': self.metrics,
               'created_at': self.created_at.isoformat(),
               'git_hash': self._get_git_hash(),
               'dependencies': self._get_dependencies()
           }

           with open(self.model_path.parent / "metadata.json", "w") as f:
               json.dump(metadata, f, indent=2)
   ```

3. **Automated Comparison**

   ```python
   def compare_experiments(exp_ids: List[str]) -> pd.DataFrame:
       """Compare multiple experiments across key metrics."""

       comparison_data = []
       for exp_id in exp_ids:
           exp_dir = Path("experiments") / exp_id

           # Load metrics
           with open(exp_dir / "metrics.json") as f:
               metrics = json.load(f)

           # Load config
           with open(exp_dir / "config.yaml") as f:
               config = yaml.safe_load(f)

           comparison_data.append({
               'experiment_id': exp_id,
               'recall@10': metrics.get('test_recall_10', 0),
               'ndcg@10': metrics.get('test_ndcg_10', 0),
               'training_time': metrics.get('training_time_minutes', 0),
               'parameters': metrics.get('total_parameters', 0),
               'model_type': config.get('model', {}).get('use_enhanced', False),
               'embedding_dim': config.get('model', {}).get('embedding_dim', 0)
           })

       return pd.DataFrame(comparison_data).sort_values('recall@10', ascending=False)
   ```

#### 2.1.2 Integration with Training Pipeline

**Files to Modify:**

- `src/trainers/base_trainer.py` - Add experiment tracking
- `src/train.py` and `src/train_improved.py` - Add experiment CLI args
- `Makefile` - Add experiment management commands

**Implementation:**

```python
# Enhanced trainer with experiment tracking
class BaseTrainer:
    def __init__(self, config, experiment_manager=None):
        self.config = config
        self.experiment_manager = experiment_manager

    def train(self):
        # Start experiment tracking
        if self.experiment_manager:
            exp_id = self.experiment_manager.start_experiment(
                name=f"train_{self.config['model_type']}",
                config=self.config
            )

        # Regular training loop
        for epoch in range(self.epochs):
            # ... training code ...

            # Log metrics to experiment
            if self.experiment_manager:
                self.experiment_manager.log_metrics(epoch, metrics)

        # Save final model version
        if self.experiment_manager:
            self.experiment_manager.save_model_version(model, final_metrics)
```

### 2.2 Hyperparameter Optimization

#### 2.2.1 Optuna Integration

**Implementation Steps:**

1. **Create Optimization Framework**

   ```python
   # src/hyperopt.py
   import optuna

   class HyperparameterOptimizer:
       def __init__(self, graph, objective_metric='recall@10'):
           self.graph = graph
           self.objective_metric = objective_metric

       def objective(self, trial):
           """Optuna objective function."""

           # Sample hyperparameters
           config = {
               'model': {
                   'embedding_dim': trial.suggest_int('embedding_dim', 32, 128, step=32),
                   'hidden_dim': trial.suggest_int('hidden_dim', 32, 128, step=32),
                   'num_layers': trial.suggest_int('num_layers', 1, 4),
                   'heads': trial.suggest_categorical('heads', [2, 4, 8]),
                   'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                   'use_enhanced': trial.suggest_categorical('use_enhanced', [True, False])
               },
               'training': {
                   'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
                   'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
                   'epochs': 50  # Fixed for fair comparison
               }
           }

           # Train model with sampled hyperparameters
           trainer = AdvancedTrainer(config, self.graph)
           metrics = trainer.train()

           # Return objective metric
           return metrics[self.objective_metric]

       def optimize(self, n_trials=100):
           """Run hyperparameter optimization."""

           study = optuna.create_study(
               direction='maximize',
               study_name=f'spotify_engine_optimization_{datetime.now():%Y%m%d_%H%M%S}'
           )

           study.optimize(self.objective, n_trials=n_trials)

           return study.best_params, study.best_value
   ```

2. **CLI Integration**

   ```bash
   # New command: optimize hyperparameters
   python -m src.hyperopt --trials 50 --metric recall@10 --timeout 3600

   # Makefile target
   optimize:
    python -m src.hyperopt --trials 20 --metric recall@10
   ```

#### 2.2.2 Advanced Search Strategies

```python
# Multi-objective optimization
def multi_objective_function(trial):
    """Optimize for multiple objectives simultaneously."""

    # Train model
    metrics = train_with_config(trial_config)

    # Return multiple objectives
    return [
        metrics['recall@10'],      # Accuracy
        1.0 / metrics['training_time'],  # Speed (inverse)
        metrics['diversity@10'],   # Diversity
        1.0 / metrics['parameters']     # Model size (inverse)
    ]

# Bayesian optimization with constraints
def constrained_objective(trial):
    """Optimization with resource constraints."""

    # Sample parameters
    config = sample_config(trial)

    # Estimate resource usage
    estimated_memory = estimate_memory_usage(config)
    estimated_time = estimate_training_time(config)

    # Apply constraints
    if estimated_memory > 8000:  # 8GB limit
        raise optuna.TrialPruned()

    if estimated_time > 3600:  # 1 hour limit
        raise optuna.TrialPruned()

    return train_and_evaluate(config)
```

### 2.3 Advanced Training Features

#### 2.3.1 Multi-Objective Training

**Implementation:**

```python
# src/losses.py - Enhanced loss functions
class MultiObjectiveLoss(nn.Module):
    def __init__(self,
                 accuracy_weight=0.7,
                 diversity_weight=0.2,
                 coverage_weight=0.1):
        super().__init__()
        self.accuracy_weight = accuracy_weight
        self.diversity_weight = diversity_weight
        self.coverage_weight = coverage_weight

    def forward(self, predictions, targets, user_histories, catalog_items):
        # Accuracy loss (standard ranking loss)
        accuracy_loss = F.binary_cross_entropy_with_logits(predictions, targets)

        # Diversity loss (encourage diverse recommendations)
        diversity_loss = self._compute_diversity_loss(predictions, user_histories)

        # Coverage loss (encourage long-tail recommendations)
        coverage_loss = self._compute_coverage_loss(predictions, catalog_items)

        total_loss = (
            self.accuracy_weight * accuracy_loss +
            self.diversity_weight * diversity_loss +
            self.coverage_weight * coverage_loss
        )

        return total_loss, {
            'accuracy_loss': accuracy_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'coverage_loss': coverage_loss.item()
        }
```

#### 2.3.2 Advanced Regularization

```python
# Enhanced regularization techniques
class EnhancedRegularization:
    def __init__(self, model):
        self.model = model

    def attention_entropy_loss(self, attention_weights):
        """Encourage diverse attention patterns."""
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8))
        return -entropy  # Negative because we want to maximize entropy

    def embedding_orthogonality_loss(self):
        """Encourage orthogonal embeddings for different entities."""
        losses = []
        embeddings = [
            self.model.user_embedding.weight,
            self.model.song_embedding.weight,
            self.model.artist_embedding.weight
        ]

        for emb in embeddings:
            # Compute Gram matrix
            gram = torch.matmul(emb, emb.t())
            # Encourage orthogonality (Gram matrix close to identity)
            identity = torch.eye(emb.size(0), device=emb.device)
            ortho_loss = F.mse_loss(gram, identity)
            losses.append(ortho_loss)

        return sum(losses) / len(losses)
```

#### 2.3.3 Cross-Validation Support

```python
# src/cross_validation.py
class TimeSeriesCrossValidator:
    """Cross-validation respecting temporal ordering."""

    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, sessions_df):
        """Generate time-aware splits."""

        # Sort by timestamp
        sessions_sorted = sessions_df.sort_values('timestamp')

        # Calculate split points
        n_sessions = len(sessions_sorted)
        test_size_abs = int(n_sessions * self.test_size)

        splits = []
        for i in range(self.n_splits):
            # Progressive train/test split
            test_start = n_sessions - test_size_abs * (i + 1)
            test_end = n_sessions - test_size_abs * i

            train_idx = sessions_sorted.index[:test_start]
            test_idx = sessions_sorted.index[test_start:test_end]

            splits.append((train_idx, test_idx))

        return splits

# Usage in training
def cross_validate_model(config, sessions_df):
    """Perform cross-validation."""

    cv = TimeSeriesCrossValidator(n_splits=5)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(sessions_df)):
        print(f"Training fold {fold + 1}/5")

        # Create train/test splits
        train_sessions = sessions_df.loc[train_idx]
        test_sessions = sessions_df.loc[test_idx]

        # Build graphs
        train_graph = build_graph_from_sessions(train_sessions)
        test_graph = build_graph_from_sessions(test_sessions)

        # Train model
        trainer = AdvancedTrainer(config, train_graph)
        model = trainer.train()

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_graph)
        scores.append(test_metrics)

    # Aggregate results
    avg_metrics = {}
    for metric in scores[0].keys():
        avg_metrics[metric] = np.mean([s[metric] for s in scores])
        avg_metrics[f"{metric}_std"] = np.std([s[metric] for s in scores])

    return avg_metrics
```

### 2.4 Enhanced Evaluation Framework

#### 2.4.1 Comprehensive Metrics Suite

```python
# src/metrics_comprehensive.py
class ComprehensiveEvaluator:
    def __init__(self, model, graph):
        self.model = model
        self.graph = graph

    def evaluate_all(self, test_users, k=10):
        """Comprehensive evaluation across all metrics."""

        results = {}

        # Standard accuracy metrics
        results.update(self._evaluate_accuracy(test_users, k))

        # Diversity metrics
        results.update(self._evaluate_diversity(test_users, k))

        # Coverage metrics
        results.update(self._evaluate_coverage(test_users, k))

        # Fairness metrics
        results.update(self._evaluate_fairness(test_users, k))

        # Efficiency metrics
        results.update(self._evaluate_efficiency(test_users, k))

        return results

    def _evaluate_diversity(self, test_users, k):
        """Evaluate recommendation diversity."""

        diversities = []
        for user_id in test_users:
            recs = self.model.recommend(user_id, k=k)

            # Intra-list diversity (average pairwise distance)
            song_embeddings = self.model.get_song_embeddings(recs)
            pairwise_distances = torch.pdist(song_embeddings)
            avg_distance = pairwise_distances.mean().item()

            diversities.append(avg_distance)

        return {
            'diversity@10': np.mean(diversities),
            'diversity@10_std': np.std(diversities)
        }

    def _evaluate_coverage(self, test_users, k):
        """Evaluate catalog coverage."""

        all_recommendations = set()
        total_recommendations = 0

        for user_id in test_users:
            recs = self.model.recommend(user_id, k=k)
            all_recommendations.update(recs.tolist())
            total_recommendations += len(recs)

        # Catalog coverage
        total_songs = self.graph['song'].num_nodes
        coverage = len(all_recommendations) / total_songs

        # Popularity bias (Gini coefficient)
        rec_counts = Counter()
        for user_id in test_users:
            recs = self.model.recommend(user_id, k=k)
            rec_counts.update(recs.tolist())

        gini = self._compute_gini_coefficient(list(rec_counts.values()))

        return {
            'catalog_coverage': coverage,
            'recommendation_gini': gini,
            'unique_recommendations': len(all_recommendations)
        }
```

#### 2.4.2 A/B Testing Framework

```python
# src/ab_testing.py
class ABTestFramework:
    def __init__(self, control_model, treatment_model):
        self.control_model = control_model
        self.treatment_model = treatment_model

    def run_experiment(self, test_users, split_ratio=0.5, k=10):
        """Run A/B test between two models."""

        # Randomly assign users to control/treatment
        np.random.shuffle(test_users)
        split_point = int(len(test_users) * split_ratio)

        control_users = test_users[:split_point]
        treatment_users = test_users[split_point:]

        # Evaluate both groups
        control_metrics = self._evaluate_group(
            self.control_model, control_users, k
        )
        treatment_metrics = self._evaluate_group(
            self.treatment_model, treatment_users, k
        )

        # Statistical significance testing
        significance_results = self._test_significance(
            control_metrics, treatment_metrics
        )

        return {
            'control': control_metrics,
            'treatment': treatment_metrics,
            'significance': significance_results,
            'sample_sizes': {
                'control': len(control_users),
                'treatment': len(treatment_users)
            }
        }

    def _test_significance(self, control_metrics, treatment_metrics):
        """Test statistical significance of differences."""
        from scipy import stats

        results = {}

        for metric in control_metrics:
            if metric.endswith('_values'):  # Raw values for statistical testing
                control_vals = control_metrics[metric]
                treatment_vals = treatment_metrics[metric]

                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(control_vals, treatment_vals)

                # Effect size (Cohen's d)
                pooled_std = np.sqrt(
                    ((len(control_vals) - 1) * np.var(control_vals) +
                     (len(treatment_vals) - 1) * np.var(treatment_vals)) /
                    (len(control_vals) + len(treatment_vals) - 2)
                )
                cohens_d = (np.mean(treatment_vals) - np.mean(control_vals)) / pooled_std

                results[metric.replace('_values', '')] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'effect_size': cohens_d,
                    'improvement': np.mean(treatment_vals) - np.mean(control_vals)
                }

        return results
```

## Implementation Timeline

### Week 1-2: Experiment Management

- [ ] Implement ExperimentManager class
- [ ] Add versioning and metadata tracking
- [ ] Integrate with existing training pipeline
- [ ] Create comparison and rollback functionality

### Week 3-4: Hyperparameter Optimization

- [ ] Integrate Optuna framework
- [ ] Implement multi-objective optimization
- [ ] Add resource constraints and pruning
- [ ] Create CLI interface for optimization

### Week 5-6: Advanced Training & Evaluation

- [ ] Implement multi-objective loss functions
- [ ] Add advanced regularization techniques
- [ ] Create cross-validation framework
- [ ] Develop comprehensive evaluation suite
- [ ] Build A/B testing infrastructure

## Success Metrics

### Technical Metrics

- **Automation**: 90% of experiments automatically tracked
- **Reproducibility**: 100% of experiments reproducible from metadata
- **Efficiency**: 50% reduction in manual hyperparameter tuning time
- **Coverage**: Comprehensive metrics covering accuracy, diversity, fairness

### Business Metrics

- **Model Performance**: 10-15% improvement in recommendation quality
- **Development Speed**: 30% faster iteration cycles
- **Experiment Throughput**: 3x more experiments per week
- **Decision Quality**: Data-driven model selection with statistical validation

## File Structure

```text
spotify-engine/
├── src/
│   ├── experiment_manager.py      # NEW: Experiment tracking
│   ├── hyperopt.py               # NEW: Hyperparameter optimization
│   ├── losses.py                 # ENHANCED: Multi-objective losses
│   ├── cross_validation.py       # NEW: CV framework
│   ├── metrics_comprehensive.py  # NEW: Extended metrics
│   ├── ab_testing.py             # NEW: A/B testing
│   └── trainers/
│       └── base_trainer.py       # ENHANCED: Experiment integration
├── experiments/                   # NEW: Experiment storage
│   ├── experiment_id_1/
│   │   ├── config.yaml
│   │   ├── model.ckpt
│   │   ├── metrics.json
│   │   └── metadata.json
│   └── experiment_id_2/
└── docs/
    └── technical/
        ├── experiment-management.md  # NEW: Experiment docs
        ├── hyperparameter-tuning.md # NEW: Optimization guide
        └── evaluation-framework.md  # NEW: Evaluation guide
```

## Dependencies

### New Requirements

```txt
# Hyperparameter optimization
optuna>=3.0.0
ray[tune]>=2.0.0  # Alternative to Optuna

# Statistical testing
scipy>=1.9.0
statsmodels>=0.13.0

# Experiment tracking
mlflow>=2.0.0  # Optional: MLflow integration
wandb>=0.13.0  # Optional: Weights & Biases integration

# Configuration management
hydra-core>=1.2.0
omegaconf>=2.2.0
```

### Configuration Updates

```yaml
# config/experiment.yaml
experiment:
  tracking:
    enabled: true
    backend: "local" # or "mlflow", "wandb"
    base_dir: "experiments"

  optimization:
    framework: "optuna"
    n_trials: 50
    timeout: 3600
    pruning: true

  evaluation:
    metrics: ["recall@10", "ndcg@10", "diversity@10", "coverage"]
    cross_validation:
      enabled: true
      n_splits: 5
      strategy: "temporal"
```

## Integration Points

### Makefile Updates

```makefile
# Phase 2 commands
.PHONY: experiment optimize compare-experiments cv-evaluate ab-test

# Start new experiment
experiment:
 python -m src.train_improved --experiment-name $(NAME) --track

# Hyperparameter optimization
optimize:
 python -m src.hyperopt --trials 50 --metric recall@10

# Compare experiments
compare-experiments:
 python -m src.experiment_manager compare --experiments $(EXPS)

# Cross-validation
cv-evaluate:
 python -m src.cross_validation --config config/experiment.yaml

# A/B testing
ab-test:
 python -m src.ab_testing --control $(CONTROL) --treatment $(TREATMENT)
```

This comprehensive plan provides a clear roadmap for implementing Phase 2 features while maintaining the KISS principle and building on the solid foundation established in Phase 1.
