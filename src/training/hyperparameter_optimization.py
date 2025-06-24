"""
Hyperparameter optimization for EEG classification models.
Includes Bayesian optimization, population-based training, and AutoML.
"""

import optuna
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import PopulationBasedTraining, ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from pathlib import Path
import yaml
import json
import logging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import pandas as pd
from functools import partial

logger = logging.getLogger(__name__)


class BayesianOptimization:
    """Bayesian optimization with Gaussian Process surrogate model."""
    
    def __init__(self, param_space: Dict, objective_func: Callable,
                 n_trials: int = 100, n_startup_trials: int = 10,
                 acquisition_func: str = 'EI'):
        self.param_space = param_space
        self.objective_func = objective_func
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.acquisition_func = acquisition_func
        
        # Initialize Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                n_startup_trials=n_startup_trials,
                n_ei_candidates=24
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        # Results tracking
        self.history = []
        
    def _suggest_params(self, trial: optuna.Trial) -> Dict:
        """Suggest parameters for a trial."""
        params = {}
        
        for param_name, param_config in self.param_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            elif param_type == 'discrete':
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name,
                    param_config['low'],
                    param_config['high'],
                    param_config['q']
                )
                
        return params
        
    def optimize(self) -> Dict:
        """Run Bayesian optimization."""
        def objective_wrapper(trial):
            # Suggest parameters
            params = self._suggest_params(trial)
            
            # Evaluate objective
            score = self.objective_func(params, trial)
            
            # Store history
            self.history.append({
                'trial': trial.number,
                'params': params,
                'score': score
            })
            
            return score
            
        # Run optimization
        self.study.optimize(objective_wrapper, n_trials=self.n_trials)
        
        # Get results
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'history': self.history,
            'study': self.study
        }
        
    def get_importance(self) -> pd.DataFrame:
        """Get parameter importance scores."""
        importance = optuna.importance.get_param_importances(self.study)
        return pd.DataFrame(
            importance.items(),
            columns=['parameter', 'importance']
        ).sort_values('importance', ascending=False)


class PopulationBasedTrainer:
    """Population-based training for hyperparameter optimization."""
    
    def __init__(self, trainable_func: Callable, param_space: Dict,
                 num_samples: int = 10, perturbation_interval: int = 5,
                 metric: str = 'val_accuracy', mode: str = 'max'):
        self.trainable_func = trainable_func
        self.param_space = param_space
        self.num_samples = num_samples
        self.perturbation_interval = perturbation_interval
        self.metric = metric
        self.mode = mode
        
        # Setup PBT scheduler
        self.scheduler = PopulationBasedTraining(
            time_attr='training_iteration',
            metric=metric,
            mode=mode,
            perturbation_interval=perturbation_interval,
            hyperparam_mutations=self._get_mutations()
        )
        
    def _get_mutations(self) -> Dict:
        """Get hyperparameter mutations for PBT."""
        mutations = {}
        
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'float':
                mutations[param_name] = tune.uniform(
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'int':
                mutations[param_name] = tune.randint(
                    param_config['low'],
                    param_config['high'] + 1
                )
            elif param_config['type'] == 'categorical':
                mutations[param_name] = tune.choice(param_config['choices'])
                
        return mutations
        
    def train(self, resources_per_trial: Dict = None) -> Dict:
        """Run population-based training."""
        if resources_per_trial is None:
            resources_per_trial = {'cpu': 2, 'gpu': 0.5}
            
        # Configure Ray Tune
        config = {}
        for param_name, param_config in self.param_space.items():
            if param_config['type'] == 'float':
                config[param_name] = tune.uniform(
                    param_config['low'],
                    param_config['high']
                )
            elif param_config['type'] == 'int':
                config[param_name] = tune.randint(
                    param_config['low'],
                    param_config['high'] + 1
                )
            elif param_config['type'] == 'categorical':
                config[param_name] = tune.choice(param_config['choices'])
                
        # Run PBT
        analysis = tune.run(
            self.trainable_func,
            name='pbt_experiment',
            scheduler=self.scheduler,
            config=config,
            num_samples=self.num_samples,
            resources_per_trial=resources_per_trial,
            progress_reporter=CLIReporter(
                metric_columns=[self.metric, 'training_iteration']
            ),
            local_dir='./ray_results'
        )
        
        # Get best trial
        best_trial = analysis.get_best_trial(self.metric, self.mode)
        
        return {
            'best_config': best_trial.config,
            'best_metric': best_trial.last_result[self.metric],
            'analysis': analysis
        }


class MultiObjectiveOptimization:
    """Multi-objective optimization for balancing multiple metrics."""
    
    def __init__(self, param_space: Dict, objectives: List[Dict],
                 n_trials: int = 100):
        self.param_space = param_space
        self.objectives = objectives
        self.n_trials = n_trials
        
        # Create multi-objective study
        directions = [obj['direction'] for obj in objectives]
        self.study = optuna.create_multi_objective_study(
            directions=directions,
            sampler=optuna.samplers.NSGAIISampler()
        )
        
    def optimize(self, objective_func: Callable) -> Dict:
        """Run multi-objective optimization."""
        def objective_wrapper(trial):
            # Suggest parameters
            params = {}
            for param_name, param_config in self.param_space.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['low'],
                        param_config['high']
                    )
                elif param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )
                    
            # Evaluate objectives
            results = objective_func(params)
            
            # Return multiple objectives
            return [results[obj['name']] for obj in self.objectives]
            
        # Run optimization
        self.study.optimize(objective_wrapper, n_trials=self.n_trials)
        
        # Get Pareto front
        pareto_trials = self.study.best_trials
        
        return {
            'pareto_front': [
                {
                    'params': trial.params,
                    'values': trial.values
                }
                for trial in pareto_trials
            ],
            'study': self.study
        }


class NeuralArchitectureSearch:
    """Neural Architecture Search for optimal model design."""
    
    def __init__(self, search_space: Dict, base_model_class,
                 dataset, config: Dict):
        self.search_space = search_space
        self.base_model_class = base_model_class
        self.dataset = dataset
        self.config = config
        
        # ASHA scheduler for early stopping
        self.scheduler = ASHAScheduler(
            metric='val_accuracy',
            mode='max',
            max_t=100,
            grace_period=10,
            reduction_factor=3
        )
        
    def _create_model(self, architecture_config: Dict) -> nn.Module:
        """Create model based on architecture configuration."""
        # Update model config with architecture parameters
        model_config = self.config.copy()
        
        # Update architecture-specific parameters
        if 'num_layers' in architecture_config:
            model_config['models']['resnet1d_gru']['gru']['num_layers'] = \
                architecture_config['num_layers']
                
        if 'hidden_size' in architecture_config:
            model_config['models']['resnet1d_gru']['gru']['hidden_size'] = \
                architecture_config['hidden_size']
                
        if 'num_blocks' in architecture_config:
            model_config['models']['resnet1d_gru']['resnet']['num_blocks'] = \
                architecture_config['num_blocks']
                
        # Create model
        return self.base_model_class(model_config)
        
    def search(self, num_samples: int = 20) -> Dict:
        """Run neural architecture search."""
        def train_model(config):
            # Create model
            model = self._create_model(config)
            
            # Training loop placeholder
            # In practice, this would include full training
            for epoch in range(100):
                # Simulate training
                accuracy = np.random.random()
                
                # Report to Ray Tune
                tune.report(
                    training_iteration=epoch,
                    val_accuracy=accuracy
                )
                
        # Define search space
        search_config = {}
        for param_name, param_config in self.search_space.items():
            if param_config['type'] == 'int':
                search_config[param_name] = tune.randint(
                    param_config['low'],
                    param_config['high'] + 1
                )
            elif param_config['type'] == 'choice':
                search_config[param_name] = tune.choice(param_config['choices'])
                
        # Run search
        analysis = tune.run(
            train_model,
            config=search_config,
            scheduler=self.scheduler,
            num_samples=num_samples,
            resources_per_trial={'cpu': 2, 'gpu': 0.5}
        )
        
        # Get best architecture
        best_trial = analysis.get_best_trial('val_accuracy', 'max')
        
        return {
            'best_architecture': best_trial.config,
            'best_accuracy': best_trial.last_result['val_accuracy'],
            'analysis': analysis
        }


class HyperparameterOptimizationPipeline:
    """Complete hyperparameter optimization pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.optimization_config = config.get('hyperparameter_optimization', {})
        
    def create_param_space(self, model_type: str) -> Dict:
        """Create parameter space for optimization."""
        if model_type == 'resnet1d_gru':
            return {
                'learning_rate': {
                    'type': 'float',
                    'low': 1e-5,
                    'high': 1e-2,
                    'log': True
                },
                'batch_size': {
                    'type': 'categorical',
                    'choices': [16, 32, 64, 128]
                },
                'dropout': {
                    'type': 'float',
                    'low': 0.1,
                    'high': 0.5
                },
                'weight_decay': {
                    'type': 'float',
                    'low': 1e-6,
                    'high': 1e-3,
                    'log': True
                },
                'hidden_size': {
                    'type': 'categorical',
                    'choices': [128, 256, 512]
                },
                'num_layers': {
                    'type': 'int',
                    'low': 1,
                    'high': 4
                }
            }
        elif model_type == 'efficientnet':
            return {
                'learning_rate': {
                    'type': 'float',
                    'low': 1e-5,
                    'high': 5e-3,
                    'log': True
                },
                'batch_size': {
                    'type': 'categorical',
                    'choices': [8, 16, 32, 64]
                },
                'dropout': {
                    'type': 'float',
                    'low': 0.2,
                    'high': 0.6
                },
                'drop_path_rate': {
                    'type': 'float',
                    'low': 0.1,
                    'high': 0.4
                },
                'label_smoothing': {
                    'type': 'float',
                    'low': 0.0,
                    'high': 0.2
                }
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def optimize_hyperparameters(self, model_type: str, 
                               train_func: Callable,
                               eval_func: Callable,
                               dataset) -> Dict:
        """Run hyperparameter optimization."""
        # Create parameter space
        param_space = self.create_param_space(model_type)
        
        # Select optimization method
        method = self.optimization_config.get('method', 'bayesian')
        
        if method == 'bayesian':
            # Bayesian optimization
            optimizer = BayesianOptimization(
                param_space=param_space,
                objective_func=lambda params, trial: self._objective(
                    params, trial, model_type, train_func, eval_func, dataset
                ),
                n_trials=self.optimization_config.get('n_trials', 100)
            )
            results = optimizer.optimize()
            
        elif method == 'pbt':
            # Population-based training
            trainer = PopulationBasedTrainer(
                trainable_func=partial(
                    self._trainable,
                    model_type=model_type,
                    train_func=train_func,
                    eval_func=eval_func,
                    dataset=dataset
                ),
                param_space=param_space,
                num_samples=self.optimization_config.get('num_samples', 10)
            )
            results = trainer.train()
            
        elif method == 'multi_objective':
            # Multi-objective optimization
            objectives = [
                {'name': 'accuracy', 'direction': 'maximize'},
                {'name': 'inference_time', 'direction': 'minimize'}
            ]
            optimizer = MultiObjectiveOptimization(
                param_space=param_space,
                objectives=objectives,
                n_trials=self.optimization_config.get('n_trials', 100)
            )
            results = optimizer.optimize(
                lambda params: self._multi_objective(
                    params, model_type, train_func, eval_func, dataset
                )
            )
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
        # Save results
        self._save_results(results, model_type)
        
        return results
        
    def _objective(self, params: Dict, trial: optuna.Trial,
                  model_type: str, train_func: Callable,
                  eval_func: Callable, dataset) -> float:
        """Objective function for optimization."""
        # Update configuration with suggested parameters
        config = self.config.copy()
        model_config = config['models'][model_type]
        
        # Update parameters
        if 'learning_rate' in params:
            model_config['training']['learning_rate'] = params['learning_rate']
        if 'batch_size' in params:
            model_config['training']['batch_size'] = params['batch_size']
        if 'dropout' in params:
            model_config['dropout'] = params['dropout']
            
        # Train model
        try:
            model = self._create_model(model_type, config)
            metrics = train_func(model, dataset, config, trial)
            
            # Evaluate
            val_metrics = eval_func(model, dataset, config)
            
            return val_metrics['balanced_accuracy']
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return 0.0
            
    def _trainable(self, config: Dict, model_type: str,
                  train_func: Callable, eval_func: Callable,
                  dataset) -> None:
        """Trainable function for Ray Tune."""
        # Implementation for Ray Tune training
        pass
        
    def _multi_objective(self, params: Dict, model_type: str,
                        train_func: Callable, eval_func: Callable,
                        dataset) -> Dict:
        """Multi-objective evaluation."""
        # Train and evaluate model
        # Return multiple metrics
        return {
            'accuracy': 0.9,  # Placeholder
            'inference_time': 0.1  # Placeholder
        }
        
    def _create_model(self, model_type: str, config: Dict) -> nn.Module:
        """Create model instance."""
        # Import model class based on type
        if model_type == 'resnet1d_gru':
            from ..models import ResNet1D_GRU
            return ResNet1D_GRU(config)
        elif model_type == 'efficientnet':
            from ..models import EfficientNetSpectrogram
            return EfficientNetSpectrogram(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def _save_results(self, results: Dict, model_type: str):
        """Save optimization results."""
        save_dir = Path(self.config['paths']['logs_dir']) / 'hyperopt_results'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(save_dir / f'{model_type}_results.json', 'w') as f:
            # Convert non-serializable objects
            save_dict = {
                'best_params': results.get('best_params', results.get('best_config')),
                'best_value': results.get('best_value', results.get('best_metric')),
                'timestamp': str(pd.Timestamp.now())
            }
            json.dump(save_dict, f, indent=2)
            
        logger.info(f"Saved hyperparameter optimization results to {save_dir}")


class AutoMLPipeline:
    """Automated machine learning pipeline."""
    
    def __init__(self, config: Dict, time_budget: int = 3600):
        self.config = config
        self.time_budget = time_budget
        
    def run(self, dataset, target_metric: str = 'balanced_accuracy') -> Dict:
        """Run AutoML pipeline."""
        # This would integrate with AutoML frameworks like
        # AutoGluon, H2O, or custom AutoML implementation
        
        results = {
            'best_model': None,
            'best_params': {},
            'best_score': 0.0,
            'search_history': []
        }
        
        return results 