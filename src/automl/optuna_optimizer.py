
"""
AutoML: Automated hyperparameter tuning using Optuna.

Optuna uses Tree-structured Parzen Estimator (TPE):
- Builds probabilistic model of objective function
- Balances exploration vs exploitation
- Prunes bad trials early
"""

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
import torch
from typing import Dict, Any
from ..training.trainer import Trainer
from ..models.transformer import TransformerClassifier


def objective(trial, config: Dict[str, Any], train_loader, val_loader, device):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    num_heads = trial.suggest_categorical('num_heads', [4, 8, 16])
    num_layers = trial.suggest_int('num_layers', 2, 6)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    
    # Build model
    model = TransformerClassifier(
        n_features=config['data']['n_features'],
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_model * 4,
        n_classes=2,
        dropout=dropout
    )
    
    # Update config
    temp_config = config.copy()
    temp_config['training']['learning_rate'] = learning_rate
    
    # Train
    trainer = Trainer(model, temp_config, device, logger=None)
    best_f1 = trainer.fit(train_loader, val_loader, experiment_name="optuna_search")
    
    return best_f1


def run_hyperparameter_search(config: Dict[str, Any], train_loader, val_loader, device, n_trials: int = 20):
    """Run Optuna hyperparameter optimization"""
    
    study = optuna.create_study(direction='maximize')
    study.optimize(
        lambda trial: objective(trial, config, train_loader, val_loader, device),
        n_trials=n_trials
    )
    
    print("Best trial:")
    print(f"  Value (F1): {study.best_value:.4f}")
    print("  Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # Visualize
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    
    return study.best_params
