import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import mlflow
import mlflow.pytorch
from tqdm import tqdm


class Trainer:
    """
    Production-ready training loop with MLflow integration.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: torch.device,
        logger: Optional[Any] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = logger
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['training']['epochs']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Best model tracking
        self.best_f1 = 0.0
        self.best_model_state = None
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        # Calculate F1 score
        from sklearn.metrics import f1_score
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str = "transformer_experiment"
    ) -> float:
        """Full training loop with MLflow tracking"""
        
        epochs = self.config['training']['epochs']
        
        # MLflow experiment setup
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(self.config['training'])
            mlflow.log_params(self.config['model']['transformer'])
            
            for epoch in range(epochs):
                if self.logger:
                    self.logger.info(f"Epoch {epoch+1}/{epochs}")
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Learning rate step
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'val_f1': val_metrics['f1_score'],
                    'learning_rate': current_lr
                }, step=epoch)
                
                if self.logger:
                    self.logger.log_metrics(train_metrics, step=epoch)
                    self.logger.log_metrics(val_metrics, step=epoch)
                
                # Save best model
                if val_metrics['f1_score'] > self.best_f1:
                    self.best_f1 = val_metrics['f1_score']
                    self.best_model_state = self.model.state_dict().copy()
                    
                    # Save checkpoint
                    import os
                    os.makedirs('models/saved', exist_ok=True)
                    torch.save(self.best_model_state, 'models/saved/best_model.pth')
                    
                    if self.logger:
                        self.logger.info(f"New best F1: {self.best_f1:.4f}")
            
            # Log best model to MLflow
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)
                mlflow.pytorch.log_model(self.model, "best_model")
                mlflow.log_metric("best_f1_score", self.best_f1)
        
        return self.best_f1
