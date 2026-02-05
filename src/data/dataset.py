import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict


class SensorDataGenerator:
    """Generate synthetic sensor time-series data"""
    
    def __init__(
        self,
        n_samples: int = 10000,
        seq_length: int = 100,
        n_features: int = 4,
        failure_ratio: float = 0.2,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.seq_length = seq_length
        self.n_features = n_features
        self.failure_ratio = failure_ratio
        np.random.seed(seed)
    
    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sensor data.
        
        Returns:
            X: [n_samples, seq_length, n_features]
            y: [n_samples] (0=normal, 1=failure)
        """
        X = []
        y = []
        
        for i in range(self.n_samples):
            is_failure = np.random.random() < self.failure_ratio
            
            if is_failure:
                # Failure trajectory
                temp = np.linspace(30, 95, self.seq_length) + np.random.normal(0, 3, self.seq_length)
                vib = np.linspace(2, 18, self.seq_length) + np.random.normal(0, 1.5, self.seq_length)
                pres = np.linspace(100, 65, self.seq_length) + np.random.normal(0, 2, self.seq_length)
                rpm = np.linspace(1500, 1150, self.seq_length) + np.random.normal(0, 40, self.seq_length)
                label = 1
            else:
                # Normal operation
                temp = np.random.normal(30, 4, self.seq_length)
                vib = np.random.normal(3, 0.8, self.seq_length)
                pres = np.random.normal(100, 3, self.seq_length)
                rpm = np.random.normal(1500, 30, self.seq_length)
                label = 0
            
            # Stack features: [seq_length, 4]
            sequence = np.stack([temp, vib, pres, rpm], axis=1)
            X.append(sequence)
            y.append(label)
        
        return np.array(X), np.array(y)


class SensorDataset(Dataset):
    """PyTorch Dataset for sensor data"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: Optional[StandardScaler] = None):
        """
        Args:
            X: [n_samples, seq_length, n_features]
            y: [n_samples]
            scaler: Fitted StandardScaler or None
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        
        if scaler is not None:
            # Reshape for scaling: [n_samples*seq_length, n_features]
            n_samples, seq_len, n_features = X.shape
            X_reshaped = X.reshape(-1, n_features)
            X_scaled = scaler.transform(X_reshaped)
            self.X = torch.FloatTensor(X_scaled.reshape(n_samples, seq_len, n_features))
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    config: Dict,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    Create train/val/test dataloaders.
    
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    # Generate data
    generator = SensorDataGenerator(
        n_samples=config['data']['n_samples'],
        seq_length=config['data']['sequence_length'],
        n_features=config['data']['n_features'],
        seed=config['data']['random_seed']
    )
    
    X, y = generator.generate()
    
    # Split data
    from sklearn.model_selection import train_test_split
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # Fit scaler on training data only
    scaler = StandardScaler()
    n_samples, seq_len, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    scaler.fit(X_train_reshaped)
    
    # Create datasets
    train_dataset = SensorDataset(X_train, y_train, scaler)
    val_dataset = SensorDataset(X_val, y_val, scaler)
    test_dataset = SensorDataset(X_test, y_test, scaler)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler
