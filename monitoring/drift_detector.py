
"""
Data drift detection using Kolmogorov-Smirnov test.

Monitors:
- Input distribution shifts
- Prediction distribution changes
- Model confidence degradation
"""

from scipy.stats import ks_2samp
import numpy as np
from typing import Dict


class DriftDetector:
    """
    Statistical drift detection for production monitoring.
    
    Uses two-sample Kolmogorov-Smirnov test to compare:
    - Reference distribution (training data)
    - Current distribution (production data)
    """
    
    def __init__(self, reference_data: np.ndarray, threshold: float = 0.05):
        """
        Args:
            reference_data: Reference distribution [n_samples, n_features]
            threshold: P-value threshold for drift detection
        """
        self.reference_data = reference_data
        self.threshold = threshold
    
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, bool]:
        """
        Detect drift for each feature.
        
        Returns:
            Dictionary of {feature_name: is_drifting}
        """
        drift_detected = {}
        
        for i in range(self.reference_data.shape[1]):
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]
            
            # KS test: null hypothesis = same distribution
            statistic, p_value = ks_2samp(ref_feature, curr_feature)
            
            # Reject null if p < threshold → drift detected
            drift_detected[f"feature_{i}"] = p_value < self.threshold
        
        return drift_detected
    
    def monitor(self, current_data: np.ndarray):
        """Log drift detection results"""
        results = self.detect_drift(current_data)
        
        drifting_features = [k for k, v in results.items() if v]
        
        if drifting_features:
            print(f"  DRIFT DETECTED in features: {drifting_features}")
            print("Action: Consider retraining model with recent data")
        else:
            print("No drift detected")
