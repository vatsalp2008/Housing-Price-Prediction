"""
Metrics and evaluation utilities
Custom scoring functions and performance reporting
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,  # Mean Absolute Percentage Error
        'Max_Error': np.max(np.abs(y_true - y_pred)),
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], dataset_name: str = "Dataset"):
    """
    Print metrics in a formatted way
    
    Args:
        metrics: Dictionary of metrics
        dataset_name: Name of the dataset (e.g., "Train", "Test")
    """
    print(f"\n{'=' * 60}")
    print(f"{dataset_name} Performance Metrics")
    print('=' * 60)
    
    for metric_name, value in metrics.items():
        if metric_name in ['RMSE', 'MAE', 'Max_Error']:
            print(f"{metric_name:15s}: ${value:,.2f}")
        elif metric_name == 'MAPE':
            print(f"{metric_name:15s}: {value:.2f}%")
        else:
            print(f"{metric_name:15s}: {value:.4f}")
    
    print('=' * 60)


def calculate_prediction_intervals(y_pred: np.ndarray, residuals: np.ndarray, 
                                   confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals
    
    Args:
        y_pred: Predicted values
        residuals: Residuals from predictions
        confidence: Confidence level (default 95%)
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Calculate standard error
    std_error = np.std(residuals)
    
    # Calculate z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate intervals
    margin = z_score * std_error
    lower_bound = y_pred - margin
    upper_bound = y_pred + margin
    
    return lower_bound, upper_bound


def rmse_cv_score(model, X, y, cv=5):
    """
    Calculate cross-validated RMSE score
    
    Args:
        model: Scikit-learn model
        X: Features
        y: Target
        cv: Number of cross-validation folds
        
    Returns:
        Mean RMSE across folds
    """
    from sklearn.model_selection import cross_val_score
    
    # Note: cross_val_score with neg_mean_squared_error returns negative values
    scores = cross_val_score(model, X, y, 
                            scoring='neg_mean_squared_error', 
                            cv=cv, 
                            n_jobs=-1)
    
    rmse_scores = np.sqrt(-scores)
    
    logger.info(f"CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
    
    return rmse_scores.mean()


class PerformanceReport:
    """Generate comprehensive performance reports"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.train_metrics = None
        self.test_metrics = None
        
    def add_train_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Add training set metrics"""
        self.train_metrics = calculate_metrics(y_true, y_pred)
        
    def add_test_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Add test set metrics"""
        self.test_metrics = calculate_metrics(y_true, y_pred)
        
    def print_report(self):
        """Print comprehensive report"""
        print(f"\n{'#' * 70}")
        print(f"# {self.model_name:^66s} #")
        print(f"{'#' * 70}\n")
        
        if self.train_metrics:
            print_metrics(self.train_metrics, "Training Set")
        
        if self.test_metrics:
            print_metrics(self.test_metrics, "Test Set")
            
        # Check for overfitting
        if self.train_metrics and self.test_metrics:
            r2_diff = self.train_metrics['R2'] - self.test_metrics['R2']
            if r2_diff > 0.1:
                print(f"\n⚠️  WARNING: Potential overfitting detected!")
                print(f"   R² difference: {r2_diff:.4f}")
            else:
                print(f"\n✓ Model generalization looks good (R² diff: {r2_diff:.4f})")
    
    def to_dict(self) -> Dict:
        """Export report as dictionary"""
        return {
            'model_name': self.model_name,
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
        }
