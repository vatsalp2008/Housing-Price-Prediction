"""
Base Learners for Stacked Ensemble
Implements XGBoost, LightGBM, and Random Forest regressors
"""

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple
import logging
import sys
sys.path.append('..')
from config import (
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    RANDOM_FOREST_PARAMS,
    MODEL_CONFIG
)

logger = logging.getLogger(__name__)


def get_xgboost_model(params: dict = None) -> XGBRegressor:
    """
    Get XGBoost regressor with default or custom parameters
    
    Args:
        params: Custom parameters (optional)
        
    Returns:
        Configured XGBRegressor
    """
    model_params = XGBOOST_PARAMS.copy()
    if params:
        model_params.update(params)
    
    logger.info("Initializing XGBoost regressor")
    return XGBRegressor(**model_params)


def get_lightgbm_model(params: dict = None) -> LGBMRegressor:
    """
    Get LightGBM regressor with default or custom parameters
    
    Args:
        params: Custom parameters (optional)
        
    Returns:
        Configured LGBMRegressor
    """
    model_params = LIGHTGBM_PARAMS.copy()
    if params:
        model_params.update(params)
    
    logger.info("Initializing LightGBM regressor")
    return LGBMRegressor(**model_params)


def get_random_forest_model(params: dict = None) -> RandomForestRegressor:
    """
    Get Random Forest regressor with default or custom parameters
    
    Args:
        params: Custom parameters (optional)
        
    Returns:
        Configured RandomForestRegressor
    """
    model_params = RANDOM_FOREST_PARAMS.copy()
    if params:
        model_params.update(params)
    
    logger.info("Initializing Random Forest regressor")
    return RandomForestRegressor(**model_params)


def get_base_learners(custom_params: dict = None) -> List[Tuple[str, object]]:
    """
    Get all base learners for stacking
    
    Args:
        custom_params: Dictionary with custom parameters for each model
                      Format: {'xgboost': {...}, 'lightgbm': {...}, 'rf': {...}}
        
    Returns:
        List of (name, model) tuples
    """
    custom_params = custom_params or {}
    
    base_learners = [
        ('xgboost', get_xgboost_model(custom_params.get('xgboost'))),
        ('lightgbm', get_lightgbm_model(custom_params.get('lightgbm'))),
        ('random_forest', get_random_forest_model(custom_params.get('rf'))),
    ]
    
    logger.info(f"Created {len(base_learners)} base learners for ensemble")
    
    return base_learners


class BaseModelEvaluator:
    """Evaluate individual base models before stacking"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, name: str, model, X_train, y_train, X_test, y_test):
        """
        Evaluate a single model
        
        Args:
            name: Model name
            model: Model instance
            X_train, y_train: Training data
            X_test, y_test: Test data
        """
        from sklearn.metrics import mean_squared_error, r2_score
        import numpy as np
        
        # Train model
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        self.results[name] = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
        }
        
        logger.info(f"{name} - Test RMSE: ${test_rmse:,.2f}, Test R²: {test_r2:.4f}")
        
        return model
    
    def evaluate_all(self, base_learners, X_train, y_train, X_test, y_test):
        """Evaluate all base learners"""
        trained_models = []
        
        for name, model in base_learners:
            trained_model = self.evaluate_model(name, model, X_train, y_train, X_test, y_test)
            trained_models.append((name, trained_model))
        
        return trained_models
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "=" * 70)
        print("BASE LEARNER EVALUATION SUMMARY")
        print("=" * 70)
        print(f"{'Model':<20} {'Train RMSE':<15} {'Test RMSE':<15} {'Test R²':<10}")
        print("-" * 70)
        
        for name, metrics in self.results.items():
            print(f"{name:<20} ${metrics['train_rmse']:<14,.2f} "
                  f"${metrics['test_rmse']:<14,.2f} {metrics['test_r2']:<10.4f}")
        
        print("=" * 70)
