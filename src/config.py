"""
Configuration module for Housing Valuation Engine
Centralized configuration management for paths, parameters, and constants
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models_saved"


def ensure_directories():
    """
    Create the project's data, output and model directories

    Called on import so paths are ready to use, and exposed so callers can
    recreate them after a cleanup.
    """
    for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)


ensure_directories()

# Data URLs
AMES_DATASET_URL = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
AMES_DATASET_BACKUP_URL = "https://raw.githubusercontent.com/npranav10/ames_housing/master/AmesHousing.txt"

# Macroeconomic Indicators (as of January 2026)
# These can be updated with current values
MACRO_INDICATORS = {
    'mortgage_rate_30yr': 6.5,  # 30-year fixed mortgage rate (%)
    'cpi': 320.0,  # Consumer Price Index (base year 1982-84 = 100)
    'reference_year': 2026
}

# Feature Engineering Parameters
FEATURE_CONFIG = {
    'skewness_threshold': 0.75,  # Threshold for Box-Cox transformation
    'target_encoding_smoothing': 10,  # Smoothing parameter for target encoding
    # Column names follow the Ames dataset, which separates words with spaces.
    # The compact Kaggle-style spellings used previously matched nothing, so
    # every interaction term was silently skipped.
    'interaction_terms': [
        ('Overall Qual', 'Gr Liv Area'),
        ('Year Built', 'Total Bsmt SF'),
        ('Garage Area', 'Garage Cars'),
        ('1st Flr SF', '2nd Flr SF'),
        ('Overall Qual', 'Year Built'),
    ],
    'high_cardinality_cols': ['Neighborhood', 'Exterior 1st', 'Exterior 2nd'],
}

# Model Parameters
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'cv_folds': 5,
    'n_jobs': -1,
}

# Optuna Hyperparameter Optimization
OPTUNA_CONFIG = {
    'n_trials': 100,  # Number of optimization trials (reduce for quick testing)
    'n_trials_quick': 10,  # Quick mode for testing
    'timeout': 3600,  # Maximum optimization time in seconds (1 hour)
    'n_jobs': -1,
}

# Base Learner Default Hyperparameters (before optimization)
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': MODEL_CONFIG['random_state'],
    'n_jobs': MODEL_CONFIG['n_jobs'],
}

LIGHTGBM_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': MODEL_CONFIG['random_state'],
    'n_jobs': MODEL_CONFIG['n_jobs'],
    'verbose': -1,
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': MODEL_CONFIG['random_state'],
    'n_jobs': MODEL_CONFIG['n_jobs'],
}

# Meta-learner (Bayesian Ridge)
BAYESIAN_RIDGE_PARAMS = {
    'max_iter': 300,  # 'n_iter' was deprecated in scikit-learn 1.3, removed in 1.5
    'alpha_1': 1e-6,
    'alpha_2': 1e-6,
    'lambda_1': 1e-6,
    'lambda_2': 1e-6,
}

# SHAP Configuration
SHAP_CONFIG = {
    'max_display': 20,  # Number of features to display in summary plot
    'sample_size': 100,  # Number of samples for SHAP calculation (reduce for speed)
}

# LIME Configuration
LIME_CONFIG = {
    'num_features': 10,  # Number of features in explanation
    'num_samples': 5000,  # Number of samples for LIME
}

# Logging
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# Performance Thresholds
PERFORMANCE_TARGETS = {
    'r2_score': 0.85,
    'rmse': 25000,  # Target RMSE in dollars
}
