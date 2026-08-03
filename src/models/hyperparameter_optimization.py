"""
Hyperparameter Optimization using Optuna
Automated tuning for base learners with custom RMSE objective
"""

import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import logging
import sys
from pathlib import Path
# Make src/ importable when this module is run directly; the bare
# sys.path.append('..') this replaces was relative to the caller's cwd
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import OPTUNA_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OptunaOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, n_trials: int = None, quick_mode: bool = False,
                 timeout: int = None):
        """
        Initialize optimizer

        Args:
            n_trials: Number of optimization trials. Wins over quick_mode when
                given explicitly.
            quick_mode: Use reduced trials for quick testing
            timeout: Per-study wall-clock budget in seconds (defaults to
                OPTUNA_CONFIG)
        """
        if n_trials is not None:
            self.n_trials = n_trials
        elif quick_mode:
            self.n_trials = OPTUNA_CONFIG['n_trials_quick']
        else:
            self.n_trials = OPTUNA_CONFIG['n_trials']

        self.timeout = timeout if timeout is not None else OPTUNA_CONFIG['timeout']

        self.best_params = {}
        self.studies = {}

    def _run_study(self, name: str, objective, X, y):
        """
        Create and run a study for one base learner

        Args:
            name: Model name used for logging and result keys
            objective: Objective callable taking (trial, X, y)
            X: Features
            y: Target

        Returns:
            The completed study
        """
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=MODEL_CONFIG['random_state'])
        )

        study.optimize(
            lambda trial: objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )

        logger.info(f"Best RMSE: ${study.best_value:,.2f}")
        logger.info(f"Best parameters: {study.best_params}")
        logger.info(f"Trials completed: {len(study.trials)}")

        return study
        
    def _objective_xgboost(self, trial, X, y):
        """
        Objective function for XGBoost optimization
        
        Args:
            trial: Optuna trial
            X: Features
            y: Target
            
        Returns:
            Mean RMSE from cross-validation
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': MODEL_CONFIG['random_state'],
            'n_jobs': MODEL_CONFIG['n_jobs'],
        }
        
        model = XGBRegressor(**params)
        
        # Cross-validation with negative MSE
        cv_scores = cross_val_score(
            model, X, y,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=1  # XGBoost already uses n_jobs
        )
        
        # Return RMSE (lower is better)
        rmse = np.sqrt(-cv_scores.mean())
        return rmse
    
    def _objective_lightgbm(self, trial, X, y):
        """
        Objective function for LightGBM optimization
        
        Args:
            trial: Optuna trial
            X: Features
            y: Target
            
        Returns:
            Mean RMSE from cross-validation
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'random_state': MODEL_CONFIG['random_state'],
            'n_jobs': MODEL_CONFIG['n_jobs'],
            'verbose': -1,
        }
        
        model = LGBMRegressor(**params)
        
        cv_scores = cross_val_score(
            model, X, y,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        rmse = np.sqrt(-cv_scores.mean())
        return rmse
    
    def _objective_random_forest(self, trial, X, y):
        """
        Objective function for Random Forest optimization
        
        Args:
            trial: Optuna trial
            X: Features
            y: Target
            
        Returns:
            Mean RMSE from cross-validation
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
            'max_depth': trial.suggest_int('max_depth', 10, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': MODEL_CONFIG['random_state'],
            'n_jobs': MODEL_CONFIG['n_jobs'],
        }
        
        model = RandomForestRegressor(**params)
        
        cv_scores = cross_val_score(
            model, X, y,
            cv=MODEL_CONFIG['cv_folds'],
            scoring='neg_mean_squared_error',
            n_jobs=1
        )
        
        rmse = np.sqrt(-cv_scores.mean())
        return rmse
    
    def optimize_xgboost(self, X, y):
        """
        Optimize XGBoost hyperparameters
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Best parameters
        """
        logger.info("=" * 70)
        logger.info("OPTIMIZING XGBOOST HYPERPARAMETERS")
        logger.info("=" * 70)
        logger.info(f"Trials: {self.n_trials}, timeout: {self.timeout}s")

        study = self._run_study('xgboost', self._objective_xgboost, X, y)

        self.studies['xgboost'] = study
        self.best_params['xgboost'] = study.best_params

        return study.best_params
    
    def optimize_lightgbm(self, X, y):
        """
        Optimize LightGBM hyperparameters
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Best parameters
        """
        logger.info("=" * 70)
        logger.info("OPTIMIZING LIGHTGBM HYPERPARAMETERS")
        logger.info("=" * 70)
        logger.info(f"Trials: {self.n_trials}, timeout: {self.timeout}s")

        study = self._run_study('lightgbm', self._objective_lightgbm, X, y)

        self.studies['lightgbm'] = study
        self.best_params['lightgbm'] = study.best_params

        return study.best_params
    
    def optimize_random_forest(self, X, y):
        """
        Optimize Random Forest hyperparameters
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Best parameters
        """
        logger.info("=" * 70)
        logger.info("OPTIMIZING RANDOM FOREST HYPERPARAMETERS")
        logger.info("=" * 70)
        logger.info(f"Trials: {self.n_trials}, timeout: {self.timeout}s")

        study = self._run_study('random_forest', self._objective_random_forest, X, y)

        self.studies['random_forest'] = study
        self.best_params['rf'] = study.best_params  # Note: using 'rf' key for consistency

        return study.best_params
    
    def optimize_all(self, X, y):
        """
        Optimize all base learners
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary of best parameters for all models
        """
        logger.info("\n" + "=" * 70)
        logger.info("STARTING HYPERPARAMETER OPTIMIZATION FOR ALL BASE LEARNERS")
        logger.info("=" * 70)
        
        # Optimize each model
        self.optimize_xgboost(X, y)
        self.optimize_lightgbm(X, y)
        self.optimize_random_forest(X, y)
        
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        
        return self.best_params
    
    def get_optimization_summary(self):
        """Get summary of optimization results"""
        summary = {}
        
        for model_name, study in self.studies.items():
            summary[model_name] = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
            }
        
        return summary
    
    def save_best_params(self, filepath: str = None):
        """
        Save best parameters to file
        
        Args:
            filepath: Path to save file
        """
        import json
        from config import OUTPUT_DIR
        
        filepath = filepath or OUTPUT_DIR / 'best_hyperparameters.json'
        
        with open(filepath, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        logger.info(f"Best parameters saved to {filepath}")


def main():
    """Test hyperparameter optimization"""
    from data_acquisition import DataAcquisition
    from utils.preprocessing import DataPreprocessor, prepare_features_target
    from feature_engineering import FeatureEngineeringPipeline
    
    # Load and prepare data (use subset for quick testing)
    acquisition = DataAcquisition()
    df, _ = acquisition.prepare_dataset()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df, fit=True)
    df = preprocessor.remove_extreme_outliers(df)
    
    # Use subset for quick testing
    df = df.sample(n=500, random_state=42)
    
    X, y = prepare_features_target(df)
    
    fe_pipeline = FeatureEngineeringPipeline()
    X_transformed = fe_pipeline.fit_transform(X, y)
    
    # Optimize (quick mode)
    optimizer = OptunaOptimizer(quick_mode=True)
    best_params = optimizer.optimize_all(X_transformed, y)
    
    print("\n" + "=" * 70)
    print("BEST HYPERPARAMETERS")
    print("=" * 70)
    for model, params in best_params.items():
        print(f"\n{model.upper()}:")
        for param, value in params.items():
            print(f"  {param}: {value}")


if __name__ == "__main__":
    main()
