"""
Stacked Ensemble Model
Implements stacking with Bayesian Ridge meta-learner
"""

import numpy as np
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
import joblib
import logging
import sys
sys.path.append('..')
from config import BAYESIAN_RIDGE_PARAMS, MODEL_CONFIG, MODELS_DIR
from .base_learners import get_base_learners

logger = logging.getLogger(__name__)


class StackedEnsembleModel:
    """Stacked ensemble with multiple base learners and Bayesian Ridge meta-learner"""
    
    def __init__(self, base_learner_params: dict = None, meta_learner_params: dict = None):
        """
        Initialize stacked ensemble
        
        Args:
            base_learner_params: Custom parameters for base learners
            meta_learner_params: Custom parameters for meta-learner
        """
        self.base_learner_params = base_learner_params
        self.meta_learner_params = meta_learner_params or BAYESIAN_RIDGE_PARAMS
        self.model = None
        self.feature_names = None
        
    def build_model(self):
        """Build the stacking regressor"""
        # Get base learners
        base_learners = get_base_learners(self.base_learner_params)
        
        # Create meta-learner
        meta_learner = BayesianRidge(**self.meta_learner_params)
        
        # Create stacking regressor
        self.model = StackingRegressor(
            estimators=base_learners,
            final_estimator=meta_learner,
            cv=MODEL_CONFIG['cv_folds'],
            n_jobs=MODEL_CONFIG['n_jobs'],
            verbose=1
        )
        
        logger.info("Stacked ensemble model built successfully")
        logger.info(f"Base learners: {[name for name, _ in base_learners]}")
        logger.info(f"Meta-learner: Bayesian Ridge Regression")
        
        return self.model
    
    def train(self, X_train, y_train, feature_names: list = None):
        """
        Train the stacked ensemble
        
        Args:
            X_train: Training features
            y_train: Training target
            feature_names: List of feature names (optional)
        """
        if self.model is None:
            self.build_model()
        
        self.feature_names = feature_names
        
        logger.info("=" * 70)
        logger.info("TRAINING STACKED ENSEMBLE")
        logger.info("=" * 70)
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Cross-validation folds: {MODEL_CONFIG['cv_folds']}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        logger.info("Training complete!")
        
        return self.model
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def cross_validate(self, X, y, cv: int = None):
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            cv: Number of folds (default from config)
            
        Returns:
            Cross-validation scores
        """
        if self.model is None:
            self.build_model()
        
        cv = cv or MODEL_CONFIG['cv_folds']
        
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Negative MSE scores
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=MODEL_CONFIG['n_jobs'],
            verbose=1
        )
        
        # Convert to RMSE
        rmse_scores = np.sqrt(-cv_scores)
        
        logger.info(f"CV RMSE: {rmse_scores.mean():.2f} (+/- {rmse_scores.std():.2f})")
        
        return rmse_scores
    
    def get_base_predictions(self, X):
        """
        Get predictions from individual base learners
        
        Args:
            X: Features
            
        Returns:
            Dictionary of base learner predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        base_predictions = {}
        
        for name, estimator in self.model.estimators_:
            base_predictions[name] = estimator.predict(X)
        
        return base_predictions
    
    def save_model(self, filename: str = 'stacked_ensemble.pkl'):
        """
        Save trained model to disk
        
        Args:
            filename: Name of file to save
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        filepath = MODELS_DIR / filename
        
        # Save model and feature names
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'base_learner_params': self.base_learner_params,
            'meta_learner_params': self.meta_learner_params,
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, filename: str = 'stacked_ensemble.pkl'):
        """
        Load trained model from disk
        
        Args:
            filename: Name of file to load
        """
        filepath = MODELS_DIR / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names')
        self.base_learner_params = model_data.get('base_learner_params')
        self.meta_learner_params = model_data.get('meta_learner_params')
        
        logger.info(f"Model loaded from {filepath}")
        
        return self.model
    
    def get_model_info(self):
        """Get information about the model"""
        if self.model is None:
            return "Model not trained"
        
        info = {
            'base_learners': [name for name, _ in self.model.estimators_],
            'meta_learner': type(self.model.final_estimator_).__name__,
            'n_features': len(self.feature_names) if self.feature_names else None,
        }
        
        return info


def main():
    """Test stacked ensemble"""
    from data_acquisition import DataAcquisition
    from utils.preprocessing import DataPreprocessor, prepare_features_target
    from feature_engineering import FeatureEngineeringPipeline
    
    # Load and prepare data
    acquisition = DataAcquisition()
    df, _ = acquisition.prepare_dataset()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.remove_extreme_outliers(df)
    
    # Split data
    train_df, test_df = preprocessor.stratified_split(df)
    
    # Feature engineering
    X_train, y_train = prepare_features_target(train_df)
    X_test, y_test = prepare_features_target(test_df)
    
    fe_pipeline = FeatureEngineeringPipeline()
    X_train_transformed = fe_pipeline.fit_transform(X_train, y_train)
    X_test_transformed = fe_pipeline.transform(X_test)
    
    # Train stacked ensemble
    ensemble = StackedEnsembleModel()
    ensemble.train(X_train_transformed, y_train, fe_pipeline.get_feature_names())
    
    # Evaluate
    from utils.metrics import calculate_metrics, print_metrics
    
    y_train_pred = ensemble.predict(X_train_transformed)
    y_test_pred = ensemble.predict(X_test_transformed)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print_metrics(train_metrics, "Training Set")
    print_metrics(test_metrics, "Test Set")
    
    # Save model
    ensemble.save_model()


if __name__ == "__main__":
    main()
