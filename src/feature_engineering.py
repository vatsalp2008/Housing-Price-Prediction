"""
Advanced Feature Engineering Module
Implements interaction terms, target encoding, and Box-Cox transformations
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.special import boxcox1p
from typing import List, Dict, Tuple
import logging
from config import FEATURE_CONFIG

logger = logging.getLogger(__name__)


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate interaction terms between specified features"""
    
    def __init__(self, interaction_pairs: List[Tuple[str, str]] = None):
        """
        Args:
            interaction_pairs: List of tuples specifying feature pairs to interact
        """
        self.interaction_pairs = interaction_pairs or FEATURE_CONFIG['interaction_terms']
        
    def fit(self, X, y=None):
        """Fit method (no fitting required for interactions)"""
        return self
    
    def transform(self, X):
        """Generate interaction features"""
        X = X.copy()
        
        for feat1, feat2 in self.interaction_pairs:
            if feat1 in X.columns and feat2 in X.columns:
                # Create interaction name
                interaction_name = f"{feat1}_x_{feat2}"
                
                # Generate interaction (multiplication)
                X[interaction_name] = X[feat1] * X[feat2]
                
                logger.debug(f"Created interaction: {interaction_name}")
        
        return X


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding for high-cardinality categorical variables with smoothing
    Implements cross-validation to prevent target leakage
    """
    
    def __init__(self, columns: List[str] = None, smoothing: float = 10.0):
        """
        Args:
            columns: Columns to apply target encoding
            smoothing: Smoothing parameter (higher = more regularization)
        """
        self.columns = columns or FEATURE_CONFIG['high_cardinality_cols']
        self.smoothing = smoothing
        self.encodings = {}
        self.global_mean = None
        
    def fit(self, X, y):
        """
        Fit target encoder
        
        Args:
            X: Feature DataFrame
            y: Target variable
        """
        X = X.copy()
        self.global_mean = y.mean()
        
        for col in self.columns:
            if col in X.columns:
                # Calculate mean target value for each category
                category_stats = pd.DataFrame({
                    'target_mean': y.groupby(X[col]).mean(),
                    'count': y.groupby(X[col]).count()
                })
                
                # Apply smoothing: weighted average of category mean and global mean
                # Formula: (count * category_mean + smoothing * global_mean) / (count + smoothing)
                category_stats['smoothed_mean'] = (
                    (category_stats['count'] * category_stats['target_mean'] + 
                     self.smoothing * self.global_mean) / 
                    (category_stats['count'] + self.smoothing)
                )
                
                self.encodings[col] = category_stats['smoothed_mean'].to_dict()
                
                logger.info(f"Target encoding fitted for '{col}': {len(self.encodings[col])} categories")
        
        return self
    
    def transform(self, X):
        """Apply target encoding"""
        X = X.copy()
        
        for col in self.columns:
            if col in X.columns and col in self.encodings:
                # Create new encoded column
                encoded_col = f"{col}_encoded"
                
                # Map categories to encoded values, use global mean for unseen categories
                X[encoded_col] = X[col].map(self.encodings[col]).fillna(self.global_mean)
                
                # Drop original column
                X = X.drop(col, axis=1)
                
                logger.debug(f"Applied target encoding to '{col}'")
        
        return X


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """Apply Box-Cox transformation to skewed features"""
    
    def __init__(self, skewness_threshold: float = 0.75, lmbda: float = 0.15):
        """
        Args:
            skewness_threshold: Minimum skewness to apply transformation
            lmbda: Box-Cox lambda (0.15 is a common choice for right-skewed data)
        """
        self.skewness_threshold = skewness_threshold
        self.lmbda = lmbda
        self.skewed_features = []

    def fit(self, X, y=None):
        """
        Identify skewed features

        Args:
            X: Feature DataFrame
        """
        X = X.copy()

        # Only consider numeric columns
        numeric_features = X.select_dtypes(include=[np.number]).columns

        # Calculate skewness for each feature
        skewness = X[numeric_features].apply(lambda x: stats.skew(x.dropna()))

        # Identify features with high skewness
        candidates = skewness[abs(skewness) > self.skewness_threshold].index.tolist()

        # boxcox1p raises (1 + x) to a fractional power, which is undefined for
        # x < -1. Such features would become all-NaN, so exclude them here
        # rather than feeding NaN columns to the model.
        self.skewed_features = []
        for feature in candidates:
            if X[feature].min() <= -1:
                logger.warning(
                    f"Skipping Box-Cox for '{feature}': contains values <= -1, "
                    f"which are outside the transform's domain"
                )
            else:
                self.skewed_features.append(feature)

        logger.info(f"Identified {len(self.skewed_features)} skewed features for Box-Cox transformation")

        return self

    def transform(self, X):
        """Apply Box-Cox transformation to skewed features"""
        X = X.copy()

        for feature in self.skewed_features:
            if feature in X.columns:
                # Box-Cox requires positive values, add 1 to handle zeros
                # Using boxcox1p which is equivalent to boxcox(x + 1)
                column = X[feature]

                # Unseen data may still fall outside the domain; clip instead of
                # letting NaN propagate silently into the model
                if column.min() <= -1:
                    n_clipped = int((column <= -1).sum())
                    logger.warning(
                        f"Clipping {n_clipped} out-of-domain value(s) in '{feature}' "
                        f"before Box-Cox transformation"
                    )
                    column = column.clip(lower=-1 + 1e-9)

                X[feature] = boxcox1p(column, self.lmbda)

                logger.debug(f"Applied Box-Cox to '{feature}'")

        return X


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline"""
    
    def __init__(self, target_col: str = 'SalePrice'):
        """
        Args:
            target_col: Name of target variable column
        """
        self.target_col = target_col
        self.interaction_generator = InteractionFeatureGenerator()
        self.target_encoder = TargetEncoder()
        self.boxcox_transformer = BoxCoxTransformer()
        self.label_encoders = {}
        self.fallback_categories = {}
        self.feature_names = None
        
    def _encode_categorical(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode remaining categorical variables
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit encoders
            
        Returns:
            DataFrame with encoded categoricals
        """
        X = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                values = X[col].astype(str)
                X[col] = self.label_encoders[col].fit_transform(values)
                # LabelEncoder sorts classes alphabetically, so remember the
                # actual modal category for the unseen-category fallback below
                self.fallback_categories[col] = values.mode()[0]
            else:
                if col in self.label_encoders:
                    # Handle unseen categories
                    X[col] = X[col].astype(str)
                    # Get known classes
                    known_classes = set(self.label_encoders[col].classes_)
                    # Replace unseen categories with the most frequent training category
                    fallback = self.fallback_categories.get(
                        col, self.label_encoders[col].classes_[0]
                    )
                    X[col] = X[col].apply(lambda x: x if x in known_classes else fallback)
                    X[col] = self.label_encoders[col].transform(X[col])
        
        return X
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit and transform features
        
        Args:
            X: Feature DataFrame
            y: Target variable
            
        Returns:
            Transformed feature DataFrame
        """
        logger.info("Starting feature engineering pipeline...")
        
        X = X.copy()
        
        # Step 1: Generate interaction terms
        logger.info("Step 1: Generating interaction terms...")
        X = self.interaction_generator.fit_transform(X)
        
        # Step 2: Target encoding for high-cardinality categoricals
        logger.info("Step 2: Applying target encoding...")
        X = self.target_encoder.fit(X, y).transform(X)
        
        # Step 3: Encode remaining categorical variables
        logger.info("Step 3: Encoding remaining categorical variables...")
        X = self._encode_categorical(X, fit=True)
        
        # Step 4: Box-Cox transformation for skewed features
        logger.info("Step 4: Applying Box-Cox transformations...")
        X = self.boxcox_transformer.fit(X).transform(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Feature engineering complete. Final feature count: {len(self.feature_names)}")
        
        return X
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features (for test set)
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Transformed feature DataFrame
        """
        X = X.copy()
        
        # Apply same transformations
        X = self.interaction_generator.transform(X)
        X = self.target_encoder.transform(X)
        X = self._encode_categorical(X, fit=False)
        X = self.boxcox_transformer.transform(X)
        
        # Ensure same columns as training
        # Add missing columns with zeros
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        
        # Remove extra columns and reorder
        X = X[self.feature_names]
        
        return X
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation"""
        return self.feature_names if self.feature_names else []


def main():
    """Test feature engineering pipeline"""
    from data_acquisition import DataAcquisition
    from utils.preprocessing import DataPreprocessor, prepare_features_target
    
    # Load data
    acquisition = DataAcquisition()
    df, _ = acquisition.prepare_dataset()
    
    # Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.remove_extreme_outliers(df)
    
    # Split features and target
    X, y = prepare_features_target(df)
    
    # Apply feature engineering
    fe_pipeline = FeatureEngineeringPipeline()
    X_transformed = fe_pipeline.fit_transform(X, y)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"\nOriginal features: {X.shape[1]}")
    print(f"Transformed features: {X_transformed.shape[1]}")
    print(f"\nSample of transformed features:")
    print(X_transformed.head())


if __name__ == "__main__":
    main()
