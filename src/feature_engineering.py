"""
Advanced Feature Engineering Module
Implements interaction terms, target encoding, and Box-Cox transformations
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.special import boxcox1p
from typing import List, Dict, Tuple
import logging
from config import FEATURE_CONFIG, MODEL_CONFIG

logger = logging.getLogger(__name__)


class InteractionFeatureGenerator(BaseEstimator, TransformerMixin):
    """Generate interaction terms between specified features"""
    
    def __init__(self, interaction_pairs: List[Tuple[str, str]] = None):
        """
        Args:
            interaction_pairs: List of tuples specifying feature pairs to interact
        """
        # `or` would treat an empty list as unset, making it impossible to ask
        # for no interaction terms at all
        if interaction_pairs is None:
            interaction_pairs = FEATURE_CONFIG['interaction_terms']
        self.interaction_pairs = interaction_pairs
        
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

    Training rows are encoded out-of-fold: each row receives an encoding
    computed from the other folds only, so a row's own target never
    contributes to its own feature value. Unseen data is encoded with
    statistics fitted on the full training set.
    """

    def __init__(self, columns: List[str] = None, smoothing: float = 10.0,
                 n_splits: int = 5, random_state: int = None):
        """
        Args:
            columns: Columns to apply target encoding
            smoothing: Smoothing parameter (higher = more regularization)
            n_splits: Folds used to build out-of-fold training encodings
            random_state: Seed for fold assignment (defaults to MODEL_CONFIG)
        """
        # As above: an empty list means "encode nothing", not "use the default"
        if columns is None:
            columns = FEATURE_CONFIG['high_cardinality_cols']
        self.columns = columns
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state if random_state is not None else MODEL_CONFIG['random_state']
        self.encodings = {}
        self.global_mean = None

    def _smoothed_means(self, categories: pd.Series, y: pd.Series, global_mean: float) -> dict:
        """
        Compute smoothed per-category target means

        Args:
            categories: Categorical values aligned with y
            y: Target variable
            global_mean: Prior to shrink category means towards

        Returns:
            Mapping of category -> smoothed mean
        """
        grouped = y.groupby(categories)
        stats_df = pd.DataFrame({'target_mean': grouped.mean(), 'count': grouped.count()})

        # Weighted average of category mean and global mean
        # Formula: (count * category_mean + smoothing * global_mean) / (count + smoothing)
        smoothed = (
            (stats_df['count'] * stats_df['target_mean'] + self.smoothing * global_mean) /
            (stats_df['count'] + self.smoothing)
        )

        return smoothed.to_dict()

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
                # Full-data encoding, used when transforming unseen data
                self.encodings[col] = self._smoothed_means(X[col], y, self.global_mean)

                logger.info(f"Target encoding fitted for '{col}': {len(self.encodings[col])} categories")

        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit the encoder and return out-of-fold encodings for the training rows

        Using the full-data encoding on the training rows would let each row's
        own target leak into its own feature, which inflates the apparent
        importance of the encoded columns. Each fold is therefore encoded from
        the remaining folds only.

        Args:
            X: Feature DataFrame
            y: Target variable

        Returns:
            Transformed feature DataFrame
        """
        if y is None:
            raise ValueError("TargetEncoder.fit_transform requires y")

        self.fit(X, y)

        X = X.copy()
        y = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y

        n_splits = min(self.n_splits, len(X))
        if n_splits < 2:
            # Too few rows to hold anything out; fall back to the fitted encoding
            return self.transform(X)

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        for col in self.columns:
            if col not in X.columns or col not in self.encodings:
                continue

            oof = pd.Series(np.nan, index=X.index, dtype=float)

            for train_idx, valid_idx in kfold.split(X):
                fold_train = X.index[train_idx]
                fold_valid = X.index[valid_idx]

                fold_mean = y.loc[fold_train].mean()
                fold_encoding = self._smoothed_means(
                    X.loc[fold_train, col], y.loc[fold_train], fold_mean
                )

                # Categories absent from this fold's training part fall back to
                # that fold's own prior
                oof.loc[fold_valid] = X.loc[fold_valid, col].map(fold_encoding).fillna(fold_mean)

            X[f"{col}_encoded"] = oof
            X = X.drop(col, axis=1)

            logger.debug(f"Applied out-of-fold target encoding to '{col}'")

        return X

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
        # fit_transform returns out-of-fold values so training rows are not
        # encoded using their own target
        logger.info("Step 2: Applying target encoding...")
        X = self.target_encoder.fit_transform(X, y)
        
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
