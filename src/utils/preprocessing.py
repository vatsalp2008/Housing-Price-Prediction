"""
Utility functions for data preprocessing
Handles missing values, outliers, and data splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import logging
import sys
sys.path.append('..')
from config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle missing values, outliers, and data preparation"""
    
    def __init__(self):
        self.numeric_impute_values = {}
        self.categorical_impute_values = {}
        
    def handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Handle missing values with intelligent imputation
        
        Args:
            df: Input DataFrame
            fit: Whether to fit imputation values (True for train, False for test)
            
        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Separate numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Handle numeric missing values
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if fit:
                    # Use median for numeric columns
                    self.numeric_impute_values[col] = df[col].median()
                
                if col in self.numeric_impute_values:
                    # Assign back rather than fillna(inplace=True): the latter
                    # mutates a temporary and is a silent no-op under
                    # copy-on-write (the pandas 3.0 default).
                    df[col] = df[col].fillna(self.numeric_impute_values[col])
        
        # Handle categorical missing values
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                if fit:
                    # Use mode for categorical columns, or 'Missing' if no mode
                    mode_values = df[col].mode()
                    self.categorical_impute_values[col] = mode_values[0] if len(mode_values) > 0 else 'Missing'
                
                if col in self.categorical_impute_values:
                    df[col] = df[col].fillna(self.categorical_impute_values[col])
        
        return df
    
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: List[str], 
                           threshold: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR method
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            threshold: IQR multiplier (1.5 is standard, 3.0 is more conservative)
            
        Returns:
            Boolean Series indicating outlier rows
        """
        outlier_mask = pd.Series([False] * len(df), index=df.index)
        
        for col in columns:
            # is_numeric_dtype covers int32/float32 and nullable dtypes, which an
            # explicit [np.float64, np.int64] check silently skipped
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_mask |= col_outliers
                
                if col_outliers.sum() > 0:
                    logger.info(f"Column '{col}': {col_outliers.sum()} outliers detected")
        
        return outlier_mask
    
    def remove_extreme_outliers(self, df: pd.DataFrame, target_col: str = 'SalePrice') -> pd.DataFrame:
        """
        Remove extreme outliers from the dataset
        
        Args:
            df: Input DataFrame
            target_col: Target variable column name
            
        Returns:
            DataFrame with extreme outliers removed
        """
        df = df.copy()
        initial_len = len(df)
        
        # Remove extreme outliers in target variable (beyond 3 IQR)
        outliers = self.detect_outliers_iqr(df, [target_col], threshold=3.0)
        df = df[~outliers]
        
        removed = initial_len - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} extreme outliers ({removed/initial_len*100:.2f}%)")
        
        return df
    
    def stratified_split(self, df: pd.DataFrame, target_col: str = 'SalePrice',
                        test_size: float = None, random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train-test split based on target variable bins

        Args:
            df: Input DataFrame
            target_col: Target variable column name
            test_size: Proportion of test set (defaults to MODEL_CONFIG)
            random_state: Random seed for reproducibility (defaults to MODEL_CONFIG)

        Returns:
            Tuple of (train_df, test_df)
        """
        df = df.copy()

        # Fall back to the central config so changing it actually takes effect
        if test_size is None:
            test_size = MODEL_CONFIG['test_size']
        if random_state is None:
            random_state = MODEL_CONFIG['random_state']
        
        # Create bins for stratification
        df['price_bin'] = pd.qcut(df[target_col], q=5, labels=False, duplicates='drop')
        
        # Stratified split
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=df['price_bin']
        )
        
        # Remove temporary bin column
        train_df = train_df.drop('price_bin', axis=1)
        test_df = test_df.drop('price_bin', axis=1)
        
        logger.info(f"Train set: {len(train_df)} samples")
        logger.info(f"Test set: {len(test_df)} samples")
        
        return train_df, test_df


def prepare_features_target(df: pd.DataFrame, target_col: str = 'SalePrice') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable
    
    Args:
        df: Input DataFrame
        target_col: Target variable column name
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    return X, y
