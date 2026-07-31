"""
Data Acquisition Module
Downloads and prepares the Ames Housing Dataset with macroeconomic adjustments
"""

import pandas as pd
import numpy as np
import requests
from pathlib import Path
import logging
from typing import Tuple, Dict
from config import (
    DATA_DIR, 
    AMES_DATASET_URL, 
    AMES_DATASET_BACKUP_URL,
    MACRO_INDICATORS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAcquisition:
    """Handle data download, loading, and initial preprocessing"""
    
    def __init__(self):
        self.data_path = DATA_DIR / "AmesHousing.txt"
        self.processed_path = DATA_DIR / "ames_processed.csv"
        
    def download_dataset(self, force_download: bool = False) -> Path:
        """
        Download Ames Housing Dataset if not already present
        
        Args:
            force_download: Force re-download even if file exists
            
        Returns:
            Path to downloaded dataset
        """
        if self.data_path.exists() and not force_download:
            logger.info(f"Dataset already exists at {self.data_path}")
            return self.data_path
        
        logger.info("Downloading Ames Housing Dataset...")
        
        # Try primary URL first
        try:
            response = requests.get(AMES_DATASET_URL, timeout=30)
            response.raise_for_status()
            
            with open(self.data_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Dataset downloaded successfully to {self.data_path}")
            return self.data_path
            
        except Exception as e:
            logger.warning(f"Primary URL failed: {e}. Trying backup URL...")
            
            # Try backup URL
            try:
                response = requests.get(AMES_DATASET_BACKUP_URL, timeout=30)
                response.raise_for_status()
                
                with open(self.data_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Dataset downloaded from backup URL to {self.data_path}")
                return self.data_path
                
            except Exception as e2:
                logger.error(f"Both download attempts failed: {e2}")
                raise RuntimeError(
                    "Could not download dataset. Please download manually from "
                    "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
                )
    
    def load_dataset(self) -> pd.DataFrame:
        """
        Load the Ames Housing Dataset
        
        Returns:
            DataFrame with raw housing data
        """
        if not self.data_path.exists():
            logger.info("Dataset not found locally. Downloading...")
            self.download_dataset()
        
        logger.info(f"Loading dataset from {self.data_path}")
        
        # The Ames dataset is tab-delimited
        df = pd.read_csv(self.data_path, sep='\t')
        
        logger.info(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def add_macroeconomic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add macroeconomic indicators as features for market adjustment
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with macroeconomic features added
        """
        logger.info("Adding macroeconomic features...")
        
        df = df.copy()
        
        # Add current macroeconomic indicators
        df['Mortgage_Rate_30yr'] = MACRO_INDICATORS['mortgage_rate_30yr']
        df['CPI'] = MACRO_INDICATORS['cpi']
        
        # Calculate years since sale (for market adjustment)
        if 'Yr Sold' not in df.columns:
            raise KeyError(
                "Column 'Yr Sold' is required to derive Years_Since_Sale. "
                f"Available columns: {sorted(df.columns)[:10]}..."
            )
        df['Years_Since_Sale'] = MACRO_INDICATORS['reference_year'] - df['Yr Sold']
        
        # Create market adjustment factor
        # Simplified model: adjust for inflation and mortgage rate changes
        # This is a placeholder - in production, you'd use actual historical data
        base_mortgage_rate = 4.0  # Approximate rate during dataset collection
        base_cpi = 230.0  # Approximate CPI during dataset collection
        
        # Inflation adjustment factor
        inflation_factor = MACRO_INDICATORS['cpi'] / base_cpi
        
        # Mortgage rate impact (inverse relationship with prices)
        mortgage_impact = base_mortgage_rate / MACRO_INDICATORS['mortgage_rate_30yr']
        
        # Combined market adjustment factor
        df['Market_Adjustment_Factor'] = inflation_factor * mortgage_impact
        
        logger.info(f"Market adjustment factor range: "
                   f"{df['Market_Adjustment_Factor'].min():.3f} - "
                   f"{df['Market_Adjustment_Factor'].max():.3f}")
        
        return df
    
    def initial_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform initial data cleaning
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Performing initial data cleaning...")
        
        df = df.copy()
        
        # Remove Order column (just an index)
        if 'Order' in df.columns:
            df = df.drop('Order', axis=1)
        
        # Remove PID (parcel identification number - not useful for prediction)
        if 'PID' in df.columns:
            df = df.drop('PID', axis=1)
        
        # Log initial missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            logger.info(f"Columns with missing values: {len(missing_cols)}")
            logger.info(f"Total missing values: {missing_counts.sum()}")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate summary statistics for the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'n_rows': len(df),
            'n_columns': len(df.columns),
            'n_numeric': len(df.select_dtypes(include=[np.number]).columns),
            'n_categorical': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum(),
            'target_mean': df['SalePrice'].mean() if 'SalePrice' in df.columns else None,
            'target_median': df['SalePrice'].median() if 'SalePrice' in df.columns else None,
            'target_std': df['SalePrice'].std() if 'SalePrice' in df.columns else None,
        }
        
        return summary
    
    def prepare_dataset(self, save: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete data preparation pipeline
        
        Args:
            save: Whether to save processed data to disk
            
        Returns:
            Tuple of (processed DataFrame, summary statistics)
        """
        # Load raw data
        df = self.load_dataset()
        
        # Initial cleaning
        df = self.initial_cleaning(df)
        
        # Add macroeconomic features
        df = self.add_macroeconomic_features(df)
        
        # Generate summary
        summary = self.get_data_summary(df)
        
        # Log summary
        logger.info("=" * 50)
        logger.info("Dataset Summary:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 50)
        
        # Save processed data
        if save:
            df.to_csv(self.processed_path, index=False)
            logger.info(f"Processed data saved to {self.processed_path}")
        
        return df, summary


def main():
    """Main function for testing data acquisition"""
    acquisition = DataAcquisition()
    df, summary = acquisition.prepare_dataset()
    
    print("\n" + "=" * 60)
    print("DATA ACQUISITION COMPLETE")
    print("=" * 60)
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nTarget variable (SalePrice) statistics:")
    print(df['SalePrice'].describe())


if __name__ == "__main__":
    main()
