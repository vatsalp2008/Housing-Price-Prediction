"""
Permutation Importance Analysis
Feature importance through permutation testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import logging
import sys
sys.path.append('..')
from config import OUTPUT_DIR, MODEL_CONFIG

logger = logging.getLogger(__name__)


class PermutationImportanceAnalyzer:
    """Analyze feature importance using permutation testing"""
    
    def __init__(self, model, X, y, feature_names=None):
        """
        Initialize permutation importance analyzer
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            feature_names: List of feature names
        """
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.importance_results = None
        
    def calculate_importance(self, n_repeats=10, random_state=None):
        """
        Calculate permutation importance
        
        Args:
            n_repeats: Number of times to permute each feature
            random_state: Random seed
            
        Returns:
            Permutation importance results
        """
        # `or` would treat an explicit random_state=0 as unset
        if random_state is None:
            random_state = MODEL_CONFIG['random_state']
        
        logger.info("=" * 70)
        logger.info("CALCULATING PERMUTATION IMPORTANCE")
        logger.info("=" * 70)
        logger.info(f"Number of repeats: {n_repeats}")
        logger.info(f"Samples: {len(self.X)}")
        
        # Calculate permutation importance
        self.importance_results = permutation_importance(
            self.model,
            self.X,
            self.y,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=MODEL_CONFIG['n_jobs'],
            scoring='neg_mean_squared_error'
        )
        
        logger.info("Permutation importance calculation complete")
        
        return self.importance_results
    
    def get_importance_dataframe(self):
        """
        Get permutation importance as DataFrame
        
        Returns:
            DataFrame with feature importance
        """
        if self.importance_results is None:
            raise ValueError("Importance not calculated. Call calculate_importance first.")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else range(len(self.importance_results.importances_mean)),
            'importance_mean': self.importance_results.importances_mean,
            'importance_std': self.importance_results.importances_std,
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        return importance_df
    
    def plot_importance(self, top_n=20, save_path=None):
        """
        Plot permutation importance
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save plot
        """
        if self.importance_results is None:
            raise ValueError("Importance not calculated. Call calculate_importance first.")
        
        logger.info(f"Creating permutation importance plot (top {top_n} features)...")
        
        # Get importance DataFrame
        importance_df = self.get_importance_dataframe()
        
        # Select top N features
        top_features = importance_df.head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        
        # Horizontal bar plot with error bars
        y_pos = np.arange(len(top_features))
        ax.barh(y_pos, top_features['importance_mean'], 
                xerr=top_features['importance_std'],
                align='center', alpha=0.7, color='steelblue', 
                edgecolor='black', linewidth=1.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features['feature'])
        ax.invert_yaxis()  # Top feature at the top
        ax.set_xlabel('Permutation Importance (Decrease in -MSE)', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Permutation Importance', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'permutation_importance.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Permutation importance plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    @staticmethod
    def _normalize(values: pd.Series) -> pd.Series:
        """
        Scale a series to [0, 1] for cross-method comparison

        Args:
            values: Importance scores

        Returns:
            Scaled scores, all zero when every score is identical
        """
        spread = values.max() - values.min()

        # A flat series would divide by zero and yield all-NaN bars
        if spread == 0:
            return pd.Series(0.0, index=values.index)

        return (values - values.min()) / spread

    def compare_with_shap(self, shap_importance_df, top_n=15, save_path=None):
        """
        Compare permutation importance with SHAP importance
        
        Args:
            shap_importance_df: DataFrame with SHAP importance
            top_n: Number of features to compare
            save_path: Path to save plot
        """
        if self.importance_results is None:
            raise ValueError("Importance not calculated. Call calculate_importance first.")
        
        logger.info("Creating comparison plot: Permutation vs SHAP importance...")
        
        # Get permutation importance
        perm_df = self.get_importance_dataframe()
        
        # Normalize both importance scores to [0, 1] for comparison
        shap_df = shap_importance_df.copy()
        perm_df['importance_normalized'] = self._normalize(perm_df['importance_mean'])
        shap_df['importance_normalized'] = self._normalize(shap_df['importance'])
        
        # Get top features from both methods
        top_perm_features = set(perm_df.head(top_n)['feature'])
        top_shap_features = set(shap_df.head(top_n)['feature'])
        all_top_features = list(top_perm_features.union(top_shap_features))
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({'feature': all_top_features})
        comparison = comparison.merge(
            perm_df[['feature', 'importance_normalized']].rename(columns={'importance_normalized': 'permutation'}),
            on='feature',
            how='left'
        )
        comparison = comparison.merge(
            shap_df[['feature', 'importance_normalized']].rename(columns={'importance_normalized': 'shap'}),
            on='feature',
            how='left'
        )
        comparison = comparison.fillna(0)
        
        # Sort by average importance
        comparison['avg_importance'] = (comparison['permutation'] + comparison['shap']) / 2
        comparison = comparison.sort_values('avg_importance', ascending=False).head(top_n)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
        
        y_pos = np.arange(len(comparison))
        width = 0.35
        
        ax.barh(y_pos - width/2, comparison['permutation'], width, 
                label='Permutation', alpha=0.8, color='steelblue', edgecolor='black')
        ax.barh(y_pos + width/2, comparison['shap'], width,
                label='SHAP', alpha=0.8, color='coral', edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(comparison['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Normalized Importance', fontsize=12)
        ax.set_title('Feature Importance: Permutation vs SHAP', 
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'importance_comparison.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")
        
        plt.close()
        
        return save_path, comparison
    
    def detect_overfitting(self, threshold=0.01):
        """
        Detect potential overfitting by identifying features with very low importance
        
        Args:
            threshold: Importance threshold below which features are considered noise
            
        Returns:
            List of potentially noisy features
        """
        if self.importance_results is None:
            raise ValueError("Importance not calculated. Call calculate_importance first.")
        
        importance_df = self.get_importance_dataframe()
        
        # Features with importance close to zero or negative
        noisy_features = importance_df[importance_df['importance_mean'] <= threshold]
        
        if len(noisy_features) > 0:
            logger.warning(f"Found {len(noisy_features)} features with low importance (≤ {threshold})")
            logger.warning("These features may be contributing to overfitting:")
            for _, row in noisy_features.head(10).iterrows():
                logger.warning(f"  {row['feature']}: {row['importance_mean']:.6f}")
        else:
            logger.info(f"No features with importance ≤ {threshold} detected")
        
        return noisy_features
    
    def generate_full_report(self, shap_importance_df=None, n_repeats=10, top_n=20, save_dir=None):
        """
        Generate comprehensive permutation importance report
        
        Args:
            shap_importance_df: SHAP importance for comparison (optional)
            n_repeats: Number of permutation repeats
            top_n: Number of top features to display
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with results and plot paths
        """
        save_dir = save_dir or OUTPUT_DIR
        
        logger.info("=" * 70)
        logger.info("GENERATING PERMUTATION IMPORTANCE REPORT")
        logger.info("=" * 70)
        
        # Calculate importance
        self.calculate_importance(n_repeats=n_repeats)
        
        # Get importance DataFrame
        importance_df = self.get_importance_dataframe()
        
        # Generate plots
        plots = {}
        plots['importance'] = self.plot_importance(top_n=top_n, save_path=save_dir / 'permutation_importance.png')
        
        # Compare with SHAP if provided
        comparison_df = None
        if shap_importance_df is not None:
            comp_plot, comparison_df = self.compare_with_shap(
                shap_importance_df, 
                top_n=top_n, 
                save_path=save_dir / 'importance_comparison.png'
            )
            plots['comparison'] = comp_plot
        
        # Detect overfitting
        noisy_features = self.detect_overfitting()
        
        logger.info("Permutation importance report generation complete")
        
        return {
            'importance_df': importance_df,
            'comparison_df': comparison_df,
            'noisy_features': noisy_features,
            'plots': plots,
        }


def main():
    """Test permutation importance analysis"""
    from data_acquisition import DataAcquisition
    from utils.preprocessing import DataPreprocessor, prepare_features_target
    from feature_engineering import FeatureEngineeringPipeline
    from models.stacked_ensemble import StackedEnsembleModel
    
    # Load and prepare data
    acquisition = DataAcquisition()
    df, _ = acquisition.prepare_dataset()
    
    preprocessor = DataPreprocessor()

    # Split first so imputation is fit on training data only
    train_df, test_df = preprocessor.stratified_split(df)
    train_df = preprocessor.handle_missing_values(train_df, fit=True)
    test_df = preprocessor.handle_missing_values(test_df, fit=False)
    train_df = preprocessor.remove_extreme_outliers(train_df)
    
    X_train, y_train = prepare_features_target(train_df)
    X_test, y_test = prepare_features_target(test_df)
    
    fe_pipeline = FeatureEngineeringPipeline()
    X_train_transformed = fe_pipeline.fit_transform(X_train, y_train)
    X_test_transformed = fe_pipeline.transform(X_test)
    
    # Train model
    ensemble = StackedEnsembleModel()
    ensemble.train(X_train_transformed, y_train)
    
    # Permutation importance
    analyzer = PermutationImportanceAnalyzer(
        ensemble.model,
        X_test_transformed,
        y_test,
        feature_names=fe_pipeline.get_feature_names()
    )
    
    report = analyzer.generate_full_report(n_repeats=5, top_n=15)
    
    print("\n" + "=" * 70)
    print("TOP 10 FEATURES BY PERMUTATION IMPORTANCE")
    print("=" * 70)
    print(report['importance_df'].head(10))


if __name__ == "__main__":
    main()
