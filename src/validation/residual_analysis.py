"""
Residual Analysis Module
Homoscedasticity testing and residual diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan
import logging
import sys
sys.path.append('..')
from config import OUTPUT_DIR, MODEL_CONFIG

logger = logging.getLogger(__name__)


class ResidualAnalyzer:
    """Analyze model residuals for diagnostic purposes"""
    
    def __init__(self, y_true, y_pred):
        """
        Initialize residual analyzer
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
        """
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.residuals = self.y_true - self.y_pred
        self.standardized_residuals = None
        
    def calculate_standardized_residuals(self):
        """Calculate standardized residuals"""
        spread = np.std(self.residuals)

        # Perfect predictions give a zero spread; dividing would yield nan/inf
        if spread == 0:
            logger.warning("Residuals have zero spread; standardized residuals are all zero")
            std_residuals = np.zeros_like(self.residuals, dtype=float)
        else:
            std_residuals = self.residuals / spread

        self.standardized_residuals = std_residuals
        return std_residuals
    
    def plot_residuals_vs_fitted(self, save_path=None):
        """
        Plot residuals vs fitted values
        
        Args:
            save_path: Path to save plot
        """
        logger.info("Creating residuals vs fitted values plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Scatter plot
        ax.scatter(self.y_pred, self.residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        
        # Add lowess smoothing line
        smoothed = lowess(self.residuals, self.y_pred, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'b-', linewidth=2, label='LOWESS')
        
        ax.set_xlabel('Fitted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'residuals_vs_fitted.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residuals vs fitted plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def plot_qq(self, save_path=None):
        """
        Q-Q plot for normality assessment
        
        Args:
            save_path: Path to save plot
        """
        logger.info("Creating Q-Q plot...")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        stats.probplot(self.residuals, dist="norm", plot=ax)
        
        ax.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'qq_plot.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Q-Q plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def plot_residual_distribution(self, save_path=None):
        """
        Plot residual distribution histogram
        
        Args:
            save_path: Path to save plot
        """
        logger.info("Creating residual distribution plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Histogram
        ax.hist(self.residuals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = np.mean(self.residuals), np.std(self.residuals)
        x = np.linspace(self.residuals.min(), self.residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
        
        ax.set_xlabel('Residuals', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'residual_distribution.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual distribution plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def plot_scale_location(self, save_path=None):
        """
        Scale-Location plot (sqrt of standardized residuals vs fitted values)
        
        Args:
            save_path: Path to save plot
        """
        logger.info("Creating scale-location plot...")
        
        if self.standardized_residuals is None:
            self.calculate_standardized_residuals()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sqrt_std_residuals = np.sqrt(np.abs(self.standardized_residuals))
        
        ax.scatter(self.y_pred, sqrt_std_residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        
        # Add lowess smoothing
        smoothed = lowess(sqrt_std_residuals, self.y_pred, frac=0.3)
        ax.plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2, label='LOWESS')
        
        ax.set_xlabel('Fitted Values', fontsize=12)
        ax.set_ylabel('√|Standardized Residuals|', fontsize=12)
        ax.set_title('Scale-Location Plot', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'scale_location.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Scale-location plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def breusch_pagan_test(self, X):
        """
        Breusch-Pagan test for homoscedasticity
        
        Args:
            X: Feature matrix
            
        Returns:
            Test results dictionary
        """
        logger.info("Performing Breusch-Pagan test for homoscedasticity...")

        # statsmodels requires the auxiliary regression's design matrix to
        # contain an intercept and raises ValueError without one. Passing the
        # raw feature matrix only worked when it happened to carry a constant
        # column. add_constant defaults to has_constant='skip', so this is a
        # no-op when one is already present.
        exog = sm.add_constant(np.asarray(X, dtype=float))

        bp_test = het_breuschpagan(self.residuals, exog)
        
        results = {
            'lm_statistic': bp_test[0],
            'lm_pvalue': bp_test[1],
            'f_statistic': bp_test[2],
            'f_pvalue': bp_test[3],
        }
        
        logger.info("Breusch-Pagan Test Results:")
        logger.info(f"  LM Statistic: {results['lm_statistic']:.4f}")
        logger.info(f"  LM p-value: {results['lm_pvalue']:.4f}")
        logger.info(f"  F Statistic: {results['f_statistic']:.4f}")
        logger.info(f"  F p-value: {results['f_pvalue']:.4f}")
        
        if results['lm_pvalue'] > 0.05:
            logger.info("  ✓ Homoscedasticity assumption satisfied (p > 0.05)")
        else:
            logger.warning("  ⚠ Heteroscedasticity detected (p < 0.05)")
        
        return results
    
    def shapiro_wilk_test(self, random_state: int = None):
        """
        Shapiro-Wilk test for normality of residuals

        Args:
            random_state: Seed for the subsample drawn when there are more than
                5000 residuals (defaults to MODEL_CONFIG)

        Returns:
            Test results dictionary
        """
        logger.info("Performing Shapiro-Wilk test for normality...")

        if random_state is None:
            random_state = MODEL_CONFIG['random_state']

        # Use sample if dataset is too large (Shapiro-Wilk has limitations).
        # The draw is seeded so repeated runs report the same p-value.
        if len(self.residuals) > 5000:
            rng = np.random.default_rng(random_state)
            sample_residuals = rng.choice(self.residuals, 5000, replace=False)
        else:
            sample_residuals = self.residuals
        
        statistic, pvalue = stats.shapiro(sample_residuals)
        
        results = {
            'statistic': statistic,
            'pvalue': pvalue,
        }
        
        logger.info("Shapiro-Wilk Test Results:")
        logger.info(f"  Statistic: {results['statistic']:.4f}")
        logger.info(f"  p-value: {results['pvalue']:.4f}")
        
        if results['pvalue'] > 0.05:
            logger.info("  ✓ Residuals appear normally distributed (p > 0.05)")
        else:
            logger.warning("  ⚠ Residuals may not be normally distributed (p < 0.05)")
        
        return results
    
    def generate_full_report(self, X=None, save_dir=None):
        """
        Generate comprehensive residual analysis report
        
        Args:
            X: Feature matrix (for Breusch-Pagan test)
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with test results and plot paths
        """
        save_dir = save_dir or OUTPUT_DIR
        
        logger.info("=" * 70)
        logger.info("GENERATING COMPREHENSIVE RESIDUAL ANALYSIS REPORT")
        logger.info("=" * 70)
        
        # Generate plots
        plots = {
            'residuals_vs_fitted': self.plot_residuals_vs_fitted(save_dir / 'residuals_vs_fitted.png'),
            'qq_plot': self.plot_qq(save_dir / 'qq_plot.png'),
            'distribution': self.plot_residual_distribution(save_dir / 'residual_distribution.png'),
            'scale_location': self.plot_scale_location(save_dir / 'scale_location.png'),
        }
        
        # Statistical tests
        tests = {
            'shapiro_wilk': self.shapiro_wilk_test(),
        }
        
        if X is not None:
            tests['breusch_pagan'] = self.breusch_pagan_test(X)
        
        # Summary statistics
        summary = {
            'mean_residual': np.mean(self.residuals),
            'std_residual': np.std(self.residuals),
            'min_residual': np.min(self.residuals),
            'max_residual': np.max(self.residuals),
            'skewness': stats.skew(self.residuals),
            'kurtosis': stats.kurtosis(self.residuals),
        }
        
        logger.info("\nResidual Summary Statistics:")
        for key, value in summary.items():
            logger.info(f"  {key}: {value:.4f}")
        
        logger.info("\nResidual analysis report generation complete")
        
        return {
            'plots': plots,
            'tests': tests,
            'summary': summary,
        }


def main():
    """Test residual analysis"""
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
    
    # Predictions
    y_test_pred = ensemble.predict(X_test_transformed)
    
    # Residual analysis
    analyzer = ResidualAnalyzer(y_test, y_test_pred)
    report = analyzer.generate_full_report(X_test_transformed)
    
    print("\n" + "=" * 70)
    print("RESIDUAL ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
