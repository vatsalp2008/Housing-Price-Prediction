"""
SHAP (SHapley Additive exPlanations) Analysis
Global feature importance and model interpretation
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
sys.path.append('..')
from config import SHAP_CONFIG, OUTPUT_DIR, MODEL_CONFIG

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """SHAP-based model explainability"""
    
    def __init__(self, model, X_background=None, feature_names=None):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained model
            X_background: Background dataset for SHAP (subset of training data)
            feature_names: List of feature names
        """
        self.model = model
        self.X_background = X_background
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        # The rows the stored shap_values actually correspond to. Plots must use
        # this rather than the full dataset when sampling has taken place.
        self.X_explained = None
        
    def create_explainer(self, X_background=None):
        """
        Create SHAP explainer
        
        Args:
            X_background: Background dataset (optional, uses stored if not provided)
        """
        if X_background is not None:
            self.X_background = X_background
        
        logger.info("Creating SHAP TreeExplainer...")
        
        # For tree-based models, use TreeExplainer (faster and exact)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("TreeExplainer created successfully")
        except Exception as e:
            logger.warning(f"TreeExplainer failed: {e}. Falling back to KernelExplainer...")
            # Fallback to KernelExplainer for non-tree models
            if self.X_background is None:
                raise ValueError("X_background required for KernelExplainer")
            self.explainer = shap.KernelExplainer(self.model.predict, self.X_background)
            logger.info("KernelExplainer created successfully")
        
        return self.explainer
    
    def calculate_shap_values(self, X, max_samples=None):
        """
        Calculate SHAP values for dataset
        
        Args:
            X: Dataset to explain
            max_samples: Maximum number of samples to calculate (for speed)
            
        Returns:
            SHAP values
        """
        if self.explainer is None:
            self.create_explainer()
        
        # Limit samples for computational efficiency
        if max_samples and len(X) > max_samples:
            logger.info(f"Using {max_samples} samples for SHAP calculation")
            X_sample = (
                X.sample(n=max_samples, random_state=MODEL_CONFIG['random_state'])
                if isinstance(X, pd.DataFrame) else X[:max_samples]
            )
        else:
            X_sample = X
        
        logger.info(f"Calculating SHAP values for {len(X_sample)} samples...")
        self.shap_values = self.explainer.shap_values(X_sample)
        self.X_explained = X_sample

        logger.info("SHAP values calculated successfully")

        return self.shap_values
    
    def _resolve_plot_data(self, X, shap_values):
        """
        Pick the feature rows that line up with the given SHAP values

        Args:
            X: Explicitly supplied dataset, or None
            shap_values: SHAP values about to be plotted

        Returns:
            The dataset to plot against

        Raises:
            ValueError: if the row counts do not match
        """
        data = X if X is not None else self.X_explained
        if data is None:
            data = self.X_background

        if data is not None and len(data) != len(shap_values):
            # calculate_shap_values may have sampled; plotting the full dataset
            # against sampled values trips an assertion deep inside shap
            if self.X_explained is not None and len(self.X_explained) == len(shap_values):
                logger.warning(
                    f"Supplied dataset has {len(data)} rows but SHAP values cover "
                    f"{len(shap_values)}; using the explained subset instead"
                )
                return self.X_explained

            raise ValueError(
                f"SHAP values cover {len(shap_values)} rows but the dataset has "
                f"{len(data)}"
            )

        return data

    def plot_summary(self, X=None, shap_values=None, max_display=None, save_path=None):
        """
        Create SHAP summary plot (global feature importance)
        
        Args:
            X: Dataset (optional, uses stored if not provided)
            shap_values: SHAP values (optional, calculates if not provided)
            max_display: Number of top features to display
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                if X is None:
                    raise ValueError("Must provide X or calculate SHAP values first")
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        if max_display is None:
            max_display = SHAP_CONFIG['max_display']
        
        logger.info("Creating SHAP summary plot...")
        
        plot_data = self._resolve_plot_data(X, shap_values)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            plot_data,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'shap_summary.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def plot_bar(self, X=None, shap_values=None, max_display=None, save_path=None):
        """
        Create SHAP bar plot (mean absolute SHAP values)
        
        Args:
            X: Dataset
            shap_values: SHAP values
            max_display: Number of features to display
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                if X is None:
                    raise ValueError("Must provide X or calculate SHAP values first")
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        if max_display is None:
            max_display = SHAP_CONFIG['max_display']
        
        logger.info("Creating SHAP bar plot...")
        
        plot_data = self._resolve_plot_data(X, shap_values)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            plot_data,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'shap_bar.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP bar plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def plot_dependence(self, feature_idx, X=None, shap_values=None, save_path=None):
        """
        Create SHAP dependence plot for a specific feature
        
        Args:
            feature_idx: Feature index or name
            X: Dataset
            shap_values: SHAP values
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                if X is None:
                    raise ValueError("Must provide X or calculate SHAP values first")
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        logger.info(f"Creating SHAP dependence plot for feature: {feature_idx}")
        
        plot_data = self._resolve_plot_data(X, shap_values)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx,
            shap_values,
            plot_data,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        
        if save_path is None:
            feature_name = feature_idx if isinstance(feature_idx, str) else self.feature_names[feature_idx]
            save_path = OUTPUT_DIR / f'shap_dependence_{feature_name}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP dependence plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def plot_force(self, sample_idx, X=None, shap_values=None, save_path=None):
        """
        Create SHAP force plot for a single prediction
        
        Args:
            sample_idx: Index of sample to explain
            X: Dataset
            shap_values: SHAP values
            save_path: Path to save plot
        """
        if shap_values is None:
            if self.shap_values is None:
                if X is None:
                    raise ValueError("Must provide X or calculate SHAP values first")
                self.calculate_shap_values(X)
            shap_values = self.shap_values
        
        logger.info(f"Creating SHAP force plot for sample {sample_idx}")
        
        # Get expected value (base value). A force plot needs the model's
        # baseline output; the mean of the SHAP values is a different quantity
        # entirely (it is near zero by construction) and would shift every bar.
        expected_value = getattr(self.explainer, 'expected_value', None)

        if expected_value is None:
            if self.X_background is None:
                raise ValueError(
                    "Explainer exposes no expected_value and no X_background is "
                    "available to estimate the model's baseline output"
                )
            expected_value = float(np.mean(self.model.predict(self.X_background)))
            logger.warning(
                "Explainer has no expected_value; using the mean prediction over "
                "the background set as the base value"
            )
        
        # Resolve the row to explain from whichever dataset is available
        plot_data = self._resolve_plot_data(X, shap_values)
        if plot_data is None:
            raise ValueError("No dataset available to read the sample's features from")

        instance = (
            plot_data.iloc[sample_idx] if isinstance(plot_data, pd.DataFrame)
            else plot_data[sample_idx]
        )

        # Create force plot
        force_plot = shap.force_plot(
            expected_value,
            shap_values[sample_idx],
            instance,
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        if save_path is None:
            save_path = OUTPUT_DIR / f'shap_force_sample_{sample_idx}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP force plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def get_feature_importance(self, shap_values=None):
        """
        Get feature importance based on mean absolute SHAP values
        
        Args:
            shap_values: SHAP values (optional)
            
        Returns:
            DataFrame with feature importance
        """
        if shap_values is None:
            shap_values = self.shap_values
        
        if shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values first.")
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else range(len(importance)),
            'importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        return importance_df
    
    def generate_full_report(self, X, save_dir=None):
        """
        Generate comprehensive SHAP analysis report
        
        Args:
            X: Dataset to analyze
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with paths to generated plots
        """
        save_dir = save_dir or OUTPUT_DIR
        
        logger.info("=" * 70)
        logger.info("GENERATING COMPREHENSIVE SHAP REPORT")
        logger.info("=" * 70)
        
        # Calculate SHAP values
        self.calculate_shap_values(X, max_samples=SHAP_CONFIG['sample_size'])
        
        # Generate plots
        plots = {}
        
        # Summary plot
        plots['summary'] = self.plot_summary(X, save_path=save_dir / 'shap_summary.png')
        
        # Bar plot
        plots['bar'] = self.plot_bar(X, save_path=save_dir / 'shap_bar.png')
        
        # Get top features and create dependence plots
        importance_df = self.get_feature_importance()
        top_features = importance_df.head(3)['feature'].tolist()
        
        plots['dependence'] = []
        for feature in top_features:
            dep_plot = self.plot_dependence(feature, X, save_path=save_dir / f'shap_dependence_{feature}.png')
            plots['dependence'].append(dep_plot)
        
        logger.info("SHAP report generation complete")
        
        return plots, importance_df


def main():
    """Test SHAP analysis"""
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
    ensemble.train(X_train_transformed, y_train, fe_pipeline.get_feature_names())
    
    # SHAP analysis
    analyzer = SHAPAnalyzer(
        ensemble.model,
        feature_names=fe_pipeline.get_feature_names()
    )
    
    plots, importance = analyzer.generate_full_report(X_test_transformed)
    
    print("\n" + "=" * 70)
    print("TOP 10 FEATURES BY SHAP IMPORTANCE")
    print("=" * 70)
    print(importance.head(10))


if __name__ == "__main__":
    main()
