"""
LIME (Local Interpretable Model-agnostic Explanations) Analysis
Individual prediction explanations
"""

import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import sys
sys.path.append('..')
from config import LIME_CONFIG, OUTPUT_DIR

logger = logging.getLogger(__name__)


class LIMEAnalyzer:
    """LIME-based local explanations for individual predictions"""
    
    def __init__(self, model, X_train, feature_names=None, class_names=None):
        """
        Initialize LIME analyzer
        
        Args:
            model: Trained model
            X_train: Training data for LIME explainer
            feature_names: List of feature names
            class_names: Class names (for classification, None for regression)
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.class_names = class_names
        self.explainer = None
        
    def create_explainer(self):
        """Create LIME tabular explainer"""
        logger.info("Creating LIME tabular explainer...")
        
        # Convert to numpy if DataFrame
        X_train_array = self.X_train.values if isinstance(self.X_train, pd.DataFrame) else self.X_train
        
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_array,
            feature_names=self.feature_names,
            class_names=self.class_names,
            mode='regression',  # Use 'classification' for classification tasks
            verbose=False
        )
        
        logger.info("LIME explainer created successfully")
        
        return self.explainer
    
    def explain_instance(self, instance, num_features=None):
        """
        Explain a single prediction
        
        Args:
            instance: Single instance to explain (1D array or Series)
            num_features: Number of features to include in explanation
            
        Returns:
            LIME explanation object
        """
        if self.explainer is None:
            self.create_explainer()
        
        num_features = num_features or LIME_CONFIG['num_features']
        
        # Convert to numpy array if needed
        if isinstance(instance, pd.Series):
            instance_array = instance.values
        elif isinstance(instance, pd.DataFrame):
            instance_array = instance.values[0]
        else:
            instance_array = instance
        
        logger.info(f"Generating LIME explanation for instance...")
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            instance_array,
            self.model.predict,
            num_features=num_features,
            num_samples=LIME_CONFIG['num_samples']
        )
        
        return explanation
    
    def plot_explanation(self, explanation, save_path=None):
        """
        Plot LIME explanation
        
        Args:
            explanation: LIME explanation object
            save_path: Path to save plot
        """
        logger.info("Creating LIME explanation plot...")
        
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        
        if save_path is None:
            save_path = OUTPUT_DIR / 'lime_explanation.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"LIME explanation plot saved to {save_path}")
        
        plt.close()
        
        return save_path
    
    def save_html_report(self, explanation, save_path=None):
        """
        Save LIME explanation as HTML
        
        Args:
            explanation: LIME explanation object
            save_path: Path to save HTML file
        """
        if save_path is None:
            save_path = OUTPUT_DIR / 'lime_report.html'
        
        html = explanation.as_html()
        
        with open(save_path, 'w') as f:
            f.write(html)
        
        logger.info(f"LIME HTML report saved to {save_path}")
        
        return save_path
    
    def explain_and_visualize(self, instance, actual_value=None, save_prefix='lime'):
        """
        Complete explanation workflow: explain, plot, and save HTML
        
        Args:
            instance: Instance to explain
            actual_value: Actual target value (optional, for comparison)
            save_prefix: Prefix for saved files
            
        Returns:
            Dictionary with explanation and file paths
        """
        # Generate explanation
        explanation = self.explain_instance(instance)
        
        # Get prediction
        if isinstance(instance, pd.Series):
            instance_array = instance.values.reshape(1, -1)
        elif isinstance(instance, pd.DataFrame):
            instance_array = instance.values
        else:
            instance_array = instance.reshape(1, -1)
        
        prediction = self.model.predict(instance_array)[0]
        
        # Log results
        logger.info("=" * 70)
        logger.info("LIME EXPLANATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Predicted value: ${prediction:,.2f}")
        if actual_value is not None:
            logger.info(f"Actual value: ${actual_value:,.2f}")
            logger.info(f"Prediction error: ${abs(prediction - actual_value):,.2f}")
        
        # Get feature contributions
        feature_weights = explanation.as_list()
        logger.info("\nTop feature contributions:")
        for feature, weight in feature_weights[:5]:
            logger.info(f"  {feature}: {weight:+.2f}")
        
        # Save visualizations
        plot_path = self.plot_explanation(explanation, OUTPUT_DIR / f'{save_prefix}_plot.png')
        html_path = self.save_html_report(explanation, OUTPUT_DIR / f'{save_prefix}_report.html')
        
        return {
            'explanation': explanation,
            'prediction': prediction,
            'actual': actual_value,
            'plot_path': plot_path,
            'html_path': html_path,
            'feature_weights': feature_weights
        }
    
    def explain_multiple(self, X_samples, y_samples=None, n_samples=5):
        """
        Explain multiple instances
        
        Args:
            X_samples: Multiple instances to explain
            y_samples: Actual values (optional)
            n_samples: Number of samples to explain
            
        Returns:
            List of explanation results
        """
        results = []
        
        # Limit number of samples
        n = min(n_samples, len(X_samples))
        
        logger.info(f"Generating LIME explanations for {n} samples...")
        
        for i in range(n):
            instance = X_samples.iloc[i] if isinstance(X_samples, pd.DataFrame) else X_samples[i]
            actual = y_samples.iloc[i] if y_samples is not None and isinstance(y_samples, pd.Series) else (y_samples[i] if y_samples is not None else None)
            
            result = self.explain_and_visualize(
                instance,
                actual_value=actual,
                save_prefix=f'lime_sample_{i}'
            )
            
            results.append(result)
        
        logger.info(f"Generated {len(results)} LIME explanations")
        
        return results
    
    def compare_predictions(self, instances, actual_values=None):
        """
        Compare LIME explanations for multiple instances
        
        Args:
            instances: Multiple instances to compare
            actual_values: Actual values (optional)
            
        Returns:
            Comparison summary
        """
        explanations = []
        predictions = []
        
        for i, instance in enumerate(instances):
            exp = self.explain_instance(instance)
            explanations.append(exp)
            
            if isinstance(instance, pd.Series):
                instance_array = instance.values.reshape(1, -1)
            else:
                instance_array = instance.reshape(1, -1)
            
            pred = self.model.predict(instance_array)[0]
            predictions.append(pred)
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Instance': range(len(instances)),
            'Prediction': predictions,
        })
        
        if actual_values is not None:
            comparison['Actual'] = actual_values
            comparison['Error'] = abs(comparison['Prediction'] - comparison['Actual'])
        
        return comparison, explanations


def main():
    """Test LIME analysis"""
    from data_acquisition import DataAcquisition
    from utils.preprocessing import DataPreprocessor, prepare_features_target
    from feature_engineering import FeatureEngineeringPipeline
    from models.stacked_ensemble import StackedEnsembleModel
    
    # Load and prepare data
    acquisition = DataAcquisition()
    df, _ = acquisition.prepare_dataset()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.remove_extreme_outliers(df)
    
    train_df, test_df = preprocessor.stratified_split(df)
    
    X_train, y_train = prepare_features_target(train_df)
    X_test, y_test = prepare_features_target(test_df)
    
    fe_pipeline = FeatureEngineeringPipeline()
    X_train_transformed = fe_pipeline.fit_transform(X_train, y_train)
    X_test_transformed = fe_pipeline.transform(X_test)
    
    # Train model
    ensemble = StackedEnsembleModel()
    ensemble.train(X_train_transformed, y_train, fe_pipeline.get_feature_names())
    
    # LIME analysis
    analyzer = LIMEAnalyzer(
        ensemble.model,
        X_train_transformed,
        feature_names=fe_pipeline.get_feature_names()
    )
    
    # Explain a few test samples
    results = analyzer.explain_multiple(X_test_transformed, y_test, n_samples=3)
    
    print("\n" + "=" * 70)
    print("LIME EXPLANATIONS GENERATED")
    print("=" * 70)
    for i, result in enumerate(results):
        print(f"\nSample {i}:")
        print(f"  Prediction: ${result['prediction']:,.2f}")
        if result['actual']:
            print(f"  Actual: ${result['actual']:,.2f}")
        print(f"  HTML report: {result['html_path']}")


if __name__ == "__main__":
    main()
