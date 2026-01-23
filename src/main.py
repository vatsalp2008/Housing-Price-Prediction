"""
Main Pipeline - End-to-End Housing Valuation Engine
Orchestrates the complete workflow from data acquisition to model deployment
"""

import argparse
import logging
import sys
from pathlib import Path
import joblib
import json

# Add src to path
sys.path.append(str(Path(__file__).parent))

from config import OUTPUT_DIR, MODELS_DIR
from data_acquisition import DataAcquisition
from utils.preprocessing import DataPreprocessor, prepare_features_target
from utils.metrics import PerformanceReport
from feature_engineering import FeatureEngineeringPipeline
from models.stacked_ensemble import StackedEnsembleModel
from models.hyperparameter_optimization import OptunaOptimizer
from explainability.shap_analysis import SHAPAnalyzer
from explainability.lime_analysis import LIMEAnalyzer
from validation.residual_analysis import ResidualAnalyzer
from validation.permutation_importance import PermutationImportanceAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class HousingValuationPipeline:
    """Complete end-to-end housing valuation pipeline"""
    
    def __init__(self, optimize_hyperparams=False, quick_mode=False):
        """
        Initialize pipeline
        
        Args:
            optimize_hyperparams: Whether to run hyperparameter optimization
            quick_mode: Use reduced settings for quick testing
        """
        self.optimize_hyperparams = optimize_hyperparams
        self.quick_mode = quick_mode
        
        # Components
        self.data_acquisition = DataAcquisition()
        self.preprocessor = DataPreprocessor()
        self.fe_pipeline = FeatureEngineeringPipeline()
        self.model = None
        self.best_params = None
        
        # Data
        self.df = None
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_train_transformed = None
        self.X_test_transformed = None
        
        # Results
        self.train_predictions = None
        self.test_predictions = None
        self.performance_report = None
        
    def run_data_acquisition(self):
        """Step 1: Acquire and prepare data"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: DATA ACQUISITION")
        logger.info("=" * 80)
        
        self.df, summary = self.data_acquisition.prepare_dataset()
        
        logger.info("Data acquisition complete")
        return self.df
    
    def run_preprocessing(self):
        """Step 2: Preprocess data"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: DATA PREPROCESSING")
        logger.info("=" * 80)
        
        # Handle missing values
        self.df = self.preprocessor.handle_missing_values(self.df, fit=True)
        
        # Remove extreme outliers
        self.df = self.preprocessor.remove_extreme_outliers(self.df)
        
        # Split data
        self.train_df, self.test_df = self.preprocessor.stratified_split(self.df)
        
        logger.info("Preprocessing complete")
        return self.train_df, self.test_df
    
    def run_feature_engineering(self):
        """Step 3: Feature engineering"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 3: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        # Separate features and target
        self.X_train, self.y_train = prepare_features_target(self.train_df)
        self.X_test, self.y_test = prepare_features_target(self.test_df)
        
        # Apply feature engineering
        self.X_train_transformed = self.fe_pipeline.fit_transform(self.X_train, self.y_train)
        self.X_test_transformed = self.fe_pipeline.transform(self.X_test)
        
        logger.info(f"Feature engineering complete. Final features: {self.X_train_transformed.shape[1]}")
        return self.X_train_transformed, self.X_test_transformed
    
    def run_hyperparameter_optimization(self):
        """Step 4 (Optional): Hyperparameter optimization"""
        if not self.optimize_hyperparams:
            logger.info("\nSkipping hyperparameter optimization (use --optimize flag to enable)")
            return None
        
        logger.info("\n" + "=" * 80)
        logger.info("STEP 4: HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 80)
        
        optimizer = OptunaOptimizer(quick_mode=self.quick_mode)
        self.best_params = optimizer.optimize_all(self.X_train_transformed, self.y_train)
        
        # Save best parameters
        optimizer.save_best_params()
        
        logger.info("Hyperparameter optimization complete")
        return self.best_params
    
    def run_model_training(self):
        """Step 5: Train stacked ensemble"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: MODEL TRAINING")
        logger.info("=" * 80)
        
        # Create and train model
        self.model = StackedEnsembleModel(base_learner_params=self.best_params)
        self.model.train(
            self.X_train_transformed, 
            self.y_train,
            feature_names=self.fe_pipeline.get_feature_names()
        )
        
        # Save model
        self.model.save_model()
        
        logger.info("Model training complete")
        return self.model
    
    def run_evaluation(self):
        """Step 6: Evaluate model"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 80)
        
        # Generate predictions
        self.train_predictions = self.model.predict(self.X_train_transformed)
        self.test_predictions = self.model.predict(self.X_test_transformed)
        
        # Create performance report
        self.performance_report = PerformanceReport("Stacked Ensemble Model")
        self.performance_report.add_train_metrics(self.y_train, self.train_predictions)
        self.performance_report.add_test_metrics(self.y_test, self.test_predictions)
        self.performance_report.print_report()
        
        # Save report
        report_path = OUTPUT_DIR / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.performance_report.to_dict(), f, indent=2)
        
        logger.info(f"Performance report saved to {report_path}")
        return self.performance_report
    
    def run_explainability(self):
        """Step 7: Generate explainability reports"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: EXPLAINABILITY ANALYSIS (XAI)")
        logger.info("=" * 80)
        
        # SHAP analysis
        logger.info("\nGenerating SHAP analysis...")
        shap_analyzer = SHAPAnalyzer(
            self.model.model,
            feature_names=self.fe_pipeline.get_feature_names()
        )
        shap_plots, shap_importance = shap_analyzer.generate_full_report(self.X_test_transformed)
        
        # Save SHAP importance
        shap_importance.to_csv(OUTPUT_DIR / 'shap_importance.csv', index=False)
        
        # LIME analysis
        logger.info("\nGenerating LIME analysis...")
        lime_analyzer = LIMEAnalyzer(
            self.model.model,
            self.X_train_transformed,
            feature_names=self.fe_pipeline.get_feature_names()
        )
        lime_results = lime_analyzer.explain_multiple(
            self.X_test_transformed, 
            self.y_test, 
            n_samples=3
        )
        
        logger.info("Explainability analysis complete")
        return shap_importance, lime_results
    
    def run_validation(self, shap_importance):
        """Step 8: Robustness validation"""
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: ROBUSTNESS VALIDATION")
        logger.info("=" * 80)
        
        # Residual analysis
        logger.info("\nPerforming residual analysis...")
        residual_analyzer = ResidualAnalyzer(self.y_test, self.test_predictions)
        residual_report = residual_analyzer.generate_full_report(self.X_test_transformed)
        
        # Permutation importance
        logger.info("\nCalculating permutation importance...")
        perm_analyzer = PermutationImportanceAnalyzer(
            self.model.model,
            self.X_test_transformed,
            self.y_test,
            feature_names=self.fe_pipeline.get_feature_names()
        )
        perm_report = perm_analyzer.generate_full_report(
            shap_importance_df=shap_importance,
            n_repeats=5 if self.quick_mode else 10
        )
        
        logger.info("Validation complete")
        return residual_report, perm_report
    
    def run_full_pipeline(self):
        """Execute complete pipeline"""
        logger.info("\n" + "#" * 80)
        logger.info("# HOUSING VALUATION ENGINE - FULL PIPELINE EXECUTION")
        logger.info("#" * 80)
        
        try:
            # Execute all steps
            self.run_data_acquisition()
            self.run_preprocessing()
            self.run_feature_engineering()
            self.run_hyperparameter_optimization()
            self.run_model_training()
            self.run_evaluation()
            shap_importance, lime_results = self.run_explainability()
            residual_report, perm_report = self.run_validation(shap_importance)
            
            logger.info("\n" + "#" * 80)
            logger.info("# PIPELINE EXECUTION COMPLETE")
            logger.info("#" * 80)
            logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")
            logger.info(f"Model saved to: {MODELS_DIR}")
            
            return {
                'performance': self.performance_report,
                'shap_importance': shap_importance,
                'residual_report': residual_report,
                'perm_report': perm_report,
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Housing Valuation Engine')
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'train', 'predict', 'explain'],
                       help='Execution mode')
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (reduced trials/samples for testing)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = HousingValuationPipeline(
        optimize_hyperparams=args.optimize,
        quick_mode=args.quick
    )
    
    if args.mode == 'full':
        # Run complete pipeline
        results = pipeline.run_full_pipeline()
        
    elif args.mode == 'train':
        # Train only
        pipeline.run_data_acquisition()
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_hyperparameter_optimization()
        pipeline.run_model_training()
        pipeline.run_evaluation()
        
    elif args.mode == 'explain':
        # Load existing model and explain
        logger.info("Loading existing model...")
        pipeline.run_data_acquisition()
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        
        # Load model
        pipeline.model = StackedEnsembleModel()
        pipeline.model.load_model()
        
        # Generate predictions
        pipeline.test_predictions = pipeline.model.predict(pipeline.X_test_transformed)
        
        # Run explainability
        shap_importance, lime_results = pipeline.run_explainability()
        pipeline.run_validation(shap_importance)
    
    logger.info("\n✓ Execution complete!")


if __name__ == "__main__":
    main()
