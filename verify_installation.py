"""
Quick test script to verify installation and run a minimal pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        import xgboost
        import lightgbm
        import optuna
        import shap
        import lime
        import matplotlib
        import seaborn
        import scipy
        import statsmodels
        
        print("✓ All required packages installed successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False


def test_project_structure():
    """Test that project directories exist"""
    print("\nTesting project structure...")
    
    required_dirs = [
        'src',
        'src/models',
        'src/explainability',
        'src/validation',
        'src/utils',
        'data',
        'outputs',
        'models_saved',
        'notebooks',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def test_module_imports():
    """Test that custom modules can be imported"""
    print("\nTesting custom module imports...")
    
    try:
        from config import MACRO_INDICATORS, MODEL_CONFIG
        from data_acquisition import DataAcquisition
        from feature_engineering import FeatureEngineeringPipeline
        from models.stacked_ensemble import StackedEnsembleModel
        from explainability.shap_analysis import SHAPAnalyzer
        from validation.residual_analysis import ResidualAnalyzer
        
        print("✓ All custom modules imported successfully")
        return True
        
    except ImportError as e:
        print(f"✗ Module import error: {e}")
        return False


def run_quick_test():
    """Run a quick end-to-end test with minimal data"""
    print("\nRunning quick end-to-end test...")
    print("(This will download the dataset and run a minimal pipeline)")
    
    try:
        from data_acquisition import DataAcquisition
        from utils.preprocessing import DataPreprocessor, prepare_features_target
        from feature_engineering import FeatureEngineeringPipeline
        from models.base_learners import get_xgboost_model
        from utils.metrics import calculate_metrics, print_metrics
        
        # Load data
        print("\n1. Loading data...")
        acquisition = DataAcquisition()
        df, summary = acquisition.prepare_dataset()
        print(f"   Loaded {len(df)} samples")
        
        # Preprocess
        print("\n2. Preprocessing...")
        preprocessor = DataPreprocessor()
        df = preprocessor.handle_missing_values(df)
        df = preprocessor.remove_extreme_outliers(df)
        
        # Use small subset for quick test
        df_sample = df.sample(n=min(500, len(df)), random_state=42)
        train_df, test_df = preprocessor.stratified_split(df_sample)
        print(f"   Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Feature engineering
        print("\n3. Feature engineering...")
        X_train, y_train = prepare_features_target(train_df)
        X_test, y_test = prepare_features_target(test_df)
        
        fe_pipeline = FeatureEngineeringPipeline()
        X_train_transformed = fe_pipeline.fit_transform(X_train, y_train)
        X_test_transformed = fe_pipeline.transform(X_test)
        print(f"   Features: {X_train_transformed.shape[1]}")
        
        # Train simple model (just XGBoost for quick test)
        print("\n4. Training model...")
        model = get_xgboost_model({'n_estimators': 100})
        model.fit(X_train_transformed, y_train)
        
        # Evaluate
        print("\n5. Evaluating...")
        y_pred = model.predict(X_test_transformed)
        metrics = calculate_metrics(y_test, y_pred)
        print_metrics(metrics, "Test Set")
        
        print("\n" + "=" * 60)
        print("✓ QUICK TEST PASSED!")
        print("=" * 60)
        print("\nYour installation is working correctly.")
        print("Run the full pipeline with:")
        print("  python src/main.py --mode full --quick")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("HOUSING VALUATION ENGINE - INSTALLATION TEST")
    print("=" * 60)
    
    # Run tests
    imports_ok = test_imports()
    structure_ok = test_project_structure()
    modules_ok = test_module_imports()
    
    if not (imports_ok and structure_ok and modules_ok):
        print("\n" + "=" * 60)
        print("✗ TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
        return False
    
    # Run quick pipeline test
    print("\n" + "=" * 60)
    user_input = input("Run quick end-to-end test? (y/n): ")
    if user_input.lower() == 'y':
        test_ok = run_quick_test()
        return test_ok
    else:
        print("\nSkipping end-to-end test.")
        print("Run manually with: python test_installation.py")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
