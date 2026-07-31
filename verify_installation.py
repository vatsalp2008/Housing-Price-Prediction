"""
Quick test script to verify installation and run a minimal pipeline
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

#: Packages every mode needs
REQUIRED_PACKAGES = [
    'pandas', 'numpy', 'sklearn', 'xgboost', 'lightgbm',
    'optuna', 'matplotlib', 'seaborn', 'scipy', 'statsmodels',
]

#: Packages only one mode needs, mapped to what they unlock
OPTIONAL_PACKAGES = {
    'shap': 'SHAP explainability',
    'lime': 'LIME explainability (--mode explain)',
}


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    import importlib

    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"✗ Missing required packages: {', '.join(missing)}")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        return False

    print("✓ All required packages installed successfully")

    # Absent optional packages only disable one mode, so report and carry on
    for name, purpose in OPTIONAL_PACKAGES.items():
        try:
            importlib.import_module(name)
        except ImportError:
            print(f"! Optional package '{name}' missing - {purpose} unavailable")

    return True


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

        # Use small subset for quick check
        df_sample = df.sample(n=min(500, len(df)), random_state=42)

        # Same ordering as the real pipeline: split first, fit the imputer on
        # train only, and drop outliers from the training set alone
        train_df, test_df = preprocessor.stratified_split(df_sample)
        train_df = preprocessor.handle_missing_values(train_df, fit=True)
        test_df = preprocessor.handle_missing_values(test_df, fit=False)
        train_df = preprocessor.remove_extreme_outliers(train_df)
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


def main(argv=None):
    """Run all checks"""
    parser = argparse.ArgumentParser(
        description='Verify the installation and optionally run a minimal pipeline'
    )
    parser.add_argument('--run-pipeline', action='store_true',
                        help='Run the end-to-end pipeline check without prompting')
    parser.add_argument('--checks-only', action='store_true',
                        help='Run only the import and structure checks')
    args = parser.parse_args(argv)

    print("=" * 60)
    print("HOUSING VALUATION ENGINE - INSTALLATION CHECK")
    print("=" * 60)

    # Run checks
    imports_ok = test_imports()
    structure_ok = test_project_structure()
    modules_ok = test_module_imports()

    if not (imports_ok and structure_ok and modules_ok):
        print("\n" + "=" * 60)
        print("✗ CHECKS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues above before proceeding.")
        return False

    if args.checks_only:
        print("\nSkipping end-to-end check (--checks-only).")
        return True

    if args.run_pipeline:
        return run_quick_test()

    # Only prompt when there is a terminal to prompt at: piping this script or
    # running it in CI used to raise EOFError here.
    print("\n" + "=" * 60)
    if not sys.stdin.isatty():
        print("Non-interactive session; skipping end-to-end check.")
        print("Run it with: python verify_installation.py --run-pipeline")
        return True

    try:
        user_input = input("Run quick end-to-end check? (y/n): ")
    except EOFError:
        print("\nNo input available; skipping end-to-end check.")
        return True

    if user_input.strip().lower() in ('y', 'yes'):
        return run_quick_test()

    print("\nSkipping end-to-end check.")
    print("Run it with: python verify_installation.py --run-pipeline")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
