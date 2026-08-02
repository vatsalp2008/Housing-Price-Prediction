# End-to-End Interpretable Housing Valuation Engine

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated machine learning system for housing price prediction that combines **advanced feature engineering**, **stacked ensemble learning**, **automated hyperparameter optimization**, and **comprehensive explainability** (XAI) to deliver accurate, interpretable, and trustworthy valuations.

## 🌟 Key Features

### Advanced ML Architecture
- **Stacked Generalization**: XGBoost, LightGBM, and Random Forest base learners with Bayesian Ridge meta-learner
- **Automated Feature Engineering**: Interaction terms, smoothed target encoding, Box-Cox transformations
- **Hyperparameter Optimization**: Optuna-based automated tuning with custom RMSE objectives
- **Market Adjustment Layer**: Integration of macroeconomic indicators (mortgage rates, CPI)

### Explainability & Trust (XAI)
- **SHAP Analysis**: Global feature importance and dependence plots
- **LIME Reports**: Local explanations for individual predictions with HTML visualizations
- **Permutation Importance**: Robustness validation and overfitting detection
- **Residual Analysis**: Homoscedasticity testing and diagnostic plots

### Production-Ready
- Modular, maintainable codebase with clear separation of concerns
- Comprehensive logging and error handling
- Model persistence and versioning
- Command-line interface for different execution modes

## 📊 Performance Metrics

The model achieves strong performance on the Ames Housing Dataset:

- **R² Score**: > 0.85 (target: 0.85)
- **RMSE**: < $25,000 (typical house price: $180,000)
- **MAE**: Interpretable error magnitude for business decisions

*See `EXECUTIVE_SUMMARY.md` for detailed performance analysis and key price drivers.*

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd "Housing Price Prediction"

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Tests

```bash
pytest tests/
```

The suite covers the metrics helpers, preprocessing and feature engineering.
It needs no dataset download and runs in about a second.

### Basic Usage

#### 1. Run Complete Pipeline (Training + Evaluation + Explainability)

```bash
python src/main.py --mode full
```

#### 2. Train with Hyperparameter Optimization

```bash
python src/main.py --mode train --optimize
```

#### 3. Quick Test Mode (Reduced Trials)

```bash
python src/main.py --mode full --quick
```

#### 4. Generate Explanations for Existing Model

```bash
python src/main.py --mode explain
```

#### 5. Score the Held-Out Test Set with a Saved Model

```bash
python src/main.py --mode predict
```

Writes `outputs/predictions.csv` with actual price, predicted price and error
per row.

## 📁 Project Structure

```
Housing Price Prediction/
├── src/
│   ├── main.py                          # Main pipeline orchestrator
│   ├── config.py                        # Centralized configuration
│   ├── data_acquisition.py              # Data download & macroeconomic integration
│   ├── feature_engineering.py           # Advanced feature transformations
│   ├── models/
│   │   ├── base_learners.py            # XGBoost, LightGBM, Random Forest
│   │   ├── stacked_ensemble.py         # Stacking with Bayesian Ridge
│   │   └── hyperparameter_optimization.py  # Optuna optimization
│   ├── explainability/
│   │   ├── shap_analysis.py            # SHAP global importance
│   │   └── lime_analysis.py            # LIME local explanations
│   ├── validation/
│   │   ├── residual_analysis.py        # Homoscedasticity testing
│   │   └── permutation_importance.py   # Feature importance validation
│   └── utils/
│       ├── preprocessing.py            # Missing values, outliers, splitting
│       └── metrics.py                  # Performance evaluation
├── tests/                               # Pytest suite
│   ├── conftest.py                     # Puts src/ on sys.path
│   ├── test_metrics.py                 # Evaluation metrics
│   ├── test_preprocessing.py           # Imputation, outliers, splitting
│   ├── test_feature_engineering.py     # Encoding and transformations
│   ├── test_validation.py              # Residuals & permutation importance
│   └── test_shap_analysis.py           # SHAP explainability
├── data/                                # Dataset storage
├── outputs/                             # Visualizations and reports
├── models_saved/                        # Trained model artifacts
├── notebooks/                           # Jupyter notebooks for exploration
├── verify_installation.py               # Dependency & structure check
├── requirements.txt                     # Python dependencies
├── LICENSE                             # MIT license
├── README.md                           # This file
└── EXECUTIVE_SUMMARY.md                # Performance & insights report
```

## 🔬 Technical Details

### Feature Engineering Pipeline

1. **Interaction Terms**: Automatically generated for key feature pairs
   - `OverallQual × GrLivArea`
   - `YearBuilt × TotalBsmtSF`
   - `GarageArea × GarageCars`

2. **Target Encoding**: Smoothed encoding for high-cardinality categoricals
   - Prevents target leakage with cross-validation
   - Smoothing parameter: 10 (configurable)

3. **Box-Cox Transformations**: Applied to skewed features (skewness > 0.75)
   - Normalizes distributions for better model performance

### Stacked Ensemble Architecture

```
┌─────────────────────────────────────────┐
│         Base Learners (Level 0)         │
├─────────────────────────────────────────┤
│  • XGBoost Regressor                    │
│  • LightGBM Regressor                   │
│  • Random Forest Regressor              │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│      Meta-Learner (Level 1)             │
├─────────────────────────────────────────┤
│  • Bayesian Ridge Regression            │
└─────────────────────────────────────────┘
                  ↓
         Final Prediction
```

### Hyperparameter Optimization

- **Framework**: Optuna with Tree-structured Parzen Estimator (TPE)
- **Objective**: Minimize RMSE via 5-fold cross-validation
- **Search Space**: Model-specific ranges for each base learner
- **Trials**: 100 (default) or 10 (quick mode)

## 📈 Explainability (XAI)

### SHAP (Global Importance)

```python
from explainability.shap_analysis import SHAPAnalyzer

analyzer = SHAPAnalyzer(model, feature_names=feature_names)
plots, importance = analyzer.generate_full_report(X_test)
```

**Outputs**:
- `shap_summary.png`: Feature importance summary
- `shap_bar.png`: Mean absolute SHAP values
- `shap_dependence_*.png`: Feature dependence plots

### LIME (Local Explanations)

```python
from explainability.lime_analysis import LIMEAnalyzer

analyzer = LIMEAnalyzer(model, X_train, feature_names=feature_names)
result = analyzer.explain_and_visualize(instance, actual_value)
```

**Outputs**:
- `lime_report.html`: Interactive HTML explanation
- `lime_plot.png`: Feature contribution visualization

## 🔍 Validation & Robustness

### Residual Analysis

- **Residuals vs Fitted**: Check for heteroscedasticity patterns
- **Q-Q Plot**: Assess normality of residuals
- **Breusch-Pagan Test**: Statistical test for homoscedasticity (p > 0.05 desired)

### Permutation Importance

- **Overfitting Detection**: Identifies features with negligible importance
- **SHAP Comparison**: Validates feature importance across methods
- **Statistical Significance**: Multiple permutation repeats for robustness

## 📊 Outputs

After running the pipeline, check the `outputs/` directory for:

| File | Description |
|------|-------------|
| `shap_summary.png` | Global feature importance visualization |
| `lime_report.html` | Individual prediction explanations |
| `residuals_vs_fitted.png` | Residual diagnostic plot |
| `permutation_importance.png` | Feature importance via permutation |
| `importance_comparison.png` | SHAP vs Permutation comparison |
| `performance_report.json` | Detailed metrics (R², RMSE, MAE) |
| `shap_importance.csv` | Feature importance rankings |

## 🛠️ Configuration

Edit `src/config.py` to customize:

- **Macroeconomic Indicators**: Update mortgage rates and CPI
- **Feature Engineering**: Modify interaction terms and thresholds
- **Model Parameters**: Adjust base learner hyperparameters
- **Optuna Settings**: Change trial counts and timeout
- **SHAP/LIME**: Configure sample sizes and display options

## 📚 Notebooks

The `notebooks/` directory holds placeholders for interactive analysis. These
are not written yet — the equivalent functionality is available today through
the CLI documented above.

| Notebook | Planned content | Status |
|----------|-----------------|--------|
| `01_exploratory_analysis.ipynb` | Dataset overview and visualizations | Not written |
| `02_model_training.ipynb` | Step-by-step training process | Not written |
| `03_explainability_demo.ipynb` | SHAP and LIME demonstrations | Not written |

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ames Housing Dataset**: Dean De Cock (2011)
- **SHAP**: Lundberg & Lee (2017)
- **LIME**: Ribeiro et al. (2016)
- **Optuna**: Akiba et al. (2019)

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for interpretable and trustworthy AI**
# Housing-Price-Prediction
