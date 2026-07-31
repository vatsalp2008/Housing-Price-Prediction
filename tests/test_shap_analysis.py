"""Tests for the SHAP analysis module"""

import matplotlib
matplotlib.use('Agg')  # no display in test environments

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import config
from explainability.shap_analysis import SHAPAnalyzer


@pytest.fixture
def data():
    """Feature frame where f0 drives the target"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 5)), columns=[f'f{i}' for i in range(5)])
    y = pd.Series(X['f0'] * 3 + 50.0 + rng.normal(0, 0.2, 300))
    return X, y


@pytest.fixture
def analyzer(data):
    """Analyzer over a tree model, so TreeExplainer applies"""
    X, y = data
    model = RandomForestRegressor(n_estimators=15, random_state=0).fit(X, y)
    return SHAPAnalyzer(model, X_background=X.head(50), feature_names=list(X.columns))


class TestCalculateShapValues:
    def test_records_the_rows_it_explained(self, analyzer, data):
        X, _ = data
        analyzer.calculate_shap_values(X, max_samples=40)

        assert len(analyzer.X_explained) == 40
        assert len(analyzer.shap_values) == 40

    def test_uses_all_rows_when_under_the_cap(self, analyzer, data):
        X, _ = data
        analyzer.calculate_shap_values(X.head(20), max_samples=100)
        assert len(analyzer.X_explained) == 20

    def test_sampling_is_reproducible(self, analyzer, data):
        X, _ = data

        analyzer.calculate_shap_values(X, max_samples=40)
        first = list(analyzer.X_explained.index)

        analyzer.calculate_shap_values(X, max_samples=40)
        second = list(analyzer.X_explained.index)

        assert first == second

    def test_sampling_uses_the_configured_seed(self, analyzer, data):
        X, _ = data
        analyzer.calculate_shap_values(X, max_samples=40)
        expected = X.sample(n=40, random_state=config.MODEL_CONFIG['random_state']).index
        assert list(analyzer.X_explained.index) == list(expected)


class TestResolvePlotData:
    def test_falls_back_to_the_explained_subset(self, analyzer, data):
        """Regression test: the full frame was plotted against sampled values"""
        X, _ = data
        analyzer.calculate_shap_values(X, max_samples=40)

        resolved = analyzer._resolve_plot_data(X, analyzer.shap_values)

        assert len(resolved) == 40

    def test_accepts_a_matching_dataset(self, analyzer, data):
        X, _ = data
        analyzer.calculate_shap_values(X.head(30))
        resolved = analyzer._resolve_plot_data(X.head(30), analyzer.shap_values)
        assert len(resolved) == 30

    def test_raises_when_no_dataset_matches(self, analyzer, data):
        X, _ = data
        analyzer.calculate_shap_values(X.head(30))
        analyzer.X_explained = None

        with pytest.raises(ValueError, match="rows"):
            analyzer._resolve_plot_data(X, analyzer.shap_values)


class TestPlots:
    def test_summary_and_bar_plots_are_written(self, analyzer, data, tmp_path):
        X, _ = data
        analyzer.calculate_shap_values(X, max_samples=40)

        summary = analyzer.plot_summary(save_path=tmp_path / 'summary.png')
        bar = analyzer.plot_bar(save_path=tmp_path / 'bar.png')

        assert summary.stat().st_size > 0
        assert bar.stat().st_size > 0

    def test_force_plot_works_without_an_explicit_dataset(self, analyzer, data, tmp_path):
        """Regression test: plot_force dereferenced X unconditionally"""
        X, _ = data
        analyzer.calculate_shap_values(X, max_samples=30)

        path = analyzer.plot_force(0, save_path=tmp_path / 'force.png')

        assert path.stat().st_size > 0

    def test_force_plot_uses_the_model_baseline(self, analyzer, data):
        """Regression test: the fallback used mean(shap_values), near zero"""
        X, y = data
        analyzer.calculate_shap_values(X, max_samples=30)

        base = float(np.ravel(analyzer.explainer.expected_value)[0])

        assert base == pytest.approx(y.mean(), abs=2.0)
        assert abs(float(np.mean(analyzer.shap_values))) < 1.0

    def test_force_plot_errors_without_a_baseline_source(self, analyzer, data, tmp_path):
        X, _ = data
        analyzer.calculate_shap_values(X, max_samples=10)

        class Bare:
            def shap_values(self, Z):
                return np.zeros((len(Z), Z.shape[1]))

        analyzer.explainer = Bare()
        analyzer.X_background = None

        with pytest.raises(ValueError, match="expected_value"):
            analyzer.plot_force(0, save_path=tmp_path / 'force.png')


class TestFeatureImportance:
    def test_raises_before_values_exist(self, analyzer):
        with pytest.raises(ValueError):
            analyzer.get_feature_importance()

    def test_identifies_the_driving_feature(self, analyzer, data):
        X, _ = data
        analyzer.calculate_shap_values(X, max_samples=60)

        importance = analyzer.get_feature_importance()

        assert importance.iloc[0]['feature'] == 'f0'
        assert list(importance['importance']) == sorted(importance['importance'], reverse=True)


class TestFullReport:
    def test_report_completes_and_writes_every_plot(self, analyzer, data, tmp_path):
        """Regression test: this raised an assertion inside shap"""
        X, _ = data

        plots, importance = analyzer.generate_full_report(X, save_dir=tmp_path)

        assert plots['summary'].stat().st_size > 0
        assert plots['bar'].stat().st_size > 0
        assert len(plots['dependence']) == 3
        assert all(p.stat().st_size > 0 for p in plots['dependence'])
        assert importance.iloc[0]['feature'] == 'f0'
