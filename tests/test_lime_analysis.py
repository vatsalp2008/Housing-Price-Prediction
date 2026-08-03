"""Tests for the LIME analysis module"""

import matplotlib
matplotlib.use('Agg')  # no display in test environments

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

import config

lime_analysis = pytest.importorskip(
    'explainability.lime_analysis',
    reason='lime is an optional dependency',
)
LIMEAnalyzer = lime_analysis.LIMEAnalyzer


@pytest.fixture
def data():
    """Feature frame where f0 drives the target"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=[f'f{i}' for i in range(4)])
    y = pd.Series(X['f0'] * 3 + rng.normal(0, 0.2, 150))
    return X, y


@pytest.fixture
def analyzer(data):
    X, y = data
    model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    return LIMEAnalyzer(model, X, feature_names=list(X.columns))


class TestAsRow:
    def test_series_becomes_a_single_row(self, data):
        X, _ = data
        assert LIMEAnalyzer._as_row(X.iloc[0]).shape == (1, 4)

    def test_dataframe_keeps_its_rows(self, data):
        X, _ = data
        assert LIMEAnalyzer._as_row(X.head(1)).shape == (1, 4)

    def test_plain_array_is_reshaped(self):
        assert LIMEAnalyzer._as_row(np.array([1.0, 2.0, 3.0])).shape == (1, 3)


class TestSeeding:
    def test_defaults_to_the_configured_seed(self, analyzer):
        assert analyzer.random_state == config.MODEL_CONFIG['random_state']

    def test_explicit_seed_is_kept(self, data):
        X, y = data
        model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        assert LIMEAnalyzer(model, X, random_state=7).random_state == 7

    def test_explanations_are_reproducible(self, data):
        """Regression test: LIME perturbs randomly and had no seed"""
        X, y = data
        model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

        def explain():
            a = LIMEAnalyzer(model, X, feature_names=list(X.columns))
            return dict(a.explain_instance(X.iloc[0]).as_list())

        assert explain() == explain()

    def test_different_seeds_give_different_explanations(self, data):
        X, y = data
        model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)

        def explain(seed):
            a = LIMEAnalyzer(model, X, feature_names=list(X.columns), random_state=seed)
            return dict(a.explain_instance(X.iloc[0]).as_list())

        assert explain(1) != explain(2)


class TestExplainInstance:
    def test_returns_the_requested_number_of_features(self, analyzer, data):
        X, _ = data
        explanation = analyzer.explain_instance(X.iloc[0], num_features=3)
        assert len(explanation.as_list()) == 3

    def test_attributes_weight_to_the_driving_feature(self, analyzer, data):
        X, _ = data
        weights = analyzer.explain_instance(X.iloc[0]).as_list()
        strongest = max(weights, key=lambda kv: abs(kv[1]))[0]
        assert 'f0' in strongest


class TestComparePredictions:
    def test_accepts_a_dataframe(self, analyzer, data):
        """Regression test: iterating a DataFrame yielded column names"""
        X, _ = data

        comparison, explanations = analyzer.compare_predictions(X.head(3))

        assert len(comparison) == 3
        assert len(explanations) == 3

    def test_predictions_match_the_model(self, analyzer, data):
        X, _ = data
        comparison, _ = analyzer.compare_predictions(X.head(3))
        np.testing.assert_allclose(comparison['Prediction'], analyzer.model.predict(X.head(3)))

    def test_error_column_is_added_with_actuals(self, analyzer, data):
        X, y = data
        comparison, _ = analyzer.compare_predictions(X.head(3), actual_values=list(y.head(3)))

        assert 'Error' in comparison.columns
        np.testing.assert_allclose(
            comparison['Error'], (comparison['Prediction'] - comparison['Actual']).abs()
        )

    def test_accepts_a_list_of_series(self, analyzer, data):
        X, _ = data
        comparison, _ = analyzer.compare_predictions([X.iloc[0], X.iloc[1]])
        assert len(comparison) == 2

    def test_accepts_a_numpy_array(self, analyzer, data):
        X, _ = data
        comparison, _ = analyzer.compare_predictions(X.head(2).values)
        assert len(comparison) == 2


class TestReports:
    def test_html_report_is_utf8(self, data, tmp_path):
        """Regression test: the default encoding fails on non-ASCII names"""
        rng = np.random.default_rng(0)
        columns = ['площадь', 'f1', 'f2']
        X = pd.DataFrame(rng.normal(size=(80, 3)), columns=columns)
        y = pd.Series(X['площадь'] * 2 + rng.normal(0, 0.2, 80))
        model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        analyzer = LIMEAnalyzer(model, X, feature_names=columns)

        path = analyzer.save_html_report(
            analyzer.explain_instance(X.iloc[0]), tmp_path / 'report.html'
        )

        assert 'площадь' in path.read_text(encoding='utf-8')

    def test_plot_is_written(self, analyzer, data, tmp_path):
        X, _ = data
        path = analyzer.plot_explanation(
            analyzer.explain_instance(X.iloc[0]), tmp_path / 'plot.png'
        )
        assert path.stat().st_size > 0

    def test_explain_multiple_respects_the_cap(self, analyzer, data, monkeypatch, tmp_path):
        X, y = data
        monkeypatch.setattr(lime_analysis, 'OUTPUT_DIR', tmp_path)

        results = analyzer.explain_multiple(X, y, n_samples=2)

        assert len(results) == 2
        assert all(r['prediction'] is not None for r in results)

    def test_explain_multiple_stops_at_the_data_length(self, analyzer, data, monkeypatch, tmp_path):
        X, y = data
        monkeypatch.setattr(lime_analysis, 'OUTPUT_DIR', tmp_path)

        results = analyzer.explain_multiple(X.head(1), y.head(1), n_samples=5)

        assert len(results) == 1
