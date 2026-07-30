"""Tests for the residual analysis and permutation importance modules"""

import matplotlib
matplotlib.use('Agg')  # no display in test environments

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import config
from validation.permutation_importance import PermutationImportanceAnalyzer
from validation.residual_analysis import ResidualAnalyzer


@pytest.fixture
def analyzer():
    """Residual analyzer over well-behaved predictions"""
    rng = np.random.default_rng(0)
    y_true = rng.normal(200000, 50000, 300)
    y_pred = y_true + rng.normal(0, 5000, 300)
    return ResidualAnalyzer(y_true, y_pred)


class TestResidualAnalyzer:
    def test_residuals_are_true_minus_predicted(self):
        a = ResidualAnalyzer([10.0, 20.0], [8.0, 25.0])
        np.testing.assert_allclose(a.residuals, [2.0, -5.0])

    def test_standardized_residuals_have_unit_spread(self, analyzer):
        standardized = analyzer.calculate_standardized_residuals()
        assert np.std(standardized) == pytest.approx(1.0)

    def test_zero_spread_residuals_do_not_divide_by_zero(self):
        """Regression test: perfect predictions produced nan/inf"""
        a = ResidualAnalyzer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

        standardized = a.calculate_standardized_residuals()

        assert np.all(np.isfinite(standardized))
        assert np.all(standardized == 0)

    def test_shapiro_wilk_returns_statistic_and_pvalue(self, analyzer):
        results = analyzer.shapiro_wilk_test()
        assert 0.0 <= results['pvalue'] <= 1.0
        assert np.isfinite(results['statistic'])

    def test_shapiro_wilk_is_reproducible_when_subsampling(self):
        """Regression test: the >5000 subsample used an unseeded RNG"""
        rng = np.random.default_rng(0)
        y_true = rng.normal(200000, 50000, 6000)
        a = ResidualAnalyzer(y_true, y_true + rng.normal(0, 5000, 6000))

        assert a.shapiro_wilk_test()['pvalue'] == a.shapiro_wilk_test()['pvalue']

    def test_shapiro_wilk_seed_changes_the_subsample(self):
        rng = np.random.default_rng(0)
        y_true = rng.normal(200000, 50000, 6000)
        a = ResidualAnalyzer(y_true, y_true + rng.normal(0, 5000, 6000))

        assert a.shapiro_wilk_test(random_state=1)['pvalue'] != a.shapiro_wilk_test(random_state=2)['pvalue']

    def test_breusch_pagan_works_without_a_constant_column(self, analyzer):
        """Regression test: statsmodels raised ValueError without an intercept"""
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list('abc'))

        results = analyzer.breusch_pagan_test(X)

        assert 0.0 <= results['lm_pvalue'] <= 1.0
        assert set(results) == {'lm_statistic', 'lm_pvalue', 'f_statistic', 'f_pvalue'}

    def test_breusch_pagan_does_not_double_add_a_constant(self, analyzer):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list('abc'))
        X_with_const = X.assign(const_feature=6.5)

        plain = analyzer.breusch_pagan_test(X)
        with_const = analyzer.breusch_pagan_test(X_with_const)

        assert plain['lm_pvalue'] == pytest.approx(with_const['lm_pvalue'])

    def test_plots_are_written(self, analyzer, tmp_path):
        for name in ('plot_residuals_vs_fitted', 'plot_qq',
                     'plot_residual_distribution', 'plot_scale_location'):
            path = getattr(analyzer, name)(tmp_path / f'{name}.png')
            assert path.exists() and path.stat().st_size > 0

    def test_full_report_contains_tests_and_summary(self, analyzer, tmp_path):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(300, 3)), columns=list('abc'))

        report = analyzer.generate_full_report(X, save_dir=tmp_path)

        assert 'shapiro_wilk' in report['tests']
        assert 'breusch_pagan' in report['tests']
        assert set(report['summary']) >= {'mean_residual', 'std_residual', 'skewness'}


@pytest.fixture
def perm_analyzer():
    """Permutation analyzer over a model with one dominant feature"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = X[:, 0] * 5 + rng.normal(0, 0.1, 200)
    model = LinearRegression().fit(X, y)
    return PermutationImportanceAnalyzer(model, X, y, feature_names=list('abcd'))


class TestPermutationImportanceAnalyzer:
    def test_raises_before_importance_is_calculated(self, perm_analyzer):
        with pytest.raises(ValueError):
            perm_analyzer.get_importance_dataframe()

    def test_identifies_the_dominant_feature(self, perm_analyzer):
        perm_analyzer.calculate_importance(n_repeats=5)
        top = perm_analyzer.get_importance_dataframe().iloc[0]
        assert top['feature'] == 'a'

    def test_dataframe_is_sorted_descending(self, perm_analyzer):
        perm_analyzer.calculate_importance(n_repeats=5)
        means = perm_analyzer.get_importance_dataframe()['importance_mean'].tolist()
        assert means == sorted(means, reverse=True)

    def test_random_state_zero_is_not_treated_as_unset(self, perm_analyzer):
        """Regression test: 'random_state or default' coerced 0 to 42"""
        with_zero = perm_analyzer.calculate_importance(n_repeats=5, random_state=0).importances_mean.copy()
        with_default = perm_analyzer.calculate_importance(n_repeats=5).importances_mean.copy()

        assert config.MODEL_CONFIG['random_state'] != 0
        assert not np.allclose(with_zero, with_default)

    def test_random_state_zero_is_reproducible(self, perm_analyzer):
        first = perm_analyzer.calculate_importance(n_repeats=5, random_state=0).importances_mean.copy()
        second = perm_analyzer.calculate_importance(n_repeats=5, random_state=0).importances_mean.copy()

        np.testing.assert_allclose(first, second)

    def test_normalize_scales_to_unit_range(self):
        scaled = PermutationImportanceAnalyzer._normalize(pd.Series([0.0, 5.0, 10.0]))
        assert list(scaled) == [0.0, 0.5, 1.0]

    def test_normalize_handles_a_flat_series(self):
        """Regression test: a zero range produced all-NaN bars"""
        scaled = PermutationImportanceAnalyzer._normalize(pd.Series([5.0, 5.0, 5.0]))
        assert list(scaled) == [0.0, 0.0, 0.0]
        assert not scaled.isnull().any()

    def test_compare_with_shap_produces_no_nans(self, perm_analyzer, tmp_path):
        perm_analyzer.calculate_importance(n_repeats=5)
        shap_df = pd.DataFrame({'feature': list('abcd'), 'importance': [4.0, 3.0, 2.0, 1.0]})

        path, comparison = perm_analyzer.compare_with_shap(
            shap_df, top_n=4, save_path=tmp_path / 'comparison.png'
        )

        assert path.exists()
        assert comparison[['permutation', 'shap']].isnull().sum().sum() == 0

    def test_detect_overfitting_flags_useless_features(self, perm_analyzer):
        perm_analyzer.calculate_importance(n_repeats=5)

        noisy = perm_analyzer.detect_overfitting(threshold=0.01)

        # b, c and d contribute nothing to y
        assert set(noisy['feature']) >= {'b', 'c', 'd'}
        assert 'a' not in set(noisy['feature'])

    def test_plot_importance_is_written(self, perm_analyzer, tmp_path):
        perm_analyzer.calculate_importance(n_repeats=5)
        path = perm_analyzer.plot_importance(top_n=4, save_path=tmp_path / 'importance.png')
        assert path.exists() and path.stat().st_size > 0
