"""Tests for the configuration module

These guard the invariants other modules rely on. The feature-name checks in
particular exist because a mismatch between configured names and the dataset's
real column names silently disabled feature engineering entirely.
"""

import inspect

import pytest

import config
from feature_engineering import BoxCoxTransformer, TargetEncoder


class TestPaths:
    def test_project_root_contains_src(self):
        assert (config.PROJECT_ROOT / 'src').is_dir()

    def test_directories_exist_after_import(self):
        for path in (config.DATA_DIR, config.OUTPUT_DIR, config.MODELS_DIR):
            assert path.is_dir()

    def test_ensure_directories_is_idempotent(self):
        config.ensure_directories()
        config.ensure_directories()
        assert config.DATA_DIR.is_dir()

    def test_ensure_directories_recreates_a_removed_directory(self, tmp_path, monkeypatch):
        target = tmp_path / 'made'
        monkeypatch.setattr(config, 'MODELS_DIR', target)

        config.ensure_directories()

        assert target.is_dir()


class TestFeatureNames:
    """Configured column names must use the Ames spelling, which has spaces"""

    def test_interaction_names_are_not_the_compact_kaggle_form(self):
        """Regression test: compact names matched nothing and were skipped"""
        compact = {'OverallQual', 'GrLivArea', 'YearBuilt', 'TotalBsmtSF',
                   'GarageArea', 'GarageCars', '1stFlrSF', '2ndFlrSF'}

        for first, second in config.FEATURE_CONFIG['interaction_terms']:
            assert first not in compact, f"{first} is the Kaggle spelling"
            assert second not in compact, f"{second} is the Kaggle spelling"

    def test_high_cardinality_names_are_not_the_compact_form(self):
        compact = {'Exterior1st', 'Exterior2nd'}
        assert not compact & set(config.FEATURE_CONFIG['high_cardinality_cols'])

    def test_interaction_pairs_are_two_distinct_columns(self):
        for pair in config.FEATURE_CONFIG['interaction_terms']:
            assert len(pair) == 2
            assert pair[0] != pair[1]

    def test_no_duplicate_interaction_pairs(self):
        pairs = [tuple(sorted(p)) for p in config.FEATURE_CONFIG['interaction_terms']]
        assert len(pairs) == len(set(pairs))

    def test_no_duplicate_high_cardinality_columns(self):
        columns = config.FEATURE_CONFIG['high_cardinality_cols']
        assert len(columns) == len(set(columns))


class TestModelConfig:
    def test_test_size_is_a_fraction(self):
        assert 0 < config.MODEL_CONFIG['test_size'] < 1

    def test_cv_folds_allows_a_holdout(self):
        assert config.MODEL_CONFIG['cv_folds'] >= 2

    def test_random_state_is_set(self):
        assert isinstance(config.MODEL_CONFIG['random_state'], int)


class TestBaseLearnerParams:
    @pytest.mark.parametrize('params', [
        config.XGBOOST_PARAMS,
        config.LIGHTGBM_PARAMS,
        config.RANDOM_FOREST_PARAMS,
    ])
    def test_seeded_from_the_shared_random_state(self, params):
        assert params['random_state'] == config.MODEL_CONFIG['random_state']

    def test_bayesian_ridge_uses_the_supported_iteration_argument(self):
        """Regression test: n_iter was removed in scikit-learn 1.5"""
        from sklearn.linear_model import BayesianRidge

        accepted = inspect.signature(BayesianRidge.__init__).parameters
        for name in config.BAYESIAN_RIDGE_PARAMS:
            assert name in accepted, f"BayesianRidge does not accept {name!r}"

    def test_bayesian_ridge_params_construct_a_model(self):
        from sklearn.linear_model import BayesianRidge

        assert BayesianRidge(**config.BAYESIAN_RIDGE_PARAMS) is not None


class TestOptunaConfig:
    def test_quick_mode_runs_fewer_trials(self):
        assert config.OPTUNA_CONFIG['n_trials_quick'] < config.OPTUNA_CONFIG['n_trials']

    def test_timeout_is_positive(self):
        assert config.OPTUNA_CONFIG['timeout'] > 0


class TestDefaultsReachTheirConsumers:
    def test_boxcox_threshold_comes_from_config(self):
        assert BoxCoxTransformer().skewness_threshold == config.FEATURE_CONFIG['skewness_threshold']

    def test_target_encoder_smoothing_comes_from_config(self):
        assert TargetEncoder().smoothing == config.FEATURE_CONFIG['target_encoding_smoothing']

    def test_target_encoder_defaults_to_the_configured_columns(self):
        assert TargetEncoder().columns == config.FEATURE_CONFIG['high_cardinality_cols']

    def test_target_encoder_seeds_from_model_config(self):
        assert TargetEncoder().random_state == config.MODEL_CONFIG['random_state']


class TestPerformanceTargets:
    def test_targets_are_defined_and_sane(self):
        assert 0 < config.PERFORMANCE_TARGETS['r2_score'] <= 1
        assert config.PERFORMANCE_TARGETS['rmse'] > 0
