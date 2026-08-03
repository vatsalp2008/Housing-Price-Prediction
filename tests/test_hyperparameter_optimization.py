"""Tests for the Optuna hyperparameter optimizer"""

import json

import numpy as np
import pandas as pd
import pytest

import config
from models.hyperparameter_optimization import OptunaOptimizer


@pytest.fixture
def data():
    """Small, fast regression problem"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=list('abc'))
    y = pd.Series(X['a'] * 3 + rng.normal(0, 0.2, 80))
    return X, y


class TestTrialBudget:
    def test_defaults_to_the_configured_trial_count(self):
        assert OptunaOptimizer().n_trials == config.OPTUNA_CONFIG['n_trials']

    def test_quick_mode_uses_the_reduced_count(self):
        assert OptunaOptimizer(quick_mode=True).n_trials == config.OPTUNA_CONFIG['n_trials_quick']

    def test_explicit_n_trials_beats_quick_mode(self):
        """Regression test: quick_mode silently discarded an explicit value"""
        assert OptunaOptimizer(n_trials=3, quick_mode=True).n_trials == 3

    def test_explicit_n_trials_is_used_without_quick_mode(self):
        assert OptunaOptimizer(n_trials=7).n_trials == 7


class TestTimeout:
    def test_defaults_to_the_configured_timeout(self):
        """Regression test: the configured timeout was never applied"""
        assert OptunaOptimizer().timeout == config.OPTUNA_CONFIG['timeout']

    def test_explicit_timeout_is_kept(self):
        assert OptunaOptimizer(timeout=5).timeout == 5

    def test_zero_timeout_is_not_treated_as_unset(self):
        assert OptunaOptimizer(timeout=0).timeout == 0

    def test_timeout_stops_a_long_study_early(self, data):
        """A huge trial budget must be cut short by the wall-clock budget"""
        X, y = data
        optimizer = OptunaOptimizer(n_trials=100000, timeout=5)

        optimizer.optimize_random_forest(X, y)

        completed = len(optimizer.studies['random_forest'].trials)
        assert 0 < completed < 100000


class TestOptimize:
    def test_random_forest_returns_usable_params(self, data):
        X, y = data
        optimizer = OptunaOptimizer(n_trials=2)

        params = optimizer.optimize_random_forest(X, y)

        assert set(params) >= {'n_estimators', 'max_depth', 'min_samples_split'}
        assert optimizer.best_params['rf'] == params

    def test_study_records_the_trials_it_ran(self, data):
        X, y = data
        optimizer = OptunaOptimizer(n_trials=2)

        optimizer.optimize_random_forest(X, y)

        assert len(optimizer.studies['random_forest'].trials) == 2

    def test_results_are_reproducible(self, data):
        """The sampler is seeded from MODEL_CONFIG"""
        X, y = data

        first = OptunaOptimizer(n_trials=3).optimize_random_forest(X, y)
        second = OptunaOptimizer(n_trials=3).optimize_random_forest(X, y)

        assert first == second

    def test_summary_reports_each_study(self, data):
        X, y = data
        optimizer = OptunaOptimizer(n_trials=2)
        optimizer.optimize_random_forest(X, y)

        summary = optimizer.get_optimization_summary()

        assert 'random_forest' in summary
        assert summary['random_forest']['n_trials'] == 2
        assert np.isfinite(summary['random_forest']['best_value'])

    def test_objective_returns_a_positive_rmse(self, data):
        X, y = data
        optimizer = OptunaOptimizer(n_trials=1)
        optimizer.optimize_random_forest(X, y)

        assert optimizer.studies['random_forest'].best_value > 0


class TestSaveBestParams:
    def test_writes_valid_json(self, data, tmp_path):
        X, y = data
        optimizer = OptunaOptimizer(n_trials=2)
        optimizer.optimize_random_forest(X, y)

        path = tmp_path / 'best.json'
        optimizer.save_best_params(path)

        payload = json.loads(path.read_text())
        assert 'rf' in payload
        assert payload['rf'] == optimizer.best_params['rf']

    def test_writes_an_empty_mapping_before_optimizing(self, tmp_path):
        path = tmp_path / 'empty.json'
        OptunaOptimizer().save_best_params(path)
        assert json.loads(path.read_text()) == {}
