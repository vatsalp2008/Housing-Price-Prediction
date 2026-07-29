"""Tests for the evaluation metrics helpers"""

import numpy as np
import pytest

from utils.metrics import (
    PerformanceReport,
    calculate_metrics,
    calculate_prediction_intervals,
)


class TestCalculateMetrics:
    """calculate_metrics should report standard regression metrics"""

    def test_perfect_predictions(self):
        y = np.array([100.0, 200.0, 300.0])
        metrics = calculate_metrics(y, y)

        assert metrics['RMSE'] == pytest.approx(0.0)
        assert metrics['MAE'] == pytest.approx(0.0)
        assert metrics['R2'] == pytest.approx(1.0)
        assert metrics['MAPE'] == pytest.approx(0.0)
        assert metrics['Max_Error'] == pytest.approx(0.0)

    def test_known_values(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        metrics = calculate_metrics(y_true, y_pred)

        # errors of +10 and -20
        assert metrics['MAE'] == pytest.approx(15.0)
        assert metrics['RMSE'] == pytest.approx(np.sqrt((100 + 400) / 2))
        assert metrics['Max_Error'] == pytest.approx(20.0)
        # |10/100| and |20/200| are both 10%
        assert metrics['MAPE'] == pytest.approx(10.0)

    def test_mape_skips_zero_targets(self):
        """A zero target must not produce inf or a divide-by-zero warning"""
        y_true = np.array([100.0, 0.0])
        y_pred = np.array([110.0, 5.0])

        with np.errstate(divide='raise', invalid='raise'):
            metrics = calculate_metrics(y_true, y_pred)

        assert np.isfinite(metrics['MAPE'])
        # only the non-zero row contributes: |10/100| = 10%
        assert metrics['MAPE'] == pytest.approx(10.0)

    def test_mape_is_nan_when_all_targets_zero(self):
        metrics = calculate_metrics(np.array([0.0, 0.0]), np.array([1.0, 2.0]))
        assert np.isnan(metrics['MAPE'])
        # the other metrics remain well defined
        assert metrics['MAE'] == pytest.approx(1.5)

    def test_accepts_lists(self):
        metrics = calculate_metrics([100.0, 200.0], [100.0, 200.0])
        assert metrics['R2'] == pytest.approx(1.0)


class TestPredictionIntervals:
    """calculate_prediction_intervals should bracket the predictions"""

    def test_bounds_straddle_predictions(self):
        y_pred = np.array([100.0, 200.0])
        residuals = np.array([-5.0, 0.0, 5.0])

        lower, upper = calculate_prediction_intervals(y_pred, residuals)

        assert np.all(lower < y_pred)
        assert np.all(upper > y_pred)
        # symmetric around the prediction
        np.testing.assert_allclose(y_pred - lower, upper - y_pred)

    def test_higher_confidence_widens_interval(self):
        y_pred = np.array([100.0])
        residuals = np.array([-5.0, 0.0, 5.0])

        narrow = calculate_prediction_intervals(y_pred, residuals, confidence=0.80)
        wide = calculate_prediction_intervals(y_pred, residuals, confidence=0.99)

        assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])

    def test_zero_residuals_give_zero_width(self):
        lower, upper = calculate_prediction_intervals(
            np.array([100.0]), np.zeros(5)
        )
        assert lower[0] == pytest.approx(100.0)
        assert upper[0] == pytest.approx(100.0)


class TestPerformanceReport:
    """PerformanceReport should collect train and test metrics"""

    def test_to_dict_round_trip(self):
        report = PerformanceReport("Test Model")
        report.add_train_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        report.add_test_metrics(np.array([1.0, 2.0]), np.array([1.1, 2.1]))

        payload = report.to_dict()

        assert payload['model_name'] == "Test Model"
        assert payload['train_metrics']['R2'] == pytest.approx(1.0)
        assert payload['test_metrics']['MAE'] == pytest.approx(0.1)

    def test_metrics_are_none_before_being_added(self):
        report = PerformanceReport("Empty")
        assert report.to_dict()['train_metrics'] is None
        assert report.to_dict()['test_metrics'] is None
