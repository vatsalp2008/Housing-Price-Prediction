"""Tests for the data preprocessing utilities"""

import numpy as np
import pandas as pd
import pytest

import config
from utils.preprocessing import DataPreprocessor, prepare_features_target


@pytest.fixture
def frame():
    """Small frame with a numeric gap and a categorical gap"""
    return pd.DataFrame({
        'numeric': [1.0, 2.0, np.nan, 4.0],
        'category': ['a', 'b', None, 'a'],
        'SalePrice': [100.0, 200.0, 300.0, 400.0],
    })


class TestHandleMissingValues:
    """Imputation must actually fill values and must not leak across frames"""

    def test_fills_numeric_and_categorical(self, frame):
        out = DataPreprocessor().handle_missing_values(frame)

        assert out.isnull().sum().sum() == 0
        # median of 1, 2, 4
        assert out['numeric'].iloc[2] == pytest.approx(2.0)
        # mode of a, b, a
        assert out['category'].iloc[2] == 'a'

    def test_does_not_mutate_input(self, frame):
        DataPreprocessor().handle_missing_values(frame)
        assert frame.isnull().sum().sum() == 2

    def test_works_under_copy_on_write(self, frame, monkeypatch):
        """Regression test: chained fillna(inplace=True) was a silent no-op here"""
        monkeypatch.setattr(pd.options.mode, 'copy_on_write', True)

        out = DataPreprocessor().handle_missing_values(frame)

        assert out.isnull().sum().sum() == 0

    def test_reuses_training_values_on_later_frames(self, frame):
        preprocessor = DataPreprocessor()
        preprocessor.handle_missing_values(frame, fit=True)

        unseen = pd.DataFrame({
            'numeric': [np.nan],
            'category': [None],
            'SalePrice': [500.0],
        })
        out = preprocessor.handle_missing_values(unseen, fit=False)

        # training median/mode, not anything derived from `unseen`
        assert out['numeric'].iloc[0] == pytest.approx(2.0)
        assert out['category'].iloc[0] == 'a'

    def test_handles_columns_complete_in_train_but_missing_later(self):
        """Regression test: values were only stored for columns with training gaps"""
        train = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'category': ['a', 'a', 'b'],
            'SalePrice': [100.0, 200.0, 300.0],
        })
        preprocessor = DataPreprocessor()
        preprocessor.handle_missing_values(train, fit=True)

        unseen = pd.DataFrame({
            'numeric': [np.nan],
            'category': [None],
            'SalePrice': [400.0],
        })
        out = preprocessor.handle_missing_values(unseen, fit=False)

        assert out.isnull().sum().sum() == 0

    def test_all_nan_numeric_column_does_not_store_nan(self):
        frame = pd.DataFrame({'empty': [np.nan, np.nan], 'SalePrice': [1.0, 2.0]})
        preprocessor = DataPreprocessor()

        out = preprocessor.handle_missing_values(frame, fit=True)

        assert not np.isnan(preprocessor.numeric_impute_values['empty'])
        assert out.isnull().sum().sum() == 0


class TestDetectOutliersIqr:
    """Outlier detection must cover every numeric dtype"""

    OUTLIER_VALUES = [100, 105, 110, 102, 108, 10000]

    @pytest.mark.parametrize('dtype', ['float64', 'int64', 'int32', 'float32', 'Int64'])
    def test_detects_outlier_across_dtypes(self, dtype):
        """Regression test: only float64/int64 were checked before"""
        frame = pd.DataFrame({'SalePrice': pd.Series(self.OUTLIER_VALUES, dtype=dtype)})

        mask = DataPreprocessor().detect_outliers_iqr(frame, ['SalePrice'])

        assert mask.sum() == 1
        assert bool(mask.iloc[-1]) is True

    def test_ignores_non_numeric_columns(self):
        frame = pd.DataFrame({'category': ['a', 'b', 'c']})
        mask = DataPreprocessor().detect_outliers_iqr(frame, ['category'])
        assert mask.sum() == 0

    def test_ignores_missing_columns(self):
        frame = pd.DataFrame({'SalePrice': [1.0, 2.0]})
        mask = DataPreprocessor().detect_outliers_iqr(frame, ['does_not_exist'])
        assert mask.sum() == 0

    def test_higher_threshold_flags_fewer_rows(self):
        frame = pd.DataFrame({'SalePrice': [100.0, 105.0, 110.0, 102.0, 108.0, 400.0]})
        preprocessor = DataPreprocessor()

        loose = preprocessor.detect_outliers_iqr(frame, ['SalePrice'], threshold=1.5).sum()
        strict = preprocessor.detect_outliers_iqr(frame, ['SalePrice'], threshold=10.0).sum()

        assert loose >= strict


class TestRemoveExtremeOutliers:
    def test_drops_extreme_target_rows(self):
        frame = pd.DataFrame({
            'SalePrice': [100.0] * 20 + [100000.0],
        })
        out = DataPreprocessor().remove_extreme_outliers(frame)
        assert len(out) < len(frame)

    def test_keeps_frame_when_nothing_extreme(self):
        frame = pd.DataFrame({'SalePrice': np.linspace(100, 200, 30)})
        out = DataPreprocessor().remove_extreme_outliers(frame)
        assert len(out) == len(frame)


class TestStratifiedSplit:
    """The split must honour MODEL_CONFIG and stay reproducible"""

    @pytest.fixture
    def frame(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            'x': rng.normal(size=200),
            'SalePrice': rng.normal(180000, 40000, 200),
        })

    def test_defaults_come_from_model_config(self, frame):
        """Regression test: test_size and random_state were hardcoded"""
        train, test = DataPreprocessor().stratified_split(frame)

        expected_test = int(len(frame) * config.MODEL_CONFIG['test_size'])
        assert len(test) == expected_test
        assert len(train) + len(test) == len(frame)

    def test_explicit_arguments_override_config(self, frame):
        _, test = DataPreprocessor().stratified_split(frame, test_size=0.5)
        assert len(test) == 100

    def test_helper_column_is_removed(self, frame):
        train, test = DataPreprocessor().stratified_split(frame)
        assert 'price_bin' not in train.columns
        assert 'price_bin' not in test.columns

    def test_is_reproducible(self, frame):
        first, _ = DataPreprocessor().stratified_split(frame, random_state=7)
        second, _ = DataPreprocessor().stratified_split(frame, random_state=7)
        pd.testing.assert_frame_equal(first, second)

    def test_does_not_mutate_input(self, frame):
        columns_before = list(frame.columns)
        DataPreprocessor().stratified_split(frame)
        assert list(frame.columns) == columns_before


class TestPrepareFeaturesTarget:
    def test_splits_target_off(self):
        frame = pd.DataFrame({'x': [1, 2], 'SalePrice': [100.0, 200.0]})
        X, y = prepare_features_target(frame)

        assert 'SalePrice' not in X.columns
        assert list(y) == [100.0, 200.0]
