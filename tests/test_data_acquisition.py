"""Tests for the data acquisition module"""

import numpy as np
import pandas as pd
import pytest
import requests

from config import MACRO_INDICATORS
from data_acquisition import DataAcquisition


@pytest.fixture
def acquisition(tmp_path):
    """Acquisition object pointed at a temporary data directory"""
    obj = DataAcquisition()
    obj.data_path = tmp_path / 'AmesHousing.txt'
    obj.processed_path = tmp_path / 'ames_processed.csv'
    return obj


@pytest.fixture
def raw_frame():
    """Minimal frame with the columns the pipeline depends on"""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        'Order': range(20),
        'PID': range(1000, 1020),
        'Yr Sold': rng.integers(2006, 2011, 20),
        'Gr Liv Area': rng.integers(800, 2500, 20),
        'Neighborhood': rng.choice(['NAmes', 'CollgCr'], 20),
        'SalePrice': rng.normal(180000, 40000, 20),
    })


class TestDownload:
    def test_existing_file_is_reused(self, acquisition):
        acquisition.data_path.write_bytes(b'CACHED')

        def fail(*args, **kwargs):
            raise AssertionError("should not hit the network")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(requests, 'get', fail)
            assert acquisition.download_dataset() == acquisition.data_path

    def test_failure_leaves_the_existing_file_intact(self, acquisition, monkeypatch):
        """Regression test: a partial write used to clobber the dataset"""
        acquisition.data_path.write_bytes(b'GOOD DATA')

        def boom(url, timeout=None):
            raise requests.ConnectionError('network down')

        monkeypatch.setattr(requests, 'get', boom)

        with pytest.raises(RuntimeError, match="Could not download"):
            acquisition.download_dataset(force_download=True)

        assert acquisition.data_path.read_bytes() == b'GOOD DATA'

    def test_failure_leaves_no_partial_file(self, acquisition, monkeypatch):
        def boom(url, timeout=None):
            raise requests.ConnectionError('network down')

        monkeypatch.setattr(requests, 'get', boom)

        with pytest.raises(RuntimeError):
            acquisition.download_dataset(force_download=True)

        assert list(acquisition.data_path.parent.glob('*.part')) == []

    def test_error_names_both_attempts(self, acquisition, monkeypatch):
        def boom(url, timeout=None):
            raise requests.ConnectionError('network down')

        monkeypatch.setattr(requests, 'get', boom)

        with pytest.raises(RuntimeError) as excinfo:
            acquisition.download_dataset(force_download=True)

        assert 'primary' in str(excinfo.value)
        assert 'backup' in str(excinfo.value)

    def test_backup_is_used_when_primary_fails(self, acquisition, monkeypatch):
        calls = []

        class Response:
            content = b'FROM BACKUP'

            def raise_for_status(self):
                pass

        def get(url, timeout=None):
            calls.append(url)
            if len(calls) == 1:
                raise requests.ConnectionError('primary down')
            return Response()

        monkeypatch.setattr(requests, 'get', get)

        acquisition.download_dataset(force_download=True)

        assert len(calls) == 2
        assert acquisition.data_path.read_bytes() == b'FROM BACKUP'

    def test_success_replaces_the_file(self, acquisition, monkeypatch):
        acquisition.data_path.write_bytes(b'OLD')

        class Response:
            content = b'NEW'

            def raise_for_status(self):
                pass

        monkeypatch.setattr(requests, 'get', lambda url, timeout=None: Response())

        acquisition.download_dataset(force_download=True)

        assert acquisition.data_path.read_bytes() == b'NEW'


class TestMacroeconomicFeatures:
    def test_adds_the_expected_columns(self, acquisition, raw_frame):
        out = acquisition.add_macroeconomic_features(raw_frame)

        for column in ('Mortgage_Rate_30yr', 'CPI', 'Years_Since_Sale',
                       'Market_Adjustment_Factor'):
            assert column in out.columns

    def test_years_since_sale_is_relative_to_the_reference_year(self, acquisition):
        frame = pd.DataFrame({'Yr Sold': [2006, 2010], 'SalePrice': [1.0, 2.0]})

        out = acquisition.add_macroeconomic_features(frame)

        reference = MACRO_INDICATORS['reference_year']
        assert list(out['Years_Since_Sale']) == [reference - 2006, reference - 2010]

    def test_adjustment_factor_combines_both_terms(self, acquisition, raw_frame):
        out = acquisition.add_macroeconomic_features(raw_frame)

        expected = (MACRO_INDICATORS['cpi'] / 230.0) * (4.0 / MACRO_INDICATORS['mortgage_rate_30yr'])
        assert out['Market_Adjustment_Factor'].iloc[0] == pytest.approx(expected)

    def test_adjustment_factor_is_constant(self, acquisition, raw_frame):
        """It is derived from scalars, so it cannot vary by row"""
        out = acquisition.add_macroeconomic_features(raw_frame)
        assert out['Market_Adjustment_Factor'].nunique() == 1

    def test_missing_year_column_is_reported_clearly(self, acquisition):
        """Regression test: this raised a bare KeyError"""
        frame = pd.DataFrame({'SalePrice': [1.0, 2.0]})

        with pytest.raises(KeyError, match="Yr Sold"):
            acquisition.add_macroeconomic_features(frame)

    def test_does_not_mutate_input(self, acquisition, raw_frame):
        columns_before = list(raw_frame.columns)
        acquisition.add_macroeconomic_features(raw_frame)
        assert list(raw_frame.columns) == columns_before


class TestInitialCleaning:
    def test_drops_identifier_columns(self, acquisition, raw_frame):
        out = acquisition.initial_cleaning(raw_frame)

        assert 'Order' not in out.columns
        assert 'PID' not in out.columns
        assert 'SalePrice' in out.columns

    def test_tolerates_absent_identifier_columns(self, acquisition):
        frame = pd.DataFrame({'SalePrice': [1.0]})
        assert acquisition.initial_cleaning(frame).shape == (1, 1)


class TestSummary:
    def test_counts_rows_columns_and_types(self, acquisition, raw_frame):
        summary = acquisition.get_data_summary(raw_frame)

        assert summary['n_rows'] == len(raw_frame)
        assert summary['n_columns'] == raw_frame.shape[1]
        assert summary['n_categorical'] == 1
        assert summary['target_mean'] == pytest.approx(raw_frame['SalePrice'].mean())

    def test_target_stats_are_none_without_the_target(self, acquisition):
        summary = acquisition.get_data_summary(pd.DataFrame({'x': [1, 2]}))
        assert summary['target_mean'] is None
