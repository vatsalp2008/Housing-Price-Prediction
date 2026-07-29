"""Tests for the feature engineering pipeline"""

import numpy as np
import pandas as pd
import pytest

from feature_engineering import (
    BoxCoxTransformer,
    FeatureEngineeringPipeline,
    InteractionFeatureGenerator,
    TargetEncoder,
)


class TestInteractionFeatureGenerator:
    def test_creates_product_column(self):
        frame = pd.DataFrame({'OverallQual': [2, 3], 'GrLivArea': [10.0, 20.0]})

        out = InteractionFeatureGenerator([('OverallQual', 'GrLivArea')]).transform(frame)

        assert list(out['OverallQual_x_GrLivArea']) == [20.0, 60.0]

    def test_skips_pairs_with_absent_columns(self):
        frame = pd.DataFrame({'OverallQual': [2, 3]})

        out = InteractionFeatureGenerator([('OverallQual', 'Missing')]).transform(frame)

        assert 'OverallQual_x_Missing' not in out.columns

    def test_does_not_mutate_input(self):
        frame = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        InteractionFeatureGenerator([('a', 'b')]).transform(frame)
        assert list(frame.columns) == ['a', 'b']


class TestTargetEncoder:
    """Encoding must be informative without leaking a row's own target"""

    @pytest.fixture
    def data(self):
        rng = np.random.default_rng(0)
        categories = rng.choice(['A', 'B', 'C', 'D'], 200)
        levels = {'A': 100.0, 'B': 200.0, 'C': 300.0, 'D': 400.0}
        y = pd.Series([levels[c] + rng.normal(0, 10) for c in categories])
        X = pd.DataFrame({'Neighborhood': categories})
        return X, y

    def test_replaces_original_column(self, data):
        X, y = data
        out = TargetEncoder(columns=['Neighborhood']).fit_transform(X, y)

        assert 'Neighborhood' not in out.columns
        assert 'Neighborhood_encoded' in out.columns

    def test_out_of_fold_differs_from_full_data_encoding(self, data):
        """Regression test: training rows were encoded using their own target"""
        X, y = data
        encoder = TargetEncoder(columns=['Neighborhood'])

        out_of_fold = encoder.fit_transform(X, y)['Neighborhood_encoded']
        full_data = encoder.transform(X)['Neighborhood_encoded']

        assert not np.allclose(out_of_fold, full_data)

    def test_encoding_remains_predictive(self, data):
        X, y = data
        out = TargetEncoder(columns=['Neighborhood']).fit_transform(X, y)

        correlation = np.corrcoef(out['Neighborhood_encoded'], y)[0, 1]
        assert correlation > 0.8

    def test_no_missing_values_produced(self, data):
        X, y = data
        out = TargetEncoder(columns=['Neighborhood']).fit_transform(X, y)
        assert out.isnull().sum().sum() == 0

    def test_unseen_category_falls_back_to_global_mean(self, data):
        X, y = data
        encoder = TargetEncoder(columns=['Neighborhood']).fit(X, y)

        out = encoder.transform(pd.DataFrame({'Neighborhood': ['UNSEEN']}))

        assert out['Neighborhood_encoded'].iloc[0] == pytest.approx(y.mean())

    def test_smoothing_shrinks_towards_global_mean(self, data):
        X, y = data

        light = TargetEncoder(columns=['Neighborhood'], smoothing=0.01).fit(X, y)
        heavy = TargetEncoder(columns=['Neighborhood'], smoothing=1e6).fit(X, y)

        global_mean = y.mean()
        light_spread = max(abs(v - global_mean) for v in light.encodings['Neighborhood'].values())
        heavy_spread = max(abs(v - global_mean) for v in heavy.encodings['Neighborhood'].values())

        assert heavy_spread < light_spread

    def test_frame_too_small_to_split_still_works(self, data):
        X, y = data
        out = TargetEncoder(columns=['Neighborhood']).fit_transform(X.head(1), y.head(1))
        assert len(out) == 1

    def test_fit_transform_requires_target(self, data):
        X, _ = data
        with pytest.raises(ValueError):
            TargetEncoder(columns=['Neighborhood']).fit_transform(X)


class TestBoxCoxTransformer:
    """The transform must stay inside the domain where 1 + x > 0"""

    @pytest.fixture
    def skewed(self):
        rng = np.random.default_rng(0)
        return np.concatenate([rng.exponential(100, 90), [5000.0] * 10])

    def test_transforms_skewed_feature(self, skewed):
        frame = pd.DataFrame({'area': skewed})
        transformer = BoxCoxTransformer().fit(frame)

        assert 'area' in transformer.skewed_features
        out = transformer.transform(frame)
        assert not np.allclose(out['area'], frame['area'])

    def test_leaves_symmetric_feature_alone(self):
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({'balanced': rng.normal(500, 50, 200)})

        transformer = BoxCoxTransformer().fit(frame)

        assert 'balanced' not in transformer.skewed_features

    def test_skips_features_outside_the_domain(self, skewed):
        """Regression test: these became all-NaN columns"""
        frame = pd.DataFrame({'delta': -skewed})

        transformer = BoxCoxTransformer().fit(frame)
        out = transformer.transform(frame)

        assert 'delta' not in transformer.skewed_features
        assert out.isnull().sum().sum() == 0
        np.testing.assert_allclose(out['delta'], frame['delta'])

    def test_clips_out_of_domain_values_seen_only_later(self, skewed):
        transformer = BoxCoxTransformer().fit(pd.DataFrame({'area': skewed}))

        out = transformer.transform(pd.DataFrame({'area': [-5.0, 10.0, 100.0]}))

        assert out.isnull().sum().sum() == 0

    def test_lambda_is_applied(self, skewed):
        frame = pd.DataFrame({'area': skewed})

        low = BoxCoxTransformer(lmbda=0.15).fit(frame).transform(frame)
        high = BoxCoxTransformer(lmbda=0.9).fit(frame).transform(frame)

        assert not np.allclose(low['area'], high['area'])


class TestFeatureEngineeringPipeline:
    """The pipeline must produce an aligned, fully numeric feature matrix"""

    @pytest.fixture
    def split(self):
        rng = np.random.default_rng(0)
        n = 300
        frame = pd.DataFrame({
            'OverallQual': rng.integers(1, 10, n),
            'GrLivArea': rng.exponential(1500, n),
            'YearBuilt': rng.integers(1900, 2010, n),
            'TotalBsmtSF': rng.exponential(900, n),
            'Neighborhood': rng.choice(['NAmes', 'CollgCr', 'OldTown'], n),
            'Exterior1st': rng.choice(['VinylSd', 'Wd Sdng'], n),
        })
        y = pd.Series(rng.normal(180000, 50000, n))
        return frame.iloc[:240], y.iloc[:240], frame.iloc[240:], y.iloc[240:]

    def test_train_and_test_columns_align(self, split):
        X_train, y_train, X_test, _ = split
        pipeline = FeatureEngineeringPipeline()

        train_out = pipeline.fit_transform(X_train, y_train)
        test_out = pipeline.transform(X_test)

        assert list(train_out.columns) == list(test_out.columns)
        assert list(test_out.columns) == pipeline.get_feature_names()

    def test_output_is_fully_numeric(self, split):
        X_train, y_train, X_test, _ = split
        pipeline = FeatureEngineeringPipeline()

        train_out = pipeline.fit_transform(X_train, y_train)
        test_out = pipeline.transform(X_test)

        assert train_out.select_dtypes(include=[np.number]).shape[1] == train_out.shape[1]
        assert test_out.select_dtypes(include=[np.number]).shape[1] == test_out.shape[1]

    def test_no_missing_values_in_output(self, split):
        X_train, y_train, X_test, _ = split
        pipeline = FeatureEngineeringPipeline()

        assert pipeline.fit_transform(X_train, y_train).isnull().sum().sum() == 0
        assert pipeline.transform(X_test).isnull().sum().sum() == 0

    def test_unseen_category_maps_to_modal_training_value(self):
        """Regression test: the fallback used the alphabetically first class"""
        train = pd.DataFrame({'cat': ['zebra'] * 8 + ['alpha'] * 2})
        pipeline = FeatureEngineeringPipeline()
        pipeline._encode_categorical(train, fit=True)

        encoded = pipeline._encode_categorical(pd.DataFrame({'cat': ['UNSEEN']}), fit=False)
        label = pipeline.label_encoders['cat'].inverse_transform(encoded['cat'])[0]

        assert label == 'zebra'

    def test_feature_names_empty_before_fit(self):
        assert FeatureEngineeringPipeline().get_feature_names() == []
