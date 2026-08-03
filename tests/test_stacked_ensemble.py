"""Tests for the stacked ensemble model"""

import numpy as np
import pandas as pd
import pytest

import config
from feature_engineering import FeatureEngineeringPipeline
from models.base_learners import get_base_learners
from models.stacked_ensemble import StackedEnsembleModel


@pytest.fixture
def data():
    """Small regression problem driven by one feature"""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f'f{i}' for i in range(4)])
    y = pd.Series(X['f0'] * 3 + rng.normal(0, 0.2, 120))
    return X, y


@pytest.fixture
def trained(data):
    X, y = data
    model = StackedEnsembleModel()
    model.train(X, y, list(X.columns))
    return model


class TestFittedState:
    """Built and fitted are different states"""

    def test_reports_not_trained_before_fitting(self):
        model = StackedEnsembleModel()
        model.build_model()
        assert model.get_model_info() == "Model not trained"

    def test_predict_rejects_a_built_but_unfitted_model(self, data):
        """Regression test: this raised an opaque AttributeError"""
        X, _ = data
        model = StackedEnsembleModel()
        model.build_model()

        with pytest.raises(ValueError, match="not trained"):
            model.predict(X)

    def test_base_predictions_reject_an_unfitted_model(self, data):
        X, _ = data
        model = StackedEnsembleModel()
        model.build_model()

        with pytest.raises(ValueError, match="not trained"):
            model.get_base_predictions(X)

    def test_predict_rejects_a_model_never_built(self, data):
        X, _ = data
        with pytest.raises(ValueError, match="not trained"):
            StackedEnsembleModel().predict(X)


class TestTrainedModel:
    def test_predicts_one_value_per_row(self, trained, data):
        X, _ = data
        assert trained.predict(X).shape == (len(X),)

    def test_learns_the_signal(self, trained, data):
        X, y = data
        assert np.corrcoef(trained.predict(X), y)[0, 1] > 0.9

    def test_model_info_lists_the_base_learners(self, trained):
        """Regression test: estimators_ was unpacked as (name, model) pairs"""
        info = trained.get_model_info()

        assert info['base_learners'] == ['xgboost', 'lightgbm', 'random_forest']
        assert info['meta_learner'] == 'BayesianRidge'
        assert info['n_features'] == 4

    def test_base_predictions_cover_every_learner(self, trained, data):
        """Regression test: this raised TypeError on every call"""
        X, _ = data

        predictions = trained.get_base_predictions(X)

        assert set(predictions) == {'xgboost', 'lightgbm', 'random_forest'}
        assert all(v.shape == (len(X),) for v in predictions.values())


class TestPersistence:
    def test_round_trips_through_disk(self, trained, data, tmp_path, monkeypatch):
        import models.stacked_ensemble as module
        monkeypatch.setattr(module, 'MODELS_DIR', tmp_path)
        X, _ = data

        trained.save_model('roundtrip.pkl')
        loaded = StackedEnsembleModel()
        loaded.load_model('roundtrip.pkl')

        np.testing.assert_allclose(loaded.predict(X), trained.predict(X))
        assert loaded.feature_names == trained.feature_names

    def test_stores_the_feature_pipeline(self, data, tmp_path, monkeypatch):
        """Regression test: encoders had to be refit at inference time"""
        import models.stacked_ensemble as module
        monkeypatch.setattr(module, 'MODELS_DIR', tmp_path)

        rng = np.random.default_rng(0)
        raw = pd.DataFrame({
            'OverallQual': rng.integers(1, 10, 120),
            'GrLivArea': rng.exponential(1500, 120),
            'Neighborhood': rng.choice(['A', 'B', 'C'], 120),
        })
        y = pd.Series(rng.normal(180000, 40000, 120))

        pipeline = FeatureEngineeringPipeline()
        X_transformed = pipeline.fit_transform(raw, y)
        model = StackedEnsembleModel()
        model.train(X_transformed, y, pipeline.get_feature_names())
        model.save_model('with_pipeline.pkl', feature_pipeline=pipeline)

        loaded = StackedEnsembleModel()
        loaded.load_model('with_pipeline.pkl')

        assert loaded.feature_pipeline is not None
        np.testing.assert_allclose(
            loaded.feature_pipeline.transform(raw.head(5)).values,
            pipeline.transform(raw.head(5)).values,
        )

    def test_legacy_model_without_a_pipeline_still_loads(self, trained, tmp_path, monkeypatch):
        import joblib
        import models.stacked_ensemble as module
        monkeypatch.setattr(module, 'MODELS_DIR', tmp_path)

        trained.save_model('legacy.pkl')
        payload = joblib.load(tmp_path / 'legacy.pkl')
        payload.pop('feature_pipeline')
        joblib.dump(payload, tmp_path / 'legacy.pkl')

        loaded = StackedEnsembleModel()
        loaded.load_model('legacy.pkl')

        assert loaded.feature_pipeline is None

    def test_saving_an_untrained_model_is_rejected(self):
        with pytest.raises(ValueError, match="No model to save"):
            StackedEnsembleModel().save_model()

    def test_loading_a_missing_file_is_rejected(self, tmp_path, monkeypatch):
        import models.stacked_ensemble as module
        monkeypatch.setattr(module, 'MODELS_DIR', tmp_path)

        with pytest.raises(FileNotFoundError):
            StackedEnsembleModel().load_model('nope.pkl')


class TestBaseLearners:
    def test_returns_the_three_configured_learners(self):
        learners = get_base_learners()
        assert [name for name, _ in learners] == ['xgboost', 'lightgbm', 'random_forest']

    def test_custom_params_override_defaults(self):
        learners = dict(get_base_learners({'xgboost': {'n_estimators': 7}}))
        assert learners['xgboost'].get_params()['n_estimators'] == 7

    def test_defaults_come_from_config(self):
        learners = dict(get_base_learners())
        assert learners['random_forest'].get_params()['random_state'] == config.MODEL_CONFIG['random_state']
