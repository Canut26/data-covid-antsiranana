"""
tests/test_predictive_model.py
-------------------------------
Tests unitaires pour le pipeline de modélisation prédictive.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from predictive_model import (
    prepare_data,
    build_logistic_pipeline,
    build_rf_pipeline,
    FEATURES,
    TARGET,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def df_model():
    """Dataset synthétique avec toutes les features du modèle."""
    np.random.seed(0)
    n = 60
    outcome = np.array([0] * 45 + [1] * 15)

    data = {
        'Age_years':                    np.random.randint(20, 85, n),
        'qSOFA_Score':                  np.random.randint(0, 4, n),
        'CRP_mg_L':                     np.random.exponential(40, n),
        'D_Dimers_ng_mL':               np.random.exponential(600, n),
        'NLR':                          np.random.exponential(5, n) + 1,
        'Severity_ordinal':             np.random.randint(0, 3, n),
        'Comorbidity_Score':            np.random.randint(0, 5, n),
        'Symptom_Count':                np.random.randint(0, 8, n),
        'Respiratory_Rate_cpm':         np.random.randint(14, 40, n),
        'Creatinine_umol_L':            np.random.exponential(100, n) + 40,
        'Lymphocytes_G_L':              np.random.exponential(1.2, n) + 0.2,
        'Diastolic_Blood_Pressure_mmHg': np.random.randint(50, 110, n),
        'Heart_Rate_bpm':               np.random.randint(50, 140, n),
        'Urea_mmol_L':                  np.random.exponential(7, n) + 2,
        'Respiratory_Distress':         np.random.binomial(1, 0.4, n),
        'Pneumonia':                    np.random.binomial(1, 0.5, n),
        'CT_Bilateral_Involvement':     np.random.binomial(1, 0.6, n),
        'Outcome_binary':               outcome,
    }
    return pd.DataFrame(data)


# ── Tests : prepare_data ──────────────────────────────────────────────────────

class TestPrepareData:

    def test_returns_three_items(self, df_model):
        X, y, feats = prepare_data(df_model)
        assert X is not None and y is not None and feats is not None

    def test_X_shape_correct(self, df_model):
        X, y, feats = prepare_data(df_model)
        assert X.shape[0] == len(df_model)
        assert X.shape[1] == len(feats)

    def test_y_is_binary(self, df_model):
        _, y, _ = prepare_data(df_model)
        assert set(y.unique()).issubset({0, 1})

    def test_only_available_features_returned(self, df_model):
        """Seules les features présentes dans df doivent être retournées."""
        df_partial = df_model.drop(columns=['D_Dimers_ng_mL', 'NLR'])
        _, _, feats = prepare_data(df_partial)
        assert 'D_Dimers_ng_mL' not in feats
        assert 'NLR' not in feats

    def test_all_features_are_in_X_columns(self, df_model):
        X, _, feats = prepare_data(df_model)
        assert list(X.columns) == feats


# ── Tests : build_logistic_pipeline ──────────────────────────────────────────

class TestLogisticPipeline:

    def test_pipeline_has_three_steps(self):
        pipe = build_logistic_pipeline()
        assert len(pipe.steps) == 3

    def test_pipeline_steps_names(self):
        pipe = build_logistic_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert 'imputer' in step_names
        assert 'scaler' in step_names
        assert 'clf' in step_names

    def test_pipeline_fits_without_error(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_logistic_pipeline()
        pipe.fit(X, y)

    def test_pipeline_predict_shape(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_logistic_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_pipeline_predict_proba_shape(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_logistic_pipeline()
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)
        assert probas.shape == (len(y), 2)

    def test_probabilities_sum_to_one(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_logistic_pipeline()
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)


# ── Tests : build_rf_pipeline ─────────────────────────────────────────────────

class TestRFPipeline:

    def test_pipeline_has_two_steps(self):
        pipe = build_rf_pipeline()
        assert len(pipe.steps) == 2

    def test_pipeline_steps_names(self):
        pipe = build_rf_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert 'imputer' in step_names
        assert 'clf' in step_names

    def test_rf_fits_without_error(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_rf_pipeline()
        pipe.fit(X, y)

    def test_rf_predict_binary(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_rf_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert set(preds).issubset({0, 1})

    def test_rf_predict_proba_valid(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_rf_pipeline()
        pipe.fit(X, y)
        probas = pipe.predict_proba(X)
        assert probas.shape == (len(y), 2)
        assert (probas >= 0).all() and (probas <= 1).all()

    def test_rf_handles_missing_values(self, df_model):
        """Le pipeline doit gérer les NaN via l'imputer."""
        df_nan = df_model.copy()
        df_nan.loc[[0, 1, 2], 'CRP_mg_L'] = np.nan
        df_nan.loc[[5, 6], 'NLR'] = np.nan
        X, y, _ = prepare_data(df_nan)
        pipe = build_rf_pipeline()
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)

    def test_feature_importances_available(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_rf_pipeline()
        pipe.fit(X, y)
        rf = pipe.named_steps['clf']
        importances = rf.feature_importances_
        assert len(importances) == X.shape[1]
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_feature_importances_non_negative(self, df_model):
        X, y, _ = prepare_data(df_model)
        pipe = build_rf_pipeline()
        pipe.fit(X, y)
        rf = pipe.named_steps['clf']
        assert (rf.feature_importances_ >= 0).all()
