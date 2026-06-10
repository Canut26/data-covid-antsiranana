"""
tests/test_data_cleaning.py
---------------------------
Tests unitaires pour le pipeline de nettoyage des données.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from data_cleaning import (
    standardize_outcome,
    standardize_severity,
    parse_dates,
    add_derived_features,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame minimal reproduisant la structure réelle."""
    return pd.DataFrame({
        'Patient_ID':      [1, 2, 3, 4, 5],
        'Age_years':       [45, 62, 33, 71, 55],
        'Sex':             [1, 0, 1, 1, 0],
        'Outcome':         ['Survived', 'Died', 'Survived', 'Décès', 'Survived'],
        'COVID_Severity':  ['Sévère', 'Critique', 'Modérée', 'Critique', 'Sévère'],
        'Admission_Date':  ['2021-01-10', '2021-03-05', '2020-11-20', '2021-06-01', '2020-08-15'],
        'Discharge_Date':  ['2021-01-20', '2021-03-12', '2020-11-28', None, '2020-08-25'],
        # Comorbidities
        'Hypertension':            [1, 1, 0, 1, 0],
        'Diabetes_Mellitus':       [0, 1, 0, 1, 0],
        'Obesity':                 [0, 0, 1, 0, 0],
        'Heart_Disease':           [0, 1, 0, 1, 0],
        'COPD':                    [0, 0, 0, 1, 0],
        'Asthma':                  [0, 0, 0, 0, 0],
        'Chronic_Kidney_Disease':  [0, 1, 0, 0, 0],
        'History_of_Stroke':       [0, 0, 0, 1, 0],
        # Symptoms
        'Fever':       [1, 1, 0, 1, 1],
        'Cough':       [1, 1, 1, 1, 0],
        'Sputum':      [0, 0, 1, 0, 0],
        'Dyspnea':     [1, 1, 0, 1, 1],
        'Chest_Pain':  [0, 1, 0, 1, 0],
        'Arthralgia_Myalgia': [0, 0, 1, 0, 0],
        'Fatigue':     [1, 1, 1, 1, 1],
        'Rhinorrhea':  [0, 0, 1, 0, 0],
        # Biology
        'Neutrophils_G_L':  [6.5, 9.2, 4.1, 12.0, 5.5],
        'Lymphocytes_G_L':  [1.2, 0.5, 1.8, 0.3, 1.5],
    })


# ── Tests : standardize_outcome ───────────────────────────────────────────────

class TestStandardizeOutcome:

    def test_died_french_converted(self, sample_df):
        df = standardize_outcome(sample_df)
        assert 'Décès' not in df['Outcome'].values

    def test_binary_column_created(self, sample_df):
        df = standardize_outcome(sample_df)
        assert 'Outcome_binary' in df.columns

    def test_binary_values_are_0_or_1(self, sample_df):
        df = standardize_outcome(sample_df)
        assert set(df['Outcome_binary'].unique()).issubset({0, 1})

    def test_died_encoded_as_1(self, sample_df):
        df = standardize_outcome(sample_df)
        died_rows = df[df['Outcome'] == 'Died']
        assert (died_rows['Outcome_binary'] == 1).all()

    def test_survived_encoded_as_0(self, sample_df):
        df = standardize_outcome(sample_df)
        survived_rows = df[df['Outcome'] == 'Survived']
        assert (survived_rows['Outcome_binary'] == 0).all()

    def test_original_df_not_modified(self, sample_df):
        original_outcomes = sample_df['Outcome'].copy()
        standardize_outcome(sample_df)
        pd.testing.assert_series_equal(sample_df['Outcome'], original_outcomes)


# ── Tests : standardize_severity ─────────────────────────────────────────────

class TestStandardizeSeverity:

    def test_severity_ordinal_created(self, sample_df):
        df = standardize_severity(sample_df)
        assert 'Severity_ordinal' in df.columns

    def test_severity_values_correct(self, sample_df):
        df = standardize_severity(sample_df)
        expected = {'Modérée': 0, 'Sévère': 1, 'Critique': 2}
        for _, row in df.iterrows():
            assert row['Severity_ordinal'] == expected[row['COVID_Severity']]

    def test_severity_dtype_numeric(self, sample_df):
        df = standardize_severity(sample_df)
        assert pd.api.types.is_numeric_dtype(df['Severity_ordinal'])


# ── Tests : parse_dates ───────────────────────────────────────────────────────

class TestParseDates:

    def test_dates_are_datetime(self, sample_df):
        df = parse_dates(sample_df)
        assert pd.api.types.is_datetime64_any_dtype(df['Admission_Date'])
        assert pd.api.types.is_datetime64_any_dtype(df['Discharge_Date'])

    def test_invalid_dates_become_nat(self, sample_df):
        df = parse_dates(sample_df)
        # Row 3 has None Discharge_Date → should be NaT
        assert pd.isna(df.iloc[3]['Discharge_Date'])

    def test_valid_dates_parsed_correctly(self, sample_df):
        df = parse_dates(sample_df)
        assert df.iloc[0]['Admission_Date'] == pd.Timestamp('2021-01-10')


# ── Tests : add_derived_features ─────────────────────────────────────────────

class TestAddDerivedFeatures:

    def test_comorbidity_score_created(self, sample_df):
        df = add_derived_features(sample_df)
        assert 'Comorbidity_Score' in df.columns

    def test_comorbidity_score_correct(self, sample_df):
        df = add_derived_features(sample_df)
        # Patient 0: Hypertension=1 → score=1
        assert df.iloc[0]['Comorbidity_Score'] == 1
        # Patient 1: Hypertension + Diabetes + Heart_Disease + CKD = 4
        assert df.iloc[1]['Comorbidity_Score'] == 4

    def test_comorbidity_score_non_negative(self, sample_df):
        df = add_derived_features(sample_df)
        assert (df['Comorbidity_Score'] >= 0).all()

    def test_symptom_count_created(self, sample_df):
        df = add_derived_features(sample_df)
        assert 'Symptom_Count' in df.columns

    def test_symptom_count_range(self, sample_df):
        df = add_derived_features(sample_df)
        assert df['Symptom_Count'].between(0, 8).all()

    def test_nlr_created(self, sample_df):
        df = add_derived_features(sample_df)
        assert 'NLR' in df.columns

    def test_nlr_positive(self, sample_df):
        df = add_derived_features(sample_df)
        assert (df['NLR'].dropna() > 0).all()

    def test_nlr_zero_lymphocytes_is_nan(self, sample_df):
        """NLR doit être NaN si lymphocytes = 0 (division par zéro protégée)."""
        df_zero = sample_df.copy()
        df_zero.loc[0, 'Lymphocytes_G_L'] = 0
        df = add_derived_features(df_zero)
        assert pd.isna(df.iloc[0]['NLR'])

    def test_age_group_created(self, sample_df):
        df = add_derived_features(sample_df)
        assert 'Age_Group' in df.columns

    def test_age_group_categories(self, sample_df):
        df = add_derived_features(sample_df)
        valid_cats = {'<40 ans', '40-60 ans', '>60 ans'}
        actual = set(df['Age_Group'].astype(str).unique())
        assert actual.issubset(valid_cats)
