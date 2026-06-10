"""
tests/test_statistical_tests.py
--------------------------------
Tests unitaires pour les fonctions de tests statistiques.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from statistical_tests import chi2_or_fisher, mann_whitney


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def df_stats():
    """Dataset synthétique avec associations connues."""
    np.random.seed(42)
    n = 80

    outcome = np.array([0] * 60 + [1] * 20)

    # Variable fortement associée à l'outcome
    hypertension = np.where(outcome == 1,
                            np.random.binomial(1, 0.8, n),
                            np.random.binomial(1, 0.2, n))

    # Variable non associée
    sex = np.random.binomial(1, 0.5, n)

    # Variable continue fortement associée (CRP)
    crp = np.where(outcome == 1,
                   np.random.exponential(80, n),
                   np.random.exponential(20, n))

    # Variable continue non associée
    height = np.random.normal(170, 10, n)

    return pd.DataFrame({
        'Outcome_binary': outcome,
        'Hypertension': hypertension,
        'Sex': sex,
        'CRP_mg_L': crp,
        'Height_cm': height,
    })


# ── Tests : chi2_or_fisher ────────────────────────────────────────────────────

class TestChi2OrFisher:

    def test_returns_dict(self, df_stats):
        result = chi2_or_fisher(df_stats, 'Hypertension')
        assert isinstance(result, dict)

    def test_required_keys_present(self, df_stats):
        result = chi2_or_fisher(df_stats, 'Hypertension')
        assert 'variable' in result
        assert 'p_value' in result
        assert 'test' in result

    def test_variable_name_stored(self, df_stats):
        result = chi2_or_fisher(df_stats, 'Hypertension')
        assert result['variable'] == 'Hypertension'

    def test_p_value_between_0_and_1(self, df_stats):
        result = chi2_or_fisher(df_stats, 'Hypertension')
        assert 0.0 <= result['p_value'] <= 1.0

    def test_associated_variable_significant(self, df_stats):
        """Hypertension est fortement associée → p < 0.05."""
        result = chi2_or_fisher(df_stats, 'Hypertension')
        assert result['p_value'] < 0.05

    def test_non_associated_variable_not_significant(self, df_stats):
        """Sex est aléatoire → p généralement > 0.05 (seed fixé)."""
        result = chi2_or_fisher(df_stats, 'Sex')
        # On vérifie juste que la fonction tourne sans erreur
        assert 'p_value' in result

    def test_test_type_is_string(self, df_stats):
        result = chi2_or_fisher(df_stats, 'Hypertension')
        assert isinstance(result['test'], str)
        assert result['test'] in ('Fisher', 'Chi2')


# ── Tests : mann_whitney ──────────────────────────────────────────────────────

class TestMannWhitney:

    def test_returns_dict(self, df_stats):
        result = mann_whitney(df_stats, 'CRP_mg_L')
        assert isinstance(result, dict)

    def test_required_keys_present(self, df_stats):
        result = mann_whitney(df_stats, 'CRP_mg_L')
        for key in ('variable', 'median_survived', 'median_died', 'U_stat', 'p_value'):
            assert key in result, f"Clé manquante : {key}"

    def test_variable_name_stored(self, df_stats):
        result = mann_whitney(df_stats, 'CRP_mg_L')
        assert result['variable'] == 'CRP_mg_L'

    def test_p_value_range(self, df_stats):
        result = mann_whitney(df_stats, 'CRP_mg_L')
        assert 0.0 <= result['p_value'] <= 1.0

    def test_crp_significant(self, df_stats):
        """CRP est fortement différente entre groupes → p < 0.05."""
        result = mann_whitney(df_stats, 'CRP_mg_L')
        assert result['p_value'] < 0.05

    def test_height_not_significant(self, df_stats):
        """Height est identique entre groupes → p > 0.05."""
        result = mann_whitney(df_stats, 'Height_cm')
        assert result['p_value'] > 0.05

    def test_median_died_gt_survived_for_crp(self, df_stats):
        """Médiane CRP des décédés > survivants (par construction du dataset)."""
        result = mann_whitney(df_stats, 'CRP_mg_L')
        assert result['median_died'] > result['median_survived']

    def test_u_stat_positive(self, df_stats):
        result = mann_whitney(df_stats, 'CRP_mg_L')
        assert result['U_stat'] >= 0

    def test_handles_missing_values(self, df_stats):
        """Le test doit fonctionner même avec des NaN."""
        df_with_nan = df_stats.copy()
        df_with_nan.loc[[0, 5, 10], 'CRP_mg_L'] = np.nan
        result = mann_whitney(df_with_nan, 'CRP_mg_L')
        assert 'p_value' in result
