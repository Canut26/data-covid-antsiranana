"""
statistical_tests.py
---------------------
Tests statistiques comparatifs : Survivants vs Décédés, et par niveau de sévérité.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from data_cleaning import clean_pipeline


def chi2_or_fisher(df: pd.DataFrame, var: str, group: str = 'Outcome_binary') -> dict:
    """Chi² ou Fisher exact pour une variable binaire."""
    table = pd.crosstab(df[var], df[group])
    if table.shape == (2, 2):
        _, p_fisher = stats.fisher_exact(table)
        chi2, p_chi2, _, _ = stats.chi2_contingency(table)
        return {'variable': var, 'chi2': round(chi2, 3), 'p_value': round(p_fisher, 4),
                'test': 'Fisher', 'n_died': table.get(1, pd.Series()).get(1, 0)}
    else:
        chi2, p, _, _ = stats.chi2_contingency(table)
        return {'variable': var, 'chi2': round(chi2, 3), 'p_value': round(p, 4), 'test': 'Chi2'}


def mann_whitney(df: pd.DataFrame, var: str, group: str = 'Outcome_binary') -> dict:
    """Test de Mann-Whitney U pour variable continue."""
    g0 = df.loc[df[group] == 0, var].dropna()
    g1 = df.loc[df[group] == 1, var].dropna()
    stat, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
    return {
        'variable': var,
        'median_survived': round(g0.median(), 2),
        'median_died': round(g1.median(), 2),
        'U_stat': round(stat, 1),
        'p_value': round(p, 4)
    }


def full_comparative_analysis(df: pd.DataFrame) -> tuple:
    """Lance tous les tests comparatifs Survivants vs Décédés."""

    binary_vars = [
        'Sex', 'Hypertension', 'Diabetes_Mellitus', 'Obesity', 'Heart_Disease',
        'COPD', 'Chronic_Kidney_Disease', 'Fever', 'Cough', 'Dyspnea',
        'Chest_Pain', 'Fatigue', 'CT_Lung_Lesions', 'CT_Ground_Glass_Opacities',
        'CT_Bilateral_Involvement', 'Lymphopenia', 'Elevated_CRP',
        'Respiratory_Distress', 'Pneumonia', 'Respiratory_Complications'
    ]

    continuous_vars = [
        'Age_years', 'Length_of_Stay_days', 'Symptom_to_Admission_Delay_days',
        'qSOFA_Score', 'Diastolic_Blood_Pressure_mmHg', 'Temperature_C',
        'Heart_Rate_bpm', 'Respiratory_Rate_cpm',
        'Hemoglobin_g_dL', 'White_Blood_Cells_G_L', 'Neutrophils_G_L',
        'Lymphocytes_G_L', 'Platelets_G_L', 'CRP_mg_L',
        'Creatinine_umol_L', 'Urea_mmol_L', 'Glucose_mmol_L',
        'D_Dimers_ng_mL', 'NLR', 'Comorbidity_Score'
    ]

    cat_results = []
    for v in binary_vars:
        if v in df.columns:
            try:
                cat_results.append(chi2_or_fisher(df, v))
            except Exception:
                pass

    cont_results = []
    for v in continuous_vars:
        if v in df.columns:
            try:
                cont_results.append(mann_whitney(df, v))
            except Exception:
                pass

    df_cat = pd.DataFrame(cat_results).sort_values('p_value')
    df_cont = pd.DataFrame(cont_results).sort_values('p_value')

    return df_cat, df_cont


def severity_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Kruskal-Wallis entre les 3 niveaux de sévérité."""
    groups = df.groupby('COVID_Severity')
    continuous_vars = ['Age_years', 'CRP_mg_L', 'D_Dimers_ng_mL', 'NLR',
                       'qSOFA_Score', 'Length_of_Stay_days']
    results = []
    for v in continuous_vars:
        if v not in df.columns:
            continue
        group_data = [g[v].dropna().values for _, g in groups]
        stat, p = stats.kruskal(*group_data)
        medians = {name: round(g[v].median(), 2) for name, g in groups}
        results.append({'variable': v, 'H_stat': round(stat, 2), 'p_value': round(p, 4), **medians})
    return pd.DataFrame(results).sort_values('p_value')


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    df = clean_pipeline()

    print("\n=== ANALYSE COMPARATIVE : SURVIVANTS vs DÉCÉDÉS ===\n")
    df_cat, df_cont = full_comparative_analysis(df)

    print("--- Variables catégorielles (p < 0.05) ---")
    sig_cat = df_cat[df_cat['p_value'] < 0.05]
    print(sig_cat.to_string(index=False))

    print("\n--- Variables continues (p < 0.05) ---")
    sig_cont = df_cont[df_cont['p_value'] < 0.05]
    print(sig_cont.to_string(index=False))

    print("\n=== ANALYSE PAR SÉVÉRITÉ (Kruskal-Wallis) ===\n")
    df_sev = severity_analysis(df)
    print(df_sev.to_string(index=False))

    # Export
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    os.makedirs(out_dir, exist_ok=True)
    df_cat.to_csv(os.path.join(out_dir, 'categorical_tests.csv'), index=False)
    df_cont.to_csv(os.path.join(out_dir, 'continuous_tests.csv'), index=False)
    df_sev.to_csv(os.path.join(out_dir, 'severity_analysis.csv'), index=False)
    print("\n[INFO] Résultats exportés dans reports/")
