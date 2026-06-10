"""
data_cleaning.py
----------------
Pipeline de nettoyage et préparation de la base de données COVID-19 Antsiranana.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'COVID19_Database_Variables.xlsx')
CLEAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'covid_clean.csv')


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine='openpyxl')
    print(f"[INFO] Données chargées : {df.shape[0]} patients, {df.shape[1]} variables")
    return df


def standardize_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonise la variable Outcome (gère 'Décès' vs 'Died')."""
    df = df.copy()
    df['Outcome'] = df['Outcome'].replace({'Décès': 'Died'})
    df['Outcome_binary'] = (df['Outcome'] == 'Died').astype(int)
    return df


def standardize_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Encode COVID_Severity en ordinal."""
    df = df.copy()
    severity_map = {'Modérée': 0, 'Sévère': 1, 'Critique': 2}
    df['Severity_ordinal'] = df['COVID_Severity'].map(severity_map)
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['Admission_Date', 'Discharge_Date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Comorbidity score
    comorbidities = ['Hypertension', 'Diabetes_Mellitus', 'Obesity', 'Heart_Disease',
                     'COPD', 'Asthma', 'Chronic_Kidney_Disease', 'History_of_Stroke']
    df['Comorbidity_Score'] = df[comorbidities].sum(axis=1)

    # Symptom count
    symptoms = ['Fever', 'Cough', 'Sputum', 'Dyspnea', 'Chest_Pain',
                'Arthralgia_Myalgia', 'Fatigue', 'Rhinorrhea']
    df['Symptom_Count'] = df[symptoms].sum(axis=1)

    # Neutrophil-to-Lymphocyte Ratio (NLR) — marker of systemic inflammation
    df['NLR'] = df['Neutrophils_G_L'] / df['Lymphocytes_G_L'].replace(0, np.nan)

    # Age groups
    df['Age_Group'] = pd.cut(df['Age_years'],
                              bins=[0, 40, 60, 100],
                              labels=['<40 ans', '40-60 ans', '>60 ans'])
    return df


def report_missing(df: pd.DataFrame) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[INFO] Aucune valeur manquante (hors Time_to_Death).")
    else:
        print("[INFO] Valeurs manquantes :")
        print(missing.to_string())


def clean_pipeline(path: str = RAW_PATH) -> pd.DataFrame:
    df = load_data(path)
    df = standardize_outcome(df)
    df = standardize_severity(df)
    df = parse_dates(df)
    df = add_derived_features(df)
    report_missing(df)
    return df


if __name__ == '__main__':
    df_clean = clean_pipeline()
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"[INFO] Données nettoyées sauvegardées → {CLEAN_PATH}")
    print(df_clean[['Patient_ID', 'Age_years', 'Sex', 'COVID_Severity',
                     'Outcome', 'Comorbidity_Score', 'NLR']].head(10))
