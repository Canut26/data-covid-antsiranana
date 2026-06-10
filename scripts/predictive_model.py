"""
predictive_model.py
-------------------
Modèles prédictifs de mortalité COVID-19 (Régression Logistique + Random Forest).
Inclut : sélection de variables, validation croisée, courbe ROC, importance des features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import clean_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURES = [
    'Age_years', 'qSOFA_Score', 'CRP_mg_L', 'D_Dimers_ng_mL', 'NLR',
    'Severity_ordinal', 'Comorbidity_Score', 'Symptom_Count',
    'Respiratory_Rate_cpm', 'Creatinine_umol_L', 'Lymphocytes_G_L',
    'Diastolic_Blood_Pressure_mmHg', 'Heart_Rate_bpm', 'Urea_mmol_L',
    'Respiratory_Distress', 'Pneumonia', 'CT_Bilateral_Involvement'
]
TARGET = 'Outcome_binary'


def prepare_data(df: pd.DataFrame) -> tuple:
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].copy()
    y = df[TARGET].copy()
    return X, y, available


def build_logistic_pipeline() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])


def build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=5,
                                        min_samples_leaf=3, random_state=42,
                                        class_weight='balanced'))
    ])


def cross_validate_model(pipeline, X: pd.DataFrame, y: pd.Series, name: str) -> dict:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')
    acc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    print(f"\n[{name}] Validation croisée 5-fold :")
    print(f"  AUC-ROC  : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"  Accuracy : {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    return {'model': name, 'auc_mean': auc_scores.mean(), 'auc_std': auc_scores.std(),
            'acc_mean': acc_scores.mean(), 'acc_std': acc_scores.std()}


def plot_roc_curves(df_plot: pd.DataFrame, fig_path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'Logistic Regression': '#1565C0', 'Random Forest': '#2E7D32'}
    for _, row in df_plot.iterrows():
        ax.plot(row['fpr'], row['tpr'],
                label=f"{row['model']} (AUC = {row['auc']:.3f})",
                color=colors.get(row['model'], 'gray'), linewidth=2.5)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Aléatoire')
    ax.set_xlabel("1 - Spécificité (FPR)", fontsize=12)
    ax.set_ylabel("Sensibilité (TPR)", fontsize=12)
    ax.set_title("Courbes ROC — Prédiction de mortalité COVID-19\nAntsiranana, Madagascar",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {os.path.basename(fig_path)}")


def plot_feature_importance(rf_pipeline, feature_names: list, fig_path: str):
    rf = rf_pipeline.named_steps['clf']
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))

    labels_fr = {
        'Age_years': 'Âge', 'qSOFA_Score': 'qSOFA', 'CRP_mg_L': 'CRP',
        'D_Dimers_ng_mL': 'D-Dimères', 'NLR': 'NLR', 'Severity_ordinal': 'Sévérité',
        'Comorbidity_Score': 'Comorbidités', 'Symptom_Count': 'Symptômes',
        'Respiratory_Rate_cpm': 'FR (cpm)', 'Creatinine_umol_L': 'Créatinine',
        'Lymphocytes_G_L': 'Lymphocytes', 'Diastolic_Blood_Pressure_mmHg': 'PAD',
        'Heart_Rate_bpm': 'FC (bpm)', 'Urea_mmol_L': 'Urée',
        'Respiratory_Distress': 'Détresse resp.', 'Pneumonia': 'Pneumonie',
        'CT_Bilateral_Involvement': 'TDM bilatéral'
    }

    top_feats = [feature_names[i] for i in indices[:top_n]]
    top_imp = [importances[i] for i in indices[:top_n]]
    top_labels = [labels_fr.get(f, f) for f in top_feats]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, top_n))
    ax.barh(range(top_n), top_imp[::-1], color=colors, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1])
    ax.set_xlabel("Importance (Gini)")
    ax.set_title("Importance des variables — Random Forest\nPrédiction de mortalité COVID-19",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {os.path.basename(fig_path)}")


def run_full_pipeline():
    df = clean_pipeline()
    X, y, features = prepare_data(df)

    print(f"\n[INFO] Features utilisées ({len(features)}) : {features}")
    print(f"[INFO] Distribution cible : {y.value_counts().to_dict()}")

    lr_pipe = build_logistic_pipeline()
    rf_pipe = build_rf_pipeline()

    cv_results = []
    cv_results.append(cross_validate_model(lr_pipe, X, y, 'Logistic Regression'))
    cv_results.append(cross_validate_model(rf_pipe, X, y, 'Random Forest'))

    # Fit final models on full data for ROC
    roc_data = []
    for name, pipe in [('Logistic Regression', lr_pipe), ('Random Forest', rf_pipe)]:
        pipe.fit(X, y)
        y_prob = pipe.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        roc_data.append({'model': name, 'fpr': fpr, 'tpr': tpr, 'auc': auc})

    df_roc = pd.DataFrame(roc_data)
    plot_roc_curves(df_roc, os.path.join(FIGURES_DIR, '08_roc_curves.png'))
    plot_feature_importance(rf_pipe, features, os.path.join(FIGURES_DIR, '09_feature_importance.png'))

    # Classification report
    rf_pipe.fit(X, y)
    y_pred = rf_pipe.predict(X)
    print("\n[Random Forest] Rapport de classification (training) :")
    print(classification_report(y, y_pred, target_names=['Survivant', 'Décédé']))

    # Summary
    print("\n=== RÉSUMÉ DES PERFORMANCES ===")
    for r in cv_results:
        print(f"  {r['model']:25s} | AUC {r['auc_mean']:.3f} ± {r['auc_std']:.3f} "
              f"| Acc {r['acc_mean']:.3f} ± {r['acc_std']:.3f}")

    return cv_results


if __name__ == '__main__':
    run_full_pipeline()
