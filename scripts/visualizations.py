"""
visualizations.py
-----------------
Génération de toutes les figures pour l'analyse COVID-19 Antsiranana.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from data_cleaning import clean_pipeline

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Style global ─────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
COLORS = {'Survived': '#2196F3', 'Died': '#F44336',
          'Modérée': '#4CAF50', 'Sévère': '#FF9800', 'Critique': '#F44336'}
FIG_DPI = 150


def save(fig, name: str):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"[SAVED] {name}")


# ── Figure 1 : Vue d'ensemble démographique ──────────────────────────────────
def fig_demographics(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Profil démographique des patients COVID-19 – Antsiranana",
                 fontsize=14, fontweight='bold', y=1.02)

    # Âge
    axes[0].hist(df['Age_years'], bins=15, color='#2196F3', edgecolor='white', linewidth=0.8)
    axes[0].axvline(df['Age_years'].median(), color='red', linestyle='--', label=f"Médiane : {df['Age_years'].median():.1f} ans")
    axes[0].set_title("Distribution de l'âge")
    axes[0].set_xlabel("Âge (années)")
    axes[0].set_ylabel("Nombre de patients")
    axes[0].legend()

    # Sexe
    sex_counts = df['Sex'].value_counts()
    axes[1].pie(sex_counts, labels=['Hommes', 'Femmes'], colors=['#42A5F5', '#EF9A9A'],
                autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
    axes[1].set_title("Répartition par sexe")

    # Issue
    outcome_counts = df['Outcome'].replace({'Décès': 'Died'}).value_counts()
    bars = axes[2].bar(outcome_counts.index, outcome_counts.values,
                       color=[COLORS.get(x, '#90A4AE') for x in outcome_counts.index],
                       edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, outcome_counts.values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'n={val}', ha='center', fontsize=11, fontweight='bold')
    axes[2].set_title("Issue clinique")
    axes[2].set_ylabel("Nombre de patients")

    plt.tight_layout()
    save(fig, '01_demographics.png')


# ── Figure 2 : Sévérité par groupe d'âge et sexe ────────────────────────────
def fig_severity_breakdown(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Sévérité COVID-19 selon l'âge et le sexe", fontsize=13, fontweight='bold')

    sev_age = pd.crosstab(df['Age_Group'], df['COVID_Severity'], normalize='index') * 100
    sev_age[['Modérée', 'Sévère', 'Critique']].plot(
        kind='bar', ax=axes[0], color=['#4CAF50', '#FF9800', '#F44336'],
        edgecolor='white', linewidth=0.8)
    axes[0].set_title("Sévérité par groupe d'âge (%)")
    axes[0].set_xlabel("Groupe d'âge")
    axes[0].set_ylabel("% de patients")
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].legend(title='Sévérité')

    sev_sex = pd.crosstab(df['Sex'], df['COVID_Severity'], normalize='index') * 100
    sev_sex[['Modérée', 'Sévère', 'Critique']].plot(
        kind='bar', ax=axes[1], color=['#4CAF50', '#FF9800', '#F44336'],
        edgecolor='white', linewidth=0.8)
    axes[1].set_title("Sévérité par sexe (%)")
    axes[1].set_xlabel("Sexe")
    axes[1].set_ylabel("% de patients")
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].legend(title='Sévérité')

    plt.tight_layout()
    save(fig, '02_severity_breakdown.png')


# ── Figure 3 : Prévalence des comorbidités ───────────────────────────────────
def fig_comorbidities(df: pd.DataFrame):
    comorbidities = {
        'Hypertension': 'Hypertension',
        'Diabetes_Mellitus': 'Diabète',
        'Obesity': 'Obésité',
        'Heart_Disease': 'Maladie cardiaque',
        'COPD': 'BPCO',
        'Asthma': 'Asthme',
        'Chronic_Kidney_Disease': 'IRC',
        'History_of_Stroke': 'ATCD AVC',
        'Newly_Diagnosed_Diabetes': 'Diabète nouveau'
    }
    prev = {label: df[col].mean() * 100 for col, label in comorbidities.items() if col in df.columns}
    prev_sorted = dict(sorted(prev.items(), key=lambda x: x[1], reverse=True))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(prev_sorted.keys()), list(prev_sorted.values()),
                   color='#5C6BC0', edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, prev_sorted.values()):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=10)
    ax.set_xlabel("Prévalence (%)")
    ax.set_title("Prévalence des comorbidités chez les patients COVID-19\nAntsiranana, Madagascar",
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, max(prev_sorted.values()) + 10)
    plt.tight_layout()
    save(fig, '03_comorbidities.png')


# ── Figure 4 : Biomarqueurs Survivants vs Décédés ────────────────────────────
def fig_biomarkers(df: pd.DataFrame):
    biomarkers = {
        'CRP_mg_L': 'CRP (mg/L)',
        'D_Dimers_ng_mL': 'D-Dimères (ng/mL)',
        'NLR': 'Ratio N/L',
        'Lymphocytes_G_L': 'Lymphocytes (G/L)',
        'White_Blood_Cells_G_L': 'Leucocytes (G/L)',
        'Creatinine_umol_L': 'Créatinine (µmol/L)'
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    fig.suptitle("Biomarqueurs biologiques : Survivants vs Décédés",
                 fontsize=13, fontweight='bold')

    for i, (col, label) in enumerate(biomarkers.items()):
        if col not in df.columns:
            continue
        data_plot = df[[col, 'Outcome']].dropna()
        data_plot['Outcome'] = data_plot['Outcome'].replace({'Décès': 'Died'})
        sns.boxplot(data=data_plot, x='Outcome', y=col, ax=axes[i],
                    palette={'Survived': '#42A5F5', 'Died': '#EF5350'},
                    order=['Survived', 'Died'])
        axes[i].set_title(label)
        axes[i].set_xlabel('')
        axes[i].set_ylabel(label)

    plt.tight_layout()
    save(fig, '04_biomarkers_outcome.png')


# ── Figure 5 : Heatmap de corrélation ───────────────────────────────────────
def fig_correlation(df: pd.DataFrame):
    cols = ['Age_years', 'qSOFA_Score', 'CRP_mg_L', 'D_Dimers_ng_mL', 'NLR',
            'Length_of_Stay_days', 'Severity_ordinal', 'Outcome_binary',
            'Comorbidity_Score', 'Symptom_Count', 'Creatinine_umol_L']
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    labels = {
        'Age_years': 'Âge', 'qSOFA_Score': 'qSOFA', 'CRP_mg_L': 'CRP',
        'D_Dimers_ng_mL': 'D-Dimères', 'NLR': 'NLR', 'Length_of_Stay_days': 'DMS',
        'Severity_ordinal': 'Sévérité', 'Outcome_binary': 'Décès',
        'Comorbidity_Score': 'Comorbidités', 'Symptom_Count': 'Symptômes',
        'Creatinine_umol_L': 'Créatinine'
    }
    corr.index = [labels.get(c, c) for c in corr.index]
    corr.columns = [labels.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Coefficient de corrélation'})
    ax.set_title("Matrice de corrélation — Variables cliniques et biologiques",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save(fig, '05_correlation_heatmap.png')


# ── Figure 6 : Symptômes à l'admission ──────────────────────────────────────
def fig_symptoms(df: pd.DataFrame):
    symptoms = {
        'Fever': 'Fièvre', 'Cough': 'Toux', 'Dyspnea': 'Dyspnée',
        'Fatigue': 'Fatigue', 'Chest_Pain': 'Douleur thoracique',
        'Sputum': 'Expectoration', 'Arthralgia_Myalgia': 'Arthralgie/Myalgie',
        'Rhinorrhea': 'Rhinorrhée'
    }
    survived = df[df['Outcome_binary'] == 0]
    died = df[df['Outcome_binary'] == 1]

    prev_surv = {label: survived[col].mean() * 100 for col, label in symptoms.items() if col in df.columns}
    prev_died = {label: died[col].mean() * 100 for col, label in symptoms.items() if col in df.columns}

    labels_list = list(prev_surv.keys())
    x = np.arange(len(labels_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, prev_surv.values(), width, label='Survivants', color='#42A5F5', edgecolor='white')
    ax.bar(x + width / 2, prev_died.values(), width, label='Décédés', color='#EF5350', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=30, ha='right')
    ax.set_ylabel("Prévalence (%)")
    ax.set_title("Prévalence des symptômes à l'admission : Survivants vs Décédés",
                 fontsize=12, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    save(fig, '06_symptoms_comparison.png')


# ── Figure 7 : qSOFA et sévérité ────────────────────────────────────────────
def fig_qsofa(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.countplot(data=df, x='qSOFA_Score', hue='COVID_Severity', ax=axes[0],
                  palette={'Modérée': '#4CAF50', 'Sévère': '#FF9800', 'Critique': '#F44336'})
    axes[0].set_title("Distribution du score qSOFA par sévérité")
    axes[0].set_xlabel("Score qSOFA")
    axes[0].set_ylabel("Nombre de patients")

    sns.violinplot(data=df, x='Outcome', y='qSOFA_Score', ax=axes[1],
                   palette={'Survived': '#42A5F5', 'Died': '#EF5350'},
                   order=['Survived', 'Died'])
    axes[1].set_title("Score qSOFA : Survivants vs Décédés")
    axes[1].set_xlabel("Issue")
    axes[1].set_ylabel("Score qSOFA")

    plt.tight_layout()
    save(fig, '07_qsofa_analysis.png')


def generate_all(df: pd.DataFrame):
    print("[INFO] Génération des figures...")
    fig_demographics(df)
    fig_severity_breakdown(df)
    fig_comorbidities(df)
    fig_biomarkers(df)
    fig_correlation(df)
    fig_symptoms(df)
    fig_qsofa(df)
    print(f"\n[INFO] {len(os.listdir(FIGURES_DIR))} figures générées dans {FIGURES_DIR}")


if __name__ == '__main__':
    df = clean_pipeline()
    generate_all(df)
