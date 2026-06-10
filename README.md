# 🦠 COVID-19 Clinical Database — Antsiranana, Madagascar

> **Analyse épidémiologique et clinique des patients hospitalisés pour COVID-19 à Antsiranana (Diego-Suarez), Madagascar**

---

## 📋 Description du projet

Ce dépôt contient une base de données clinique de **124 patients** hospitalisés pour COVID-19 à Antsiranana (Madagascar), ainsi que tous les scripts d'analyse statistique, de visualisation et de modélisation prédictive associés.

L'objectif principal est d'identifier les **facteurs pronostiques de mortalité et de sévérité** chez les patients COVID-19 dans un contexte de ressources limitées (LMIC — Low- and Middle-Income Country).

---

## 🗂️ Structure du dépôt

```
covid-antsiranana/
│
├── data/
│   └── COVID19_Database_Variables.xlsx    # Base de données brute (124 patients, 66 variables)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb             # Nettoyage et préparation des données
│   ├── 02_exploratory_analysis.ipynb      # Analyse exploratoire (EDA)
│   ├── 03_statistical_analysis.ipynb      # Tests statistiques (Chi², Mann-Whitney, etc.)
│   └── 04_predictive_modeling.ipynb       # Modèles prédictifs (Régression logistique, Random Forest)
│
├── scripts/
│   ├── data_cleaning.py                   # Pipeline de nettoyage des données
│   ├── statistical_tests.py               # Fonctions de tests statistiques
│   ├── visualizations.py                  # Génération des graphiques
│   └── predictive_model.py                # Modèles de prédiction de mortalité
│
├── figures/
│   └── (graphiques générés automatiquement)
│
├── reports/
│   └── summary_statistics.csv             # Statistiques descriptives exportées
│
├── requirements.txt                       # Dépendances Python
└── README.md
```

---

## 📊 Description de la base de données

| Caractéristique | Valeur |
|---|---|
| Nombre de patients | 124 |
| Nombre de variables | 66 |
| Période d'étude | Mars 2020 – Janvier 2022 |
| Lieu | CHU Antsiranana, Madagascar |
| Taux de mortalité | ~20% |

### Variables principales

**Données démographiques & cliniques**
- `Age_years`, `Sex`, `Length_of_Stay_days`
- `Admission_Date`, `Discharge_Date`, `Outcome`

**Comorbidités**
- Hypertension, Diabète, Obésité, Maladie cardiaque, BPCO, IRC, etc.

**Symptômes à l'admission**
- Fièvre, Toux, Dyspnée, Douleur thoracique, Fatigue, etc.

**Imagerie & Paraclinique**
- Scanner thoracique (lésions, verre dépoli, atteinte bilatérale)
- ECG (anomalie onde T, tachycardie)
- Biologie : NFS, CRP, D-Dimères, Créatinine, Glycémie...

**Sévérité & Pronostic**
- `COVID_Severity` : Modérée / Sévère / Critique
- `Outcome` : Survived / Died
- `qSOFA_Score`

---

## 🔬 Analyses réalisées

### 1. Analyse exploratoire (EDA)
- Distribution des variables démographiques
- Prévalence des comorbidités
- Profil symptomatique à l'admission
- Distribution de la sévérité et des issues cliniques

### 2. Analyse statistique comparative
- **Variables catégorielles** : test du Chi² / Fisher exact
- **Variables continues** : test de Mann-Whitney U (non-paramétrique)
- Comparaison Survivants vs Décédés
- Comparaison par niveau de sévérité (Modérée / Sévère / Critique)

### 3. Modélisation prédictive
- **Régression logistique** (prédiction de mortalité)
- **Random Forest Classifier** (importance des variables)
- Métriques : AUC-ROC, sensibilité, spécificité, accuracy
- Validation croisée 5-fold

---

## 🚀 Installation et utilisation

### Prérequis
```bash
Python >= 3.9
```

### Installation
```bash
git clone https://github.com/Canut26/data-covid-antsiranana.git
cd data-covid-antsiranana
pip install -r requirements.txt
```

### Exécution des scripts
```bash
# Nettoyage des données
python scripts/data_cleaning.py

# Analyse statistique
python scripts/statistical_tests.py

# Génération des visualisations
python scripts/visualizations.py

# Modèle prédictif
python scripts/predictive_model.py
```

### Lancer les notebooks Jupyter
```bash
jupyter notebook notebooks/
```

---

## 📈 Résultats clés

| Indicateur | Valeur |
|---|---|
| Mortalité globale | 20.2% |
| Âge médian | ~52 ans |
| Prédominance masculine | 59.7% (74/124) |
| Sévérité "Sévère" ou "Critique" | 83.9% |
| CRP élevée | Facteur pronostique majeur |
| Lymphopénie | Associée à la mortalité |

---

## 🛠️ Technologies utilisées

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-2.x-green?logo=pandas)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-red)
![Seaborn](https://img.shields.io/badge/Seaborn-Stats-purple)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-yellow?logo=jupyter)

---

## 👤 Auteur

**Canut26**  
Étude clinique rétrospective — Antsiranana, Madagascar  
📧 *Disponible sur demande*

---

## 📄 Licence

Ce projet est sous licence **MIT**. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

## 🙏 Remerciements

- Équipe médicale du CHU d'Antsiranana
- Patients ayant contribué (données anonymisées)
- Communauté open-source Python / Data Science

---

*⚠️ Les données sont anonymisées. Aucune information permettant d'identifier un patient n'est présente dans ce dépôt.*
