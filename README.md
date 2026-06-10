# 🦠 COVID-19 Clinical Database — Antsiranana, Madagascar

[![CI](https://github.com/Canut26/data-covid-antsiranana/actions/workflows/ci.yml/badge.svg)](https://github.com/Canut26/data-covid-antsiranana/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.11-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker)](Dockerfile)
[![Streamlit App](https://img.shields.io/badge/demo-streamlit-FF4B4B?logo=streamlit)](app/streamlit_app.py)
[![Tests](https://img.shields.io/badge/tests-pytest-passing-brightgreen)](tests/)

> Analyse épidémiologique et clinique de **124 patients hospitalisés pour COVID-19** au CHU d'Antsiranana (Diego-Suarez), Madagascar · Mars 2020 – Janvier 2022

---

## 📋 Description du projet

Ce dépôt contient une base de données clinique de 124 patients, 66 variables cliniques et biologiques, 
ainsi que l'ensemble du pipeline d'analyse : nettoyage, statistiques comparatives, modélisation ML 
et un **dashboard interactif de prédiction de mortalité**.

L'objectif est d'identifier les **facteurs pronostiques de mortalité** dans un contexte LMIC 
(*Low- and Middle-Income Country*) avec des ressources diagnostiques limitées.

---

## 🚀 Demo rapide

```bash
git clone https://github.com/Canut26/data-covid-antsiranana.git
cd data-covid-antsiranana
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Ou avec Docker (recommandé) :**

```bash
docker compose up app
# → http://localhost:8501
```

---

## 🗂️ Structure du dépôt

```
covid-antsiranana/
│
├── app/
│   └── streamlit_app.py               # 🎛️  Dashboard interactif (4 pages)
│
├── data/
│   └── COVID19_Database_Variables.xlsx  # Base brute (124 patients, 66 variables)
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_statistical_analysis.ipynb
│   └── 04_predictive_modeling.ipynb
│
├── scripts/
│   ├── data_cleaning.py               # Pipeline de nettoyage
│   ├── statistical_tests.py           # Chi², Fisher, Mann-Whitney, Kruskal-Wallis
│   ├── visualizations.py              # Génération des figures
│   └── predictive_model.py            # Régression Logistique + Random Forest
│
├── tests/
│   ├── test_data_cleaning.py          # 🧪 Tests unitaires (cleaning)
│   ├── test_statistical_tests.py      # 🧪 Tests unitaires (stats)
│   └── test_predictive_model.py       # 🧪 Tests unitaires (ML)
│
├── figures/                           # 9 figures générées automatiquement
├── reports/                           # CSV statistiques exportés
│
├── .github/workflows/ci.yml           # ⚙️  CI/CD GitHub Actions
├── Dockerfile                         # 🐳 Conteneurisation
├── docker-compose.yml                 # 🐳 Orchestration (app + jupyter)
├── RESULTS.md                         # 📈 Résultats détaillés & performances ML
├── CITATION.cff                       # 📚 Référence académique
└── requirements.txt
```

---

## 📊 Base de données

| Caractéristique | Valeur |
|---|---|
| Patients | 124 |
| Variables | 66 |
| Période | Mars 2020 – Janvier 2022 |
| Lieu | CHU Antsiranana, Madagascar |
| Taux de mortalité | **20.2%** |

**Variables clés :** démographie, comorbidités, symptômes, biologie (CRP, D-Dimères, NFS), 
imagerie (TDM thoracique), sévérité (`Modérée / Sévère / Critique`), issue (`Survived / Died`), qSOFA.

---

## 🔬 Analyses réalisées

**1. Nettoyage & Feature Engineering**
- Encodage ordinal de la sévérité
- Score de comorbidités composite
- NLR (Neutrophile/Lymphocyte Ratio)
- Groupes d'âge

**2. Analyse exploratoire (EDA)**  
Distribution démographique · Prévalence des comorbidités · Profil symptomatique · Corrélations

**3. Statistiques comparatives (Survivants vs Décédés)**  
- Variables catégorielles : Chi² / Fisher exact  
- Variables continues : Mann-Whitney U (non-paramétrique)
- Sévérité : Kruskal-Wallis 3 groupes

**4. Modélisation prédictive**  
- Régression Logistique + Random Forest  
- Validation croisée stratifiée 5-fold  
- AUC-ROC, sensibilité, spécificité, importance des features

**5. Dashboard interactif (Streamlit)**  
- KPIs en temps réel · EDA interactive · Résultats statistiques · Prédiction de mortalité patient

---

## 📈 Résultats clés

| Modèle | AUC-ROC | Accuracy |
|---|---|---|
| Régression Logistique | 0.823 ± 0.048 | 0.790 |
| **Random Forest** | **0.874 ± 0.039** | **0.831** |

**Top facteurs pronostiques :** CRP élevée · NLR · Score qSOFA · D-Dimères · Lymphopénie

→ Voir [`RESULTS.md`](RESULTS.md) pour les résultats complets.

---

## 🧪 Tests

```bash
# Lancer tous les tests
pytest

# Avec couverture de code
pytest --cov=scripts --cov-report=term-missing
```

Les tests couvrent : pipeline de nettoyage, tests statistiques (Chi², Mann-Whitney), 
et les modèles ML (fit, predict, gestion des NaN).

---

## 🐳 Docker

```bash
# Dashboard seul
docker compose up app

# Dashboard + Jupyter (mode développement)
docker compose --profile dev up

# Accès :
# Dashboard  → http://localhost:8501
# Jupyter    → http://localhost:8888
```

---

## 🛠️ Technologies

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## 👤 Auteur

**Canut26** — Étude clinique rétrospective · Antsiranana, Madagascar  
📧 Disponible sur demande · 🔗 [GitHub](https://github.com/Canut26)

Pour citer ce travail : voir [`CITATION.cff`](CITATION.cff)

---

## 📄 Licence

MIT — voir [`LICENSE`](LICENSE)

---

> ⚠️ Données anonymisées. Aucune information permettant d'identifier un patient n'est présente dans ce dépôt.
