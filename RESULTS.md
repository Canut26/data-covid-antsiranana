# 📈 Résultats — COVID-19 Antsiranana, Madagascar

> Étude rétrospective · 124 patients hospitalisés · CHU Antsiranana · Mars 2020 – Janvier 2022

---

## 🏥 Statistiques descriptives

| Indicateur | Valeur |
|---|---|
| **Patients inclus** | 124 |
| **Décès** | 25 (20.2%) |
| **Survivants** | 99 (79.8%) |
| **Âge médian** | 52 ans (IQR : 41–65) |
| **Hommes** | 74 (59.7%) |
| **Durée médiane d'hospitalisation** | 9 jours |
| **Sévérité modérée** | 20 (16.1%) |
| **Sévérité sévère** | 62 (50.0%) |
| **Sévérité critique** | 42 (33.9%) |

---

## 🔬 Facteurs pronostiques de mortalité

### Variables continues (Mann-Whitney U, p < 0.05)

| Variable | Médiane Survivants | Médiane Décédés | p-value |
|---|---|---|---|
| **CRP (mg/L)** | 38.5 | 112.4 | < 0.001 |
| **D-Dimères (ng/mL)** | 420 | 1850 | < 0.001 |
| **NLR** | 4.8 | 11.2 | < 0.001 |
| **Score qSOFA** | 1.0 | 2.0 | < 0.001 |
| **Fréquence respiratoire (cpm)** | 22 | 32 | < 0.001 |
| **Créatinine (µmol/L)** | 78 | 156 | 0.002 |
| **Lymphocytes (G/L)** | 1.4 | 0.6 | 0.001 |
| **Âge (ans)** | 49 | 61 | 0.015 |

### Variables catégorielles (Chi² / Fisher exact, p < 0.05)

| Variable | % Survivants | % Décédés | p-value |
|---|---|---|---|
| **Détresse respiratoire** | 38% | 88% | < 0.001 |
| **TDM atteinte bilatérale** | 52% | 84% | 0.003 |
| **Pneumonie** | 48% | 80% | 0.004 |
| **Lymphopénie** | 28% | 68% | 0.001 |
| **CRP élevée** | 55% | 92% | 0.001 |
| **Hypertension** | 32% | 56% | 0.022 |

---

## 🤖 Performance des modèles prédictifs

### Validation croisée stratifiée 5-fold

| Modèle | AUC-ROC | Accuracy | Sensibilité | Spécificité |
|---|---|---|---|---|
| Régression Logistique | 0.823 ± 0.048 | 0.790 ± 0.052 | 0.76 | 0.80 |
| **Random Forest** | **0.874 ± 0.039** | **0.831 ± 0.044** | **0.80** | **0.84** |

> **Meilleur modèle** : Random Forest (AUC = 0.874)

### Top 5 variables — Random Forest (importance Gini)

| Rang | Variable | Importance |
|---|---|---|
| 1 | CRP (mg/L) | 0.187 |
| 2 | NLR | 0.153 |
| 3 | Score qSOFA | 0.142 |
| 4 | D-Dimères (ng/mL) | 0.118 |
| 5 | Sévérité (ordinal) | 0.097 |

---

## 🩺 Interprétation clinique

Les résultats confirment que les **marqueurs biologiques d'inflammation** (CRP, NLR, D-Dimères) 
et le **score qSOFA** sont les meilleurs prédicteurs de mortalité dans cette population.

Ces findings sont cohérents avec la littérature internationale sur les LMIC (pays à revenus 
faibles/intermédiaires) et suggèrent qu'un **panel biologique minimal** (CRP + NFS + D-Dimères) 
associé au score qSOFA permettrait de stratifier le risque de mortalité même dans des contextes 
de ressources limitées.

---

## 📊 Figures générées

| Figure | Description |
|---|---|
| `01_demographics.png` | Distribution âge, sexe, issue clinique |
| `02_severity_breakdown.png` | Sévérité par groupe d'âge et sexe |
| `03_comorbidities.png` | Prévalence des comorbidités |
| `04_biomarkers_outcome.png` | Biomarqueurs selon l'issue |
| `05_correlation_heatmap.png` | Matrice de corrélation |
| `06_symptoms_comparison.png` | Symptômes : Survivants vs Décédés |
| `07_qsofa_analysis.png` | Score qSOFA par sévérité |
| `08_roc_curves.png` | Courbes ROC des deux modèles |
| `09_feature_importance.png` | Importance des variables (Random Forest) |

---

*Données anonymisées — Étude rétrospective observationnelle — CHU Antsiranana, Madagascar*
