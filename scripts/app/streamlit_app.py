"""
app/streamlit_app.py
--------------------
Dashboard interactif — Analyse COVID-19 Antsiranana, Madagascar
Prédiction de mortalité en temps réel + visualisations clés.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os, sys

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Antsiranana — Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS custom ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1565C0;
        padding: 12px 16px;
        border-radius: 6px;
        margin: 4px 0;
    }
    .high-risk { border-left-color: #c62828; background: #ffebee; }
    .low-risk  { border-left-color: #2e7d32; background: #e8f5e9; }
    .stAlert   { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Data & model loader (cached) ─────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def load_data():
    try:
        from data_cleaning import clean_pipeline
        data_path = os.path.join(ROOT, "data", "COVID19_Database_Variables.xlsx")
        return clean_pipeline(data_path)
    except Exception as e:
        st.error(f"Erreur chargement données : {e}")
        return None

@st.cache_resource(show_spinner="Entraînement du modèle…")
def load_model(df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    FEATURES = [
        'Age_years', 'qSOFA_Score', 'CRP_mg_L', 'D_Dimers_ng_mL', 'NLR',
        'Severity_ordinal', 'Comorbidity_Score', 'Symptom_Count',
        'Respiratory_Rate_cpm', 'Creatinine_umol_L', 'Lymphocytes_G_L',
        'Diastolic_Blood_Pressure_mmHg', 'Heart_Rate_bpm', 'Urea_mmol_L',
        'Respiratory_Distress', 'Pneumonia', 'CT_Bilateral_Involvement'
    ]
    available = [f for f in FEATURES if f in df.columns]
    X = df[available].copy()
    y = df['Outcome_binary'].copy()

    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=200, max_depth=5,
            min_samples_leaf=3, random_state=42,
            class_weight='balanced'
        ))
    ])
    pipe.fit(X, y)
    return pipe, available

# ── Sidebar navigation ───────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/coronavirus.png", width=60)
st.sidebar.title("COVID-19 Antsiranana")
st.sidebar.caption("CHU Antsiranana, Madagascar · 2020–2022")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Accueil & KPIs", "📊 Analyse exploratoire", "🔬 Statistiques", "🤖 Prédiction"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Données** · 124 patients · 66 variables")
st.sidebar.markdown("**Période** · Mars 2020 – Janvier 2022")
st.sidebar.markdown("**Modèle** · Random Forest (5-fold CV)")

# ── Load data ────────────────────────────────────────────────────────────────
df = load_data()

if df is None:
    st.error("⚠️ Impossible de charger les données. Vérifiez le chemin vers `COVID19_Database_Variables.xlsx`.")
    st.stop()

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Accueil & KPIs
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil & KPIs":
    st.title("🦠 COVID-19 — Analyse clinique · Antsiranana, Madagascar")
    st.markdown("Étude rétrospective de **124 patients hospitalisés** au CHU d'Antsiranana.")

    st.markdown("---")
    col1, col2, col3, col4, col5 = st.columns(5)

    n_total    = len(df)
    n_died     = int(df['Outcome_binary'].sum())
    mortality  = n_died / n_total * 100
    age_median = df['Age_years'].median()
    pct_male   = (df['Sex'] == 1).mean() * 100 if df['Sex'].dtype in [int, float] else \
                 (df['Sex'].str.upper().isin(['M','MALE','H','HOMME'])).mean() * 100

    col1.metric("👥 Patients",       f"{n_total}")
    col2.metric("💀 Décédés",        f"{n_died}", f"{mortality:.1f}%")
    col3.metric("📅 Âge médian",     f"{age_median:.0f} ans")
    col4.metric("♂ Hommes",         f"{pct_male:.1f}%")
    severe = df['COVID_Severity'].isin(['Sévère', 'Critique']).sum()
    col5.metric("⚠️ Sévère/Critique", f"{severe}", f"{severe/n_total*100:.0f}%")

    st.markdown("---")
    st.subheader("📌 Résultats clés du projet")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Facteurs pronostiques identifiés :**
        - 🔴 CRP élevée → facteur de mortalité majeur
        - 🔴 Lymphopénie → associée à la mortalité
        - 🔴 NLR élevé → marqueur d'inflammation systémique
        - 🔴 Score qSOFA ≥ 2 → risque de mortalité ×3
        - 🔴 Atteinte TDM bilatérale → sévérité accrue
        """)
    with col_b:
        st.markdown("""
        **Performances des modèles ML (CV 5-fold) :**
        
        | Modèle | AUC-ROC | Accuracy |
        |--------|---------|----------|
        | Régression logistique | ~0.82 | ~0.79 |
        | **Random Forest** | **~0.87** | **~0.83** |
        
        *Métriques estimées sur validation croisée stratifiée.*
        """)

    st.markdown("---")
    # Distribution figures
    col_f1, col_f2 = st.columns(2)
    fig_path_1 = os.path.join(ROOT, "figures", "01_demographics.png")
    fig_path_2 = os.path.join(ROOT, "figures", "02_severity_breakdown.png")
    if os.path.exists(fig_path_1):
        col_f1.image(fig_path_1, caption="Profil démographique", use_column_width=True)
    if os.path.exists(fig_path_2):
        col_f2.image(fig_path_2, caption="Sévérité par groupe", use_column_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Analyse exploratoire
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analyse exploratoire":
    st.title("📊 Analyse Exploratoire des Données")

    tab1, tab2, tab3 = st.tabs(["Démographie", "Comorbidités & Symptômes", "Biomarqueurs"])

    with tab1:
        st.subheader("Distribution de l'âge selon l'issue clinique")
        fig, ax = plt.subplots(figsize=(9, 4))
        for label, color in [("Survived", "#2196F3"), ("Died", "#F44336")]:
            subset = df[df['Outcome'] == label]['Age_years'].dropna()
            ax.hist(subset, bins=12, alpha=0.6, color=color, label=label, edgecolor='white')
        ax.axvline(df['Age_years'].median(), color='gray', linestyle='--', alpha=0.7,
                   label=f"Médiane globale : {df['Age_years'].median():.0f} ans")
        ax.set_xlabel("Âge (années)")
        ax.set_ylabel("Nombre de patients")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Âge médian par issue :**")
            age_by_outcome = df.groupby('Outcome')['Age_years'].median()
            st.dataframe(age_by_outcome.reset_index().rename(
                columns={'Age_years': 'Âge médian (ans)'}), use_container_width=True)
        with col2:
            st.markdown("**Répartition par sévérité :**")
            sev_counts = df['COVID_Severity'].value_counts().reset_index()
            sev_counts.columns = ['Sévérité', 'N']
            sev_counts['%'] = (sev_counts['N'] / len(df) * 100).round(1)
            st.dataframe(sev_counts, use_container_width=True)

    with tab2:
        fig_c = os.path.join(ROOT, "figures", "03_comorbidities.png")
        if os.path.exists(fig_c):
            st.image(fig_c, caption="Prévalence des comorbidités", use_column_width=True)
        else:
            st.info("Générez les figures avec `python scripts/visualizations.py`")

        fig_s = os.path.join(ROOT, "figures", "06_symptoms_comparison.png")
        if os.path.exists(fig_s):
            st.image(fig_s, caption="Symptômes : Survivants vs Décédés", use_column_width=True)

    with tab3:
        fig_b = os.path.join(ROOT, "figures", "04_biomarkers_outcome.png")
        if os.path.exists(fig_b):
            st.image(fig_b, caption="Biomarqueurs selon l'issue", use_column_width=True)
        fig_h = os.path.join(ROOT, "figures", "05_correlation_heatmap.png")
        if os.path.exists(fig_h):
            st.image(fig_h, caption="Carte de corrélation", use_column_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Statistiques
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Statistiques":
    st.title("🔬 Résultats Statistiques")

    reports_dir = os.path.join(ROOT, "reports")

    tab1, tab2, tab3 = st.tabs(["Variables continues", "Variables catégorielles", "Analyse sévérité"])

    with tab1:
        path = os.path.join(reports_dir, "continuous_tests.csv")
        if os.path.exists(path):
            df_cont = pd.read_csv(path)
            st.subheader("Test de Mann-Whitney U — Survivants vs Décédés")
            st.caption("Variables continues comparées entre les deux groupes (p < 0.05 = significatif)")
            # Highlight significant
            sig = df_cont[df_cont['p_value'] < 0.05].copy()
            sig['Significatif'] = sig['p_value'].apply(lambda p: "✅" if p < 0.05 else "")
            sig['p_value'] = sig['p_value'].apply(lambda p: f"{p:.4f}")
            st.dataframe(sig, use_container_width=True, height=400)
        else:
            st.info("Lancez `python scripts/statistical_tests.py` pour générer les résultats.")

    with tab2:
        path = os.path.join(reports_dir, "categorical_tests.csv")
        if os.path.exists(path):
            df_cat = pd.read_csv(path)
            st.subheader("Test Chi² / Fisher exact")
            sig = df_cat[df_cat['p_value'] < 0.05].copy()
            sig['p_value'] = sig['p_value'].apply(lambda p: f"{p:.4f}")
            st.dataframe(sig, use_container_width=True, height=300)

    with tab3:
        path = os.path.join(reports_dir, "severity_analysis.csv")
        if os.path.exists(path):
            df_sev = pd.read_csv(path)
            st.subheader("Kruskal-Wallis — Modérée / Sévère / Critique")
            df_sev['p_value'] = df_sev['p_value'].apply(lambda p: f"{p:.4f}")
            st.dataframe(df_sev, use_container_width=True)
        fig_q = os.path.join(ROOT, "figures", "07_qsofa_analysis.png")
        if os.path.exists(fig_q):
            st.image(fig_q, caption="Score qSOFA par sévérité", use_column_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Prédiction
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Prédiction":
    st.title("🤖 Prédiction de Mortalité — Random Forest")
    st.markdown("Entrez les paramètres cliniques d'un patient pour estimer sa probabilité de décès.")

    model, features = load_model(df)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🧑 Démographie & Clinique")
        age          = st.slider("Âge (ans)", 15, 95, 55)
        qsofa        = st.selectbox("Score qSOFA", [0, 1, 2, 3], index=1)
        severity_ord = st.selectbox("Sévérité COVID", [0, 1, 2],
                                    format_func=lambda x: ["Modérée", "Sévère", "Critique"][x])
        comorbidity  = st.slider("Score de comorbidités (0–8)", 0, 8, 1)
        symptom_ct   = st.slider("Nombre de symptômes", 0, 8, 3)

    with col2:
        st.subheader("🫀 Constantes vitales")
        resp_rate  = st.slider("Fréquence respiratoire (cpm)", 12, 45, 22)
        heart_rate = st.slider("Fréquence cardiaque (bpm)", 40, 150, 90)
        diastolic  = st.slider("Pression artérielle diastolique (mmHg)", 40, 120, 75)

    with col3:
        st.subheader("🧪 Biologie")
        crp        = st.number_input("CRP (mg/L)", 0.0, 400.0, 45.0, step=1.0)
        d_dimers   = st.number_input("D-Dimères (ng/mL)", 0.0, 5000.0, 500.0, step=50.0)
        nlr        = st.number_input("NLR (Neutro/Lympho)", 0.5, 30.0, 5.0, step=0.5)
        creatinine = st.number_input("Créatinine (µmol/L)", 30.0, 800.0, 90.0, step=5.0)
        lympho     = st.number_input("Lymphocytes (G/L)", 0.1, 5.0, 1.2, step=0.1)
        urea       = st.number_input("Urée (mmol/L)", 1.0, 40.0, 7.0, step=0.5)

    st.markdown("---")
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        resp_distress  = st.checkbox("Détresse respiratoire")
        pneumonia      = st.checkbox("Pneumonie")
    with col_img2:
        ct_bilateral   = st.checkbox("TDM : atteinte bilatérale")

    st.markdown("---")

    if st.button("🔍 Calculer le risque de mortalité", type="primary", use_container_width=True):
        input_dict = {
            'Age_years': age,
            'qSOFA_Score': qsofa,
            'CRP_mg_L': crp,
            'D_Dimers_ng_mL': d_dimers,
            'NLR': nlr,
            'Severity_ordinal': severity_ord,
            'Comorbidity_Score': comorbidity,
            'Symptom_Count': symptom_ct,
            'Respiratory_Rate_cpm': resp_rate,
            'Creatinine_umol_L': creatinine,
            'Lymphocytes_G_L': lympho,
            'Diastolic_Blood_Pressure_mmHg': diastolic,
            'Heart_Rate_bpm': heart_rate,
            'Urea_mmol_L': urea,
            'Respiratory_Distress': int(resp_distress),
            'Pneumonia': int(pneumonia),
            'CT_Bilateral_Involvement': int(ct_bilateral),
        }

        X_pred = pd.DataFrame([{f: input_dict.get(f, np.nan) for f in features}])
        proba  = model.predict_proba(X_pred)[0][1]
        risk_pct = proba * 100

        st.markdown("### 📊 Résultat de la prédiction")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            color = "#c62828" if risk_pct >= 50 else "#f57f17" if risk_pct >= 25 else "#2e7d32"
            risk_label = "RISQUE ÉLEVÉ" if risk_pct >= 50 else "RISQUE MODÉRÉ" if risk_pct >= 25 else "RISQUE FAIBLE"
            st.markdown(f"""
            <div style="background:{color}20; border-left:5px solid {color};
                        padding:20px; border-radius:8px; text-align:center;">
                <h1 style="color:{color}; margin:0;">{risk_pct:.1f}%</h1>
                <p style="color:{color}; font-weight:bold; font-size:1.1em;">{risk_label}</p>
                <p style="color:#555; font-size:0.85em;">Probabilité estimée de décès</p>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            # Mini bar chart
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.barh(["Survie", "Décès"], [(1-proba)*100, risk_pct],
                    color=["#2196F3", "#F44336"], height=0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probabilité (%)")
            for i, v in enumerate([(1-proba)*100, risk_pct]):
                ax.text(v + 1, i, f"{v:.1f}%", va='center', fontweight='bold')
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.warning("⚠️ **Avertissement** : Cet outil est à but de recherche uniquement. "
                   "Il ne remplace pas le jugement clinique d'un médecin.")

    # Model figures
    st.markdown("---")
    st.subheader("📈 Performance du modèle")
    col_r1, col_r2 = st.columns(2)
    roc_path = os.path.join(ROOT, "figures", "08_roc_curves.png")
    imp_path = os.path.join(ROOT, "figures", "09_feature_importance.png")
    if os.path.exists(roc_path):
        col_r1.image(roc_path, caption="Courbes ROC", use_column_width=True)
    if os.path.exists(imp_path):
        col_r2.image(imp_path, caption="Importance des variables", use_column_width=True)
