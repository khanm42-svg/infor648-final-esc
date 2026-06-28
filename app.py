import streamlit as st
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Neighborhood Growth Predictor — East South Central",
    page_icon="📍",
    layout="wide",
)

CLUSTER_NAMES = {
    "0": "Aging Established Suburbs",
    "1": "Low-Density Rural Homeowners",
    "2": "Dense Urban Renter Core",
    "3": "Diverse Young Suburbs",
}
CLUSTER_COLORS = {"0": "#2E868A", "1": "#5B7FA6", "2": "#C0582A", "3": "#4A7C59"}

@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent
    cluster_scaler   = joblib.load(base / "cluster_scaler.joblib")
    kmeans           = joblib.load(base / "kmeans.joblib")
    pipeline         = joblib.load(base / "rf_pipeline.joblib")
    numeric_cols     = joblib.load(base / "numeric_cols.joblib")
    categorical_cols = joblib.load(base / "categorical_cols.joblib")
    train_means      = joblib.load(base / "train_means.joblib")
    feature_cols     = joblib.load(base / "feature_cols.joblib")
    return cluster_scaler, kmeans, pipeline, numeric_cols, categorical_cols, train_means, feature_cols

cluster_scaler, kmeans, pipeline, numeric_cols, categorical_cols, train_means, feature_cols = load_artifacts()

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#1B2A4A;padding:1.2rem 1.5rem;border-radius:8px;margin-bottom:1rem'>
<h2 style='color:white;margin:0'>📍 Neighborhood Growth Predictor</h2>
<p style='color:#B0C4DE;margin:0.3rem 0 0'>
East South Central Division &nbsp;·&nbsp; Alabama · Kentucky · Mississippi · Tennessee<br>
<small>Predicts whether a census tract will gain population over the next decade</small>
</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Enter a tract's characteristics below. Sliders you don't adjust will default to "
    "the average value from the training data — you only need to change what you know.",
    icon="ℹ️",
)

# ── Input form ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Location")
    state = st.selectbox("State", ["Alabama", "Kentucky", "Mississippi", "Tennessee"])
    settlement = st.selectbox("Settlement Type", ["rural", "suburban", "urban"])

    st.subheader("Population & Density")
    density = st.number_input(
        "Population density (people per km²)",
        min_value=1.0, max_value=20000.0,
        value=float(round(train_means.get("density_perkm2", 500))),
        step=50.0,
        help="How many people live per square kilometer in this tract"
    )

    st.subheader("Housing")
    vacancy_pct = st.slider(
        "Vacant housing units (%)", 0, 60,
        int(round(train_means.get("vacancy_rate", 0.12) * 100)),
        help="Share of total housing units that are vacant"
    )
    owner_pct = st.slider(
        "Owner-occupied units (%)", 0, 100,
        int(round(train_means.get("owner_occ_share", 0.65) * 100)),
        help="Share of occupied units that are owner-occupied"
    )

with col_right:
    st.subheader("Age Composition")
    pct_under18 = st.slider(
        "Population under 18 (%)", 0, 60,
        int(round(train_means.get("pct_under18", 0.23) * 100)),
    )
    pct_65plus = st.slider(
        "Population age 65+ (%)", 0, 60,
        int(round(train_means.get("pct_65plus", 0.14) * 100)),
    )
    age_18to19 = st.slider(
        "Population age 18–19 (%)", 0, 20,
        int(round(train_means.get("age_18to19_share", 0.027) * 100)),
        help="Share of tract population aged 18–19 (college-age signal)"
    )

    st.subheader("Race & Ethnicity Composition")
    race_nhwhite = st.slider(
        "Non-Hispanic White (%)", 0, 100,
        int(round(train_means.get("race_nhwhite_share", 0.72) * 100)),
    )
    race_nhblack = st.slider(
        "Non-Hispanic Black (%)", 0, 100,
        int(round(train_means.get("race_nhblack_share", 0.23) * 100)),
    )
    race_hisp = st.slider(
        "Hispanic (any race) (%)", 0, 100,
        int(round(train_means.get("race_hispwhite_share", 0.025) * 100)),
    )
    race_nhapi = st.slider(
        "Asian / Pacific Islander (%)", 0, 30,
        int(round(train_means.get("race_nhapi_share", 0.012) * 100)),
    )

# ── Build feature vector ────────────────────────────────────────────────────
def build_input_row():
    # Start from training means for every numeric feature
    row = dict(train_means)

    # Override with user inputs
    row["density_perkm2"]   = float(density)
    row["log_density"]      = float(np.log1p(density))
    row["log_pop"]          = float(np.log1p(density * train_means.get("land_area_sqkm", 50)))
    row["vacancy_rate"]     = vacancy_pct / 100
    row["owner_occ_share"]  = owner_pct / 100
    row["renter_occ_share"] = 1 - (owner_pct / 100)
    row["pct_under18"]      = pct_under18 / 100
    row["pct_65plus"]       = pct_65plus / 100
    row["age_18to19_share"] = age_18to19 / 100

    # Race shares — user-provided overrides
    row["race_nhwhite_share"]  = race_nhwhite / 100
    row["race_nhblack_share"]  = race_nhblack / 100
    row["race_nhapi_share"]    = race_nhapi / 100
    # hispwhite is the largest hispanic sub-group — approximate with full hispanic slider
    row["race_hispwhite_share"] = race_hisp / 100

    # Categorical
    row["STATE"]           = state
    row["settlement_type"] = settlement

    return row

# ── Predict ─────────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  Predict Growth", type="primary", use_container_width=True)

if predict_btn:
    row = build_input_row()

    # Build numeric-only row for clustering
    num_vals = np.array([[row[c] for c in numeric_cols]])
    num_scaled = cluster_scaler.transform(num_vals)
    cluster_id = str(kmeans.predict(num_scaled)[0])

    # Build full feature DataFrame matching what the pipeline expects
    # feature_cols = columns of X_train_with_cluster
    full_row = {col: row.get(col, 0.0) for col in feature_cols if col != "cluster"}
    full_row["cluster"] = cluster_id

    input_df = pd.DataFrame([full_row])[feature_cols]

    prob = pipeline.predict_proba(input_df)[0][1]
    pred = int(prob >= 0.5)

    # ── Display result ─────────────────────────────────────────────────────
    st.markdown("---")
    res_col, detail_col = st.columns([1, 1], gap="large")

    with res_col:
        if pred == 1:
            st.success(f"### ✅  Predicted to GROW\nConfidence: **{prob*100:.1f}%**")
        else:
            st.error(f"### ❌  Predicted NOT to grow\nConfidence: **{(1-prob)*100:.1f}%**")

        # Probability bar
        st.markdown("**Growth probability:**")
        st.progress(float(prob), text=f"{prob*100:.1f}%")

        st.markdown(f"""
**Assigned cluster:** `{cluster_id}` — {CLUSTER_NAMES[cluster_id]}

| Feature | Your Input |
|---|---|
| State | {state} |
| Settlement | {settlement} |
| Density | {density:.0f}/km² |
| Vacancy rate | {vacancy_pct}% |
| Owner-occupied | {owner_pct}% |
| Under 18 | {pct_under18}% |
| Age 65+ | {pct_65plus}% |
| Non-Hisp. White | {race_nhwhite}% |
| Non-Hisp. Black | {race_nhblack}% |
""")

    with detail_col:
        st.subheader("What the cluster means")
        color = CLUSTER_COLORS[cluster_id]
        desc = {
            "0": "Stable, older suburban neighborhoods. Medium density (~597/km²), mostly owner-occupied, high share of residents 65+. 58% of these tracts grew in the 2010s.",
            "1": "Sparse rural tracts. Very low density (~106/km²), 80% owner-occupied, predominantly white. Lowest growth rate at 54%.",
            "2": "Dense city-center tracts. High density (~1,825/km²), 77% renters, few children. Highest growth rate at 65%.",
            "3": "Diverse, younger suburban tracts. Medium-high density (~990/km²), mixed tenure, highest share of children. 58% grew in the 2010s.",
        }
        st.markdown(
            f"<div style='background:{color}22;border-left:4px solid {color};"
            f"padding:1rem;border-radius:4px'>"
            f"<b style='color:{color}'>Cluster {cluster_id}: {CLUSTER_NAMES[cluster_id]}</b><br><br>"
            f"{desc[cluster_id]}</div>",
            unsafe_allow_html=True,
        )

        st.subheader("What matters most for this prediction")
        st.markdown("""
The top predictors (from permutation importance on the test set) are:

1. **Race/ethnicity shares** — tracts with higher Asian/PI and Hispanic shares in 2010 sit in faster-growing suburban corridors
2. **Age 18–19 share** — university-adjacent tracts tend to see stronger surrounding growth
3. **Population density** — medium-density suburban tracts grow more reliably than very sparse or very dense ones

These reflect Sun Belt migration patterns during the 2010s.
""")

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Model: Random Forest (AUC = 0.84) trained on 2010 U.S. Census features for East South Central tracts. "
    "Predicts population growth over a 10-year period. INFO 648 · Summer 2026."
)
