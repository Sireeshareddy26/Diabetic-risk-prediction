import sklearn
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetic Risk Predictor",
    page_icon="🩺",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .risk-high   { background:#ffe0e0; border-left:6px solid #e53935;
                   padding:20px; border-radius:8px; }
    .risk-medium { background:#fff8e1; border-left:6px solid #fb8c00;
                   padding:20px; border-radius:8px; }
    .risk-low    { background:#e8f5e9; border-left:6px solid #43a047;
                   padding:20px; border-radius:8px; }
    .metric-card { background:white; padding:16px; border-radius:10px;
                   box-shadow:0 2px 6px rgba(0,0,0,.08); text-align:center; }
    h1 { color: #1a237e; }
</style>
""", unsafe_allow_html=True)

# ── Load model & features ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("features.pkl", "rb") as f:
        features = pickle.load(f)
    return model, features

try:
    model, model_features = load_model()
except Exception as e:
    st.error(f"❌ Could not load model files: {e}")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🩺 Diabetic Risk Predictor")
st.markdown("Fill in the patient details on the left and click **Predict** to assess diabetic risk.")
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("👤 Patient Information")

with st.sidebar:
    st.subheader("Demographics")
    age    = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Female", "Male"])

    st.subheader("Clinical Measurements")
    waist_circ   = st.number_input("Waist Circumference (cm)", min_value=50.0,  max_value=200.0, value=88.0)
    fast_glucose = st.number_input("Fasting Glucose (mg/dL)",  min_value=50.0,  max_value=400.0, value=95.0)
    fast_trig    = st.number_input("Fasting Triglycerides (mg/dL)", min_value=30.0, max_value=1000.0, value=110.0)

    st.subheader("Lifestyle & History")
    pa_level = st.selectbox("Physical Activity Level", [
        "Vigorous exercise or strenuous at work",
        "Moderate exercise at work/home",
        "Mild exercise at work/home",
        "No exercise and sedentary",
    ])
    fam_hist = st.selectbox("Family History of Diabetes", [
        "Two non-diabetic parents",
        "Either parent diabetic",
        "Both parents diabetic",
    ])

    st.subheader("Risk Scores")
    age_score    = st.slider("Age Score",              0, 30, 0)
    abd_score    = st.slider("Abdominal Obesity Score",0, 20, 0)
    pa_score     = st.slider("Physical Activity Score",0, 10, 0)
    fam_score    = st.slider("Family History Score",   0, 10, 0)

    predict_btn  = st.button("🔍 Predict Risk", use_container_width=True, type="primary")

# ── Derived metrics ───────────────────────────────────────────────────────────
tyg_index          = np.log(fast_glucose * fast_trig / 2)
total_risk_score   = age_score + abd_score + pa_score + fam_score

# ── Main panel – calculated metrics ──────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"""<div class="metric-card">
        <h4 style="margin:0;color:#555">TYG Index</h4>
        <h2 style="margin:4px 0;color:#1a237e">{tyg_index:.4f}</h2>
        <small>log(Glucose × Triglycerides / 2)</small></div>""",
        unsafe_allow_html=True)
with col2:
    st.markdown(f"""<div class="metric-card">
        <h4 style="margin:0;color:#555">Total Risk Score</h4>
        <h2 style="margin:4px 0;color:#1a237e">{total_risk_score}</h2>
        <small>Sum of all individual scores</small></div>""",
        unsafe_allow_html=True)
with col3:
    bmi_proxy = waist_circ / (age ** 0.5)
    st.markdown(f"""<div class="metric-card">
        <h4 style="margin:0;color:#555">Waist-Age Index</h4>
        <h2 style="margin:4px 0;color:#1a237e">{bmi_proxy:.2f}</h2>
        <small>Waist ÷ √Age (proxy indicator)</small></div>""",
        unsafe_allow_html=True)

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    # Build processed input matching training features
    df_proc = pd.DataFrame(0, index=[0], columns=model_features)

    num_map = {
        "Age":                          age,
        "Waist Circumference (cm)":     waist_circ,
        "Fasting Glucose (mg/dL)":      fast_glucose,
        "Fasting Triglycerides (mg/dL)":fast_trig,
        "TYG Index":                    tyg_index,
        "Age Score":                    age_score,
        "Abdominal Obesity Score":      abd_score,
        "Physical Activity Score":      pa_score,
        "Family History Score":         fam_score,
        "Total Diabetic Risk Score":    total_risk_score,
    }
    for col, val in num_map.items():
        if col in df_proc.columns:
            df_proc[col] = val

    # One-hot flags
    ohe_flags = {
        f"Gender_{gender}": 1,
        f"Physical Activity Level_{pa_level}": 1,
        f"Family History of Diabetes_{fam_hist}": 1,
    }
    for col, val in ohe_flags.items():
        if col in df_proc.columns:
            df_proc[col] = val

    try:
        prediction   = model.predict(df_proc)[0]
        probabilities = model.predict_proba(df_proc)[0]
        classes       = model.classes_

        # Result card
        if prediction == "High Risk":
            css = "risk-high"
            icon = "🔴"
            advice = "Immediate medical evaluation recommended. Lifestyle changes and possible medication needed."
        elif prediction == "Medium Risk":
            css = "risk-medium"
            icon = "🟡"
            advice = "Lifestyle modifications strongly advised. Regular monitoring recommended."
        else:
            css = "risk-low"
            icon = "🟢"
            advice = "Continue healthy habits. Annual check-ups are still recommended."

        st.markdown(f"""<div class="{css}">
            <h2>{icon} {prediction}</h2>
            <p style="margin:0">{advice}</p>
        </div>""", unsafe_allow_html=True)

        # Probability breakdown
        st.subheader("📊 Prediction Confidence")
        prob_df = pd.DataFrame({
            "Risk Category": classes,
            "Probability (%)": (probabilities * 100).round(1)
        }).sort_values("Probability (%)", ascending=False).reset_index(drop=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        with col_b:
            st.bar_chart(prob_df.set_index("Risk Category")["Probability (%)"])

        # Input summary
        with st.expander("📋 View Input Summary"):
            summary = {
                "Age": age, "Gender": gender,
                "Waist Circumference (cm)": waist_circ,
                "Fasting Glucose (mg/dL)": fast_glucose,
                "Fasting Triglycerides (mg/dL)": fast_trig,
                "TYG Index": round(tyg_index, 4),
                "Physical Activity Level": pa_level,
                "Family History": fam_hist,
                "Total Risk Score": total_risk_score,
            }
            st.table(pd.DataFrame(summary.items(), columns=["Parameter", "Value"]))

    except Exception as e:
        st.error(f"Prediction failed: {e}")

else:
    st.info("👈 Fill in the patient details in the sidebar and click **Predict Risk**.")

# ── Feature importance chart ──────────────────────────────────────────────────
with st.expander("📈 Feature Importance (Model Insight)"):
    feat_imp = pd.Series(model.feature_importances_, index=model_features)
    top10    = feat_imp.nlargest(10).sort_values()
    st.bar_chart(top10)
    st.caption("Top 10 features driving the model's predictions.")

# ── Footer note ───────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
> ⚠️ **Disclaimer:** This tool is for educational and screening purposes only.
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.
> This model achieved perfect accuracy on the training dataset which may indicate
> data leakage or a synthetic dataset — validate carefully before clinical use.
""")
