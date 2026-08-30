"""Streamlit UI for the Track 1 healthcare diagnosis MVP."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path("artifacts/diagnosis_model.joblib")

st.set_page_config(page_title="Health Diagnosis Assistant", page_icon="🩺", layout="wide")
st.title("🩺 Health Diagnosis Assistant")
st.caption("Track 1 MVP — supports clinical decision-making; it is not a medical diagnosis.")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error("Model not found. Train it first: `python train.py --data path/to/track1_participant_dataset.csv`")
    st.stop()

model = load_model()
feature_names = model.feature_names_in_.tolist()
preprocessor = model.named_steps["preprocess"]
categorical = preprocessor.transformers_[0][2]
numeric = preprocessor.transformers_[1][2]

with st.sidebar:
    st.header("Patient information")
    values: dict[str, object] = {}
    for feature in feature_names:
        if feature in categorical:
            if feature == "sex":
                options = ["Female", "Male"]
            elif feature == "season":
                options = ["Dry", "Rainy"]
            elif feature == "month":
                options = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            elif feature == "state":
                options = ["Lagos", "Kano", "Oyo", "Rivers", "Bauchi", "Benue"]
            else:
                options = ["0", "1"]
            values[feature] = st.selectbox(feature.replace("_", " ").title(), options)
        else:
            default = 0.0
            if feature == "age":
                values[feature] = st.number_input(
                "Age", min_value=0, max_value=120, value=30, step=1
                )
                continue
            elif feature == "bmi": default = 22.0
            elif feature == "temperature_c": default = 37.0
            elif feature == "heart_rate": default = 80
            elif feature == "resp_rate": default = 18
            elif feature == "spo2": default = 98
            elif feature == "sbp": default = 120
            elif feature == "dbp": default = 80
            elif feature == "hemoglobin": default = 13.0
            elif feature == "wbc": default = 7.0
            elif feature == "platelets": default = 250000
            values[feature] = st.number_input(feature.replace("_", " ").title(), value=float(default))

if st.button("Assess patient", type="primary", use_container_width=True):
    patient = pd.DataFrame([values], columns=feature_names)
    probabilities = model.predict_proba(patient)[0]
    ranked = pd.DataFrame({"Diagnosis": model.classes_, "Probability": probabilities}).sort_values("Probability", ascending=False)
    top = ranked.iloc[0]
    st.subheader(f"Most likely pattern: {top['Diagnosis']}")
    st.metric("Model confidence", f"{top['Probability']:.1%}")
    st.bar_chart(ranked.set_index("Diagnosis"))
    st.dataframe(ranked.style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)
    st.warning("For demonstration only. A qualified clinician must evaluate symptoms, examination findings, and laboratory results.")
