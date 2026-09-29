"""
NCD Referral Risk Predictor — Streamlit app.

A triage tool for community screening teams: enter a patient's
screening measurements and the app estimates the probability that
the patient should be referred to a higher-level facility.

Model: logistic-regression pipeline trained on the Module 8A
synthetic NCD screening dataset (1,500 records, 2024).

DISCLAIMER: trained on synthetic teaching data — not for clinical use.
"""

import os

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "model.pkl"

CONDITIONS = [
    "Hypertension",
    "Diabetes",
    "Asthma",
    "Sickle Cell Disease",
    "Chronic Kidney Disease",
    "No NCD detected",
]
DIAGNOSIS_STATUS = ["Newly diagnosed", "Previously known", "Not applicable"]
CONTROLLED = ["Yes", "No", "Not applicable"]
YES_NO = ["No", "Yes"]


@st.cache_resource
def load_model():
    """Load the model bundle once and cache it for the session."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def validate_inputs(data):
    """Return (errors, warnings) for the entered screening record."""
    errors, warnings = [], []

    if data["diastolic_bp"] >= data["systolic_bp"]:
        errors.append(
            "Diastolic BP must be lower than systolic BP — "
            "please check the blood-pressure values."
        )

    no_ncd = data["condition"] == "No NCD detected"
    if no_ncd and data["diagnosis_status"] != "Not applicable":
        errors.append(
            "A patient with 'No NCD detected' cannot have a "
            "diagnosis status — set it to 'Not applicable'."
        )
    if not no_ncd and data["diagnosis_status"] == "Not applicable":
        errors.append(
            "A diagnosed condition needs a diagnosis status "
            "('Newly diagnosed' or 'Previously known')."
        )
    if no_ncd and data["controlled"] != "Not applicable":
        errors.append(
            "'Controlled' only applies to a detected condition — "
            "set it to 'Not applicable' for a negative screen."
        )

    # Soft checks against the ranges seen in the training data.
    if not 15.5 <= data["bmi"] <= 46.0:
        warnings.append("BMI is outside the range seen in training data (15.5–46).")
    if not 3.6 <= data["blood_glucose_mmol_l"] <= 20.0:
        warnings.append("Blood glucose is outside the training range (3.6–20.0).")
    if not 18 <= data["patient_age"] <= 88:
        warnings.append("Age is outside the training range (18–88).")

    return errors, warnings


def main():
    st.set_page_config(page_title="NCD Referral Risk Predictor", page_icon="🏥")
    st.title("NCD Referral Risk Predictor")
    st.caption(
        "Estimates whether a screened patient should be referred to a "
        "higher-level facility. Demo app built on synthetic training data."
    )

    bundle = load_model()
    if bundle is None:
        st.error(
            f"Model file '{MODEL_PATH}' was not found. "
            "Train it first with `python train.py`, then restart the app."
        )
        st.stop()
    model = bundle["model"]

    with st.sidebar:
        st.header("About the model")
        metrics = bundle.get("metrics", {})
        st.write(f"Holdout accuracy: **{metrics.get('accuracy', 0):.1%}**")
        st.write(f"ROC-AUC: **{metrics.get('roc_auc', 0):.3f}**")
        st.write("Model: logistic regression (balanced class weights)")
        st.divider()
        st.caption(
            "Synthetic teaching data — not for clinical or research use."
        )

    st.subheader("Patient details")
    col1, col2 = st.columns(2)
    with col1:
        patient_age = st.number_input("Age (years)", min_value=0, max_value=120, value=40)
        sex = st.selectbox("Sex", ["Female", "Male"])
        bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=70.0,
                              value=25.0, step=0.1)
        smoker = st.radio("Current smoker", YES_NO, horizontal=True)
        family_history = st.radio("Family history of NCD", YES_NO, horizontal=True)
    with col2:
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=260, value=128)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=30, max_value=180, value=82)
        blood_glucose = st.number_input("Fasting blood glucose (mmol/L)",
                                        min_value=1.0, max_value=35.0, value=5.5, step=0.1)

    st.subheader("Screening result")
    col3, col4, col5 = st.columns(3)
    with col3:
        condition = st.selectbox("Condition identified", CONDITIONS)
    with col4:
        diagnosis_status = st.selectbox("Diagnosis status", DIAGNOSIS_STATUS)
    with col5:
        controlled = st.selectbox("Condition controlled", CONTROLLED)

    record = {
        "patient_age": patient_age,
        "sex": sex,
        "bmi": bmi,
        "smoker": smoker,
        "family_history_ncd": family_history,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "blood_glucose_mmol_l": blood_glucose,
        "condition": condition,
        "diagnosis_status": diagnosis_status,
        "controlled": controlled,
    }

    errors, warnings = validate_inputs(record)
    for w in warnings:
        st.warning(w)

    if st.button("Assess referral risk", type="primary"):
        if errors:
            for e in errors:
                st.error(e)
            st.stop()

        try:
            X = pd.DataFrame([record])[bundle["features"]]
            proba = float(model.predict_proba(X)[0, 1])
        except Exception as exc:  # never crash on a bad record
            st.error(f"Prediction failed: {exc}")
            st.stop()

        referral = proba >= 0.5
        if referral:
            st.write("## 🔺 Referral recommended")
        else:
            st.write("## ✅ Referral not indicated")

        st.metric("Estimated referral probability", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))
        st.write(
            "The model weighs the screening result, blood pressure, glucose, "
            "and history. Patients with an uncontrolled or newly diagnosed "
            "NCD are the most likely to need referral."
        )
        st.caption(
            "This is a decision-support demo, not medical advice. "
            "Always apply clinical judgement."
        )


if __name__ == "__main__":
    main()
