# NCD Referral Risk Predictor

A Streamlit web app that estimates whether a patient screened at a community
health post should be referred to a higher-level facility. Built for
Module 8B — Integrating AI into Model Deployment.

**Author:** Daniel Abba
**Live app:** _add your Streamlit Cloud URL here_
**Demo video:** _add your video link here_

## What it does

A health worker enters the patient's screening details (age, sex, BMI,
blood pressure, fasting glucose, smoking status, family history, and the
condition identified at screening). The app returns a referral
recommendation with an estimated probability, so borderline cases are
easy to spot.

## Tech stack

| Layer | Tool |
|---|---|
| Model | scikit-learn logistic regression (balanced class weights), saved with joblib |
| App | Streamlit |
| Container | Docker (`python:3.11-slim`, non-root user) |
| Hosting | Streamlit Community Cloud |

The model was trained on the Module 8A synthetic NCD screening dataset
(1,500 records, Jan–Dec 2024, three Nigerian states). Holdout accuracy
**86.7%**, ROC-AUC **0.925**. The raw `sex` column mixed `F`/`M` with
`Female`/`Male` — that is cleaned inside `train.py` before training.

## Files

```
app.py               Streamlit application (UI + prediction)
model.pkl            Trained pipeline (preprocessing + classifier)
train.py             Reproduces model.pkl from data/ncd_screening.csv
requirements.txt     Pinned dependencies
Dockerfile           Container recipe
docker-compose.yml   One-command local testing
```

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
# opens http://localhost:8501
```

## Run it with Docker

```bash
docker build -t ncd-app:v1 .
docker run -p 8501:8501 ncd-app:v1
# or: docker compose up --build
```

## Retrain the model

```bash
python train.py   # reads data/ncd_screening.csv, writes model.pkl
```

## How to use the app

1. Fill in the patient details on the left and the screening result on the right.
2. The form blocks inconsistent entries (e.g. diastolic ≥ systolic, or a
   diagnosis status on a "No NCD detected" record) and warns when values
   fall outside the training ranges.
3. Click **Assess referral risk** — the app shows the recommendation and
   the estimated referral probability.

## Note

The dataset is 100% synthetic teaching data. This app is a deployment
demo and must not be used for clinical decisions.
