# Explainable Text-Risk Screening System

A screening aid that reads a piece of text and returns a **calibrated probability**, a **Low / Moderate / High**
concern tier, and the **words that drove the result**.

> **Not a diagnosis.** It only detects word patterns similar to labelled examples and can be wrong.
> Please don't enter real personal information in the demo.

**Live demo:** _paste your Streamlit link here_

## How it works
Clean text (NLTK) -> TF-IDF (1-2 grams) -> Logistic Regression -> probability calibration -> tiers -> word-level explanation.
Trained on ~51k public statements (Kaggle "Sentiment Analysis for Mental Health"), concern vs normal.
Test F1 0.95, ROC-AUC 0.98. Post length alone gives AUC 0.93, so performance was also checked inside length groups.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` Streamlit interface
- `utils.py` text cleaning, tier rule, explanation
- `models/risk_model.joblib` trained model
