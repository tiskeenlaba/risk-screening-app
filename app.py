"""Streamlit demo:  streamlit run app.py"""
import joblib
import pandas as pd
import streamlit as st

from utils import clean_text, explain, tier_for

st.set_page_config(page_title="Explainable Text-Risk Screening", page_icon="🩺", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load("models/risk_model.joblib")


art = load_artifact()
base, calibrated = art["base"], art["calibrated"]
low_thr, high_thr = art["low_thr"], art["high_thr"]

st.title("Explainable Text-Risk Screening")
st.warning(
    "This is a **screening aid, not a diagnosis**. It only looks for word patterns "
    "similar to posts labelled as showing concern. It cannot know your situation, and "
    "it can be wrong. Never use it to make decisions about a person's health."
)

text = st.text_area("Paste a post or message to analyze:", height=160,
                    placeholder="e.g. I have been feeling really tired and empty lately...")

if st.button("Analyze", type="primary"):
    cleaned = clean_text(text)
    if len(cleaned.split()) < 3:
        st.info("Please enter a little more text (at least a few meaningful words).")
    else:
        prob = float(calibrated.predict_proba([cleaned])[0, 1])
        tier = tier_for(prob, low_thr, high_thr)
        colour = {"Low": "green", "Moderate": "orange", "High": "red"}[tier]

        c1, c2 = st.columns(2)
        c1.metric("Concern tier", tier)
        c2.metric("Estimated probability", f"{prob:.0%}")
        st.markdown(f"Result: :{colour}[**{tier} level of concern patterns detected**]")

        if tier == "High":
            st.error(
                "If you or someone you know is struggling, please talk to someone you trust "
                "or a mental-health professional. In India you can call Tele-MANAS (14416). "
                "If you are in immediate danger, contact local emergency services."
            )

        up, down = explain(base, cleaned, top_k=8)
        st.subheader("Why did the model say this?")
        left, right = st.columns(2)
        with left:
            st.markdown("**Words pushing the score UP**")
            if up:
                st.bar_chart(pd.DataFrame(up, columns=["word", "contribution"]).set_index("word"),
                             color="#d62728")
            else:
                st.caption("None found.")
        with right:
            st.markdown("**Words pushing the score DOWN**")
            if down:
                d = pd.DataFrame(down, columns=["word", "contribution"])
                d["contribution"] = d["contribution"].abs()
                st.bar_chart(d.set_index("word"), color="#2ca02c")
            else:
                st.caption("None found.")

with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. Clean the text (lowercase, remove stopwords, lemmatize)\n"
        "2. Convert to numbers with **TF-IDF**\n"
        "3. **Logistic Regression** gives a score\n"
        "4. **Calibration** turns it into a trustworthy probability\n"
        f"5. Tiers: Low < {low_thr:.2f} ≤ Moderate < {high_thr:.2f} ≤ High\n"
        "6. Each word's contribution = its weight × its TF-IDF value"
    )
    m = art["metrics"]
    st.caption(f"Test ROC-AUC: {m['roc_auc']:.3f}")
