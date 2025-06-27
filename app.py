import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib 
import numpy as np
from tensorflow.keras.models import load_model

@st.cache_resource
def load_artifacts():
    encoder   = load_model("encoder.keras", compile=False)
    classifier = joblib.load("voting_model.pkl")
    return encoder, classifier
    
encoder, classifier = load_artifacts()

st.set_page_config(layout="centered", page_title="Epileptic Seizure Recognition")
st.title("Epileptic Seizure Recognition")
st.caption("Upload CSV file dengan 178 fitur EEG")

uploaded_file = st.file_uploader("Drag and drop file here", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if df.shape[1] != 178:
        st.error("CSV must have exactly 178 columns (features)")
    else:
        st.subheader("📈 EEG Signal")
        fig, ax = plt.subplots()
        ax.plot(df.iloc[0].values, color="purple", linewidth=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("EEG Signal (from uploaded CSV)")
        st.pyplot(fig)

        X_raw = df.values.astype("float32")
        X_reshaped = X_raw.reshape((-1, 178, 1))

        X_encoded = encoder.predict(X_reshaped)
        X_flattened = X_encoded.reshape((X_encoded.shape[0], -1))
        
        y_pred = classifier.predict(X_flattened)
        proba = (classifier.predict_proba(X_flattened)[:, 1]
             if hasattr(classifier, "predict_proba") else None)

        st.subheader("🧪 Prediction Results")
        results = pd.DataFrame({
            "Segment #": np.arange(len(y_pred)) + 1,
            "Prediction": np.where(y_pred == 1, "Seizure", "Non-Seizure"),
            "Confidence": np.round(proba, 3) if proba is not None else "-"
        })
        st.table(results)

