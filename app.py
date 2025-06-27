import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib 
import numpy as np

model = joblib.load('voting_model.pkl') 

st.set_page_config(layout="centered", page_title="Epileptic Seizure Recognition")
st.title("🧠 Epileptic Seizure Recognition")
st.caption("Upload CSV file with 178 EEG features")

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

        prediction = model.predict(df)[0]
        label = "Seizure" if prediction == 1 else "Non-Seizure"

        st.subheader("🧪 Prediction Result:")
        st.markdown(f"**Prediction:** `{label}`", unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0][1] 
            st.progress(proba)
            st.markdown(f"Confidence: `{proba:.2f}`")

