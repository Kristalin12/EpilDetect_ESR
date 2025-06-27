import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib 
import numpy as np
import pywt  
from tensorflow.keras.models import load_model

@st.cache_resource
def load_artifacts():
    encoder   = load_model("encoder.keras", compile=False)      
    clf       = joblib.load("voting_model.pkl")            
    scaler    = joblib.load("scaler.pkl")                      
    return encoder, clf, scaler

encoder, clf, scaler = load_artifacts()

def row_to_wavelet87(row_178, fs=173.61, wavelet_name="morl"):
    """
    row_178 : 1-D ndarray shape (178,)
    returns : 1-D ndarray shape (87,)  (coeffs at time-index 0 for each scale)
    """
    scales = np.arange(0.25, (fs/20)+0.25, 0.1)       
    coeffs, _ = pywt.cwt(row_178, scales, wavelet_name)
    return coeffs[:, 0] 

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
        st.write("Input data:")
        st.write(df)
        fig, ax = plt.subplots()
        ax.plot(df.iloc[0].values, color="purple", linewidth=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("EEG Signal (from uploaded CSV)")
        st.pyplot(fig)

        X_wavelet = np.vstack([row_to_wavelet87(r) for r in df.values])
        X_scaled = scaler.transform(X_wavelet)
        
        X_resh   = X_scaled.reshape((-1, 87, 1))
        latent   = encoder.predict(X_resh, verbose=0)
        X_flat   = latent.reshape((latent.shape[0], -1))
        
        y_pred   = clf.predict(X_flat)
        proba    = clf.predict_proba(X_flat)[:, 1]

        if y_pred[0] == 1:
            st.markdown("# ⚠️ **Seizure Detected** ⚠️", unsafe_allow_html=True)
            st.markdown("#### **Recommended Action:** Seek immediate medical attention.")
        else:
            st.markdown("# ✅ **No Seizure Detected**", unsafe_allow_html=True)
            st.markdown("#### **Recommended Action:** Continue monitoring as usual.")
