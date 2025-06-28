import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib 
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
from PIL import Image

def get_asset_path(filename):
    return os.path.join(os.path.dirname(__file__), 'asset', filename)

neuroscan_logo_path = get_asset_path('Logo.png')

if os.path.exists(neuroscan_logo_path):
    st.set_page_config(
        page_title="NeuroScan - About",
        page_icon=Image.open(neuroscan_logo_path),
        layout="wide"
    )
else:
    st.set_page_config(
        page_title="NeuroScan - About",
        page_icon=Image.open(neuroscan_logo_path),
        layout="wide"
    )
    
# style
st.markdown("""
<style>
    /* Base responsive styles */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Mobile specific adjustments */
    @media (max-width: 640px) {
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        p, li {
            font-size: 0.9rem !important;
        }
        .main .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
    }
    
    /* Improve spacing on mobile */
    .stImage {
        margin-bottom: 1rem;
    }
    
    /* Make sure images don't overflow on mobile */
    img {
        max-width: 100%;
        height: auto;
    }
    
    /* Better column layout on mobile */
    @media (max-width: 640px) {
        .row-widget.stHorizontal {
            flex-direction: column;
        }
        .row-widget.stHorizontal > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
    
    /* Styling for the logout button */
    .logout-btn {
        background-color: #f44336 !important;
        color: white !important;
        font-weight: 500 !important;
        border: none !important;
        padding: 8px 12px !important;
        border-radius: 5px !important;
        transition: all 0.3s !important;
        margin-top: 10px;
    }
    
    .logout-btn:hover {
        background-color: #d32f2f !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard
if os.path.exists(neuroscan_logo_path):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(Image.open(neuroscan_logo_path), width=120, use_container_width=True)
    with col2:
        st.title("NeuroScan - Epileptic Seizure Recognition")
else:
    st.title("NeuroScan - Epileptic Seizure Recognition")

# sidebar
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Dashboard", "Dataset", "Pengujian", "Tentang Peneliti"],
        icons=["bar-chart", "folder", "gear", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

@st.cache_resource
def load_artifacts():
    encoder   = load_model("encoder.keras", compile=False)      
    clf       = joblib.load("voting_model.pkl")            
    scaler    = joblib.load("scaler.pkl")                      
    return encoder, clf, scaler

encoder, clf, scaler = load_artifacts()

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

        X_scaled = scaler.transform(df.values)
        X_resh   = X_scaled.reshape((-1, 178, 1))
        latent   = encoder.predict(X_resh, verbose=0)
        X_flat   = latent.reshape((latent.shape[0], -1))
        
        y_pred_proba = clf.predict_proba(X_flat)[:, 1]
        y_pred       = (y_pred_proba >= 0.4).astype(int)
        
        if y_pred[0] == 1:
            st.markdown("# ⚠️ **Seizure Detected** ⚠️", unsafe_allow_html=True)
            st.markdown("#### **Recommended Action:** Seek immediate medical attention.")
        else:
            st.markdown("# ✅ **No Seizure Detected**", unsafe_allow_html=True)
            st.markdown("#### **Recommended Action:** Continue monitoring as usual.")
