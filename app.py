import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib 
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
from PIL import Image

# Page Config
st.set_page_config(page_title="Epileptic Seizure Recognition", layout="wide")

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
    /* Main background */
    .stApp {
        background-color: #2E0F15 !important;
        color: #FAFFF0 !important;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    .css-1d391kg {
        background-color: #842C35 !important;
        backdrop-filter: blur(6px);
        color: #FAFFF0 !important;
        border-right: 1px solid #EB5456;
    }

    /* Headings */
    h1, h2, h3 {
        color: #FF9FA2;
    }

    /* Metrics */
    [data-testid="metric-container"] {
        background-color: #EB5456;
        color: #FAFFF0;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(216, 76, 76, 0.3);
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        background: #842C35;
        border-radius: 10px;
    }
    .stTabs [role="tab"] {
        color: #FAFFF0;
        padding: 10px;
    }
    .stTabs [aria-selected="true"] {
        border-bottom: 4px solid #FF6968;
        color: #FF9FA2;
    }

    /* Buttons */
    .stButton>button {
        background-color: #D84C4C;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #B43E3E;
        transition: 0.3s ease-in-out;
    }

    /* Inputs */
    .css-1cpxqw2 {
        background-color: #3A1D20 !important;
        border: 1px solid #EB5456 !important;
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)

# Dashboard Header
if os.path.exists(neuroscan_logo_path):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(Image.open(neuroscan_logo_path), width=120, output_format="auto")
    with col2:
        st.markdown("<h1>NeuroScan</h1>", unsafe_allow_html=True)
        st.markdown("<h2>EEG-based epileptic seizure recognition<h2>")
else:
    st.markdown("<h1>NeuroScan</h1>", unsafe_allow_html=True)

st.markdown("---")
with st.expander("🔍 Tentang NeuroScan"):
    st.markdown("""
    **What is Epilepsy?**  
    Epilepsy is a brain disorder that causes repeated seizures. Seizures are sudden bursts of electrical activity in the brain that can affect behavior, movement, or awareness.

    **What is a Seizure?**  
    A seizure is like a short circuit in the brain. It can cause jerking movements, confusion, staring spells, or even loss of consciousness — depending on the type.

    **What is DALY?**  
    DALY stands for *Disability-Adjusted Life Year*. It measures how much healthy life is lost due to illness or death. The higher the number, the bigger the impact of epilepsy on a person's life.

    **What does this dashboard do?**  
    This dashboard shows epilepsy data for different age groups. It lets you:
    - View current and past epilepsy burden
    - Predict future trends using machine learning
    - Explore patterns by age and gender

    **What ML (Machine Learning) is used?**  
    A simple method called **Linear Regression** is used here to predict future values. It looks at trends from the past and draws a line to guess future numbers.
    """)

st.markdown("---")

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
