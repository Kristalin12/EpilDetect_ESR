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

# Helper – asset path (logo, icons, etc.)
def get_asset_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "asset", filename)

logo_path = get_asset_path("Logo.png")
logo_img  = Image.open(logo_path) if os.path.exists(logo_path) else None
    
# Global CSS styling
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

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Dashboard", "Dataset", "Pengujian", "Tentang Peneliti"],
        icons=["bar-chart", "folder", "cpu", "person-circle"],
        menu_icon="cast",
        default_index=0,
    )


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

# Sidebar
with st.sidebar:
    selected = option_menu(
        "Menu",
        ["Dashboard", "Dataset", "Pengujian", "Tentang Peneliti"],
        icons=["bar-chart", "folder", "gear", "info-circle"],
        menu_icon="cast",
        default_index=0,
    )

# Dashboard
if selected == "Dashboard":
    st.title("📊 Dashboard NeuroScan")
    st.markdown("""
        **Apa itu NeuroScan?**  
        NeuroScan adalah platform berbasis EEG (Electroencephalogram) untuk mendeteksi kejang epilepsi secara otomatis menggunakan metode pembelajaran mesin.

        **Tujuan Dashboard Ini:**  
        - Memberikan penjelasan dasar tentang epilepsi dan pentingnya deteksi dini
        - Menjelaskan cara kerja sistem prediksi
        - Menyediakan visualisasi hasil prediksi secara mudah dan intuitif

        **Fitur Utama:**  
        - Lihat dataset pelatihan  
        - Uji data EEG baru  
        - Lihat prediksi secara langsung  
        - Pelajari latar belakang dan peneliti

        **Metode yang Digunakan:**  
        - **CNN Autoencoder**: Untuk ekstraksi fitur dari sinyal EEG  
        - **Voting Classifier (Logistic Regression + SVM)**: Untuk klasifikasi kejang  
        - **Standard Scaler**: Untuk normalisasi data
    """)
