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


if os.path.exists(logo_path):
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(Image.open(logo_path), width=120, output_format="auto")
    with col2:
        st.markdown("<h1>NeuroScan</h1>", unsafe_allow_html=True)
        st.markdown("EEG-based epileptic seizure recognition")
else:
    st.markdown("<h1>NeuroScan</h1>", unsafe_allow_html=True)

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
