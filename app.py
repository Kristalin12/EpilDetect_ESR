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
    .big-title {
        font-size: 38px;
        font-weight: 900;
        color: #1a1a1a;
    }
    .section-subtitle {
        font-size: 18px;
        font-weight: 400;
        color: #333333;
    }
    .box {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 20px;
        background-color: #fafafa;
        margin: 10px;
    }
    .box-title {
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
        color: #2c2c2c;
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
    st.header("▶️Tentang NeuroScan")
    st.markdown("""
        **Apa itu NeuroScan?**  
        NeuroScan adalah implementasi sistem deteksi kejang epilepsi secara otomatis menggunakan metode Convolutional Autoencoder untuk ekstraksi fitur dan gabnungan Regresi Logistik dan SVM menggunakan soft voting untuk klasifikasi sebagai hasil dari penelitian “Optimasi Regresi Logistik dan Support Vector Machine Menggunakan Convolutional Autoencoder Untuk Deteksi Kejang Epilepsi”..

        **Model CAE dan LR-SVM:**  
        
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
