import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib 
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu
from PIL import Image

# --- Page Config ---
st.set_page_config(page_title="Epileptic Seizure Recognition", layout="wide")

# --- Helper ---
def get_asset_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "asset", filename)

logo_path = get_asset_path("Logo.png")
logo_img  = Image.open(logo_path) if os.path.exists(logo_path) else None

if os.path.exists(logo_path):
    st.set_page_config(
        page_title="NeuroScan - About",
        page_icon=Image.open(logo_path),
        layout="wide"
    )
else:
    st.set_page_config(
        page_title="NeuroScan - About",
        page_icon=Image.open(logo_path),
        layout="wide"
    )

# --- CSS styling ---
st.markdown("""
    <style>
    /* Base responsive styles */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #1a535c;
        text-align: center;
        padding-top: 50px;
    }
    .description {
        font-size: 20px;
        color: #4ecdc4;
        text-align: center;
        padding-bottom: 30px;
    }
    .file-uploader {
        text-align: center;
        padding-top: 50px;
        padding-bottom: 50px;
    }
    .footer {
        font-size: 14px;
        color: #ffffff;
        text-align: center;
        padding-top: 30px;
        padding-bottom: 20px;
    }
    .section-subtitle {
        font-size: 18px;
        font-weight: 400;
        color: #333333;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        padding-top: 30px;
    }
    .horizontal-scroll {
        display: flex;
        overflow-x: auto;
        padding: 10px;
        gap: 20px;
    }
    .horizontal-scroll::-webkit-scrollbar {
        height: 8px;
    }
    .horizontal-scroll::-webkit-scrollbar-thumb {
        background: #ccc;
        border-radius: 4px;
    }
    .card {
        min-width: 250px;
        max-width: 300px;
        padding: 20px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        flex-shrink: 0;
    }
    .card-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 10px;
    }
    .scroll-container {
        height: 500px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
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

# Sidebar menu
st.sidebar.markdown("## 🧠 NeuroScan")
with st.sidebar:
    selected = option_menu(
        menu_title="",
        options=["🏠 Dashboard", "📊Dataset", "🧬Klasifikasi EEG", "👤Tentang Peneliti"],
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f8f9fa"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px"},
            "nav-link-selected": {"background-color": "#4ecdc4", "color": "white"},
        }
    )

# Dashboard
if selected == "🏠 Dashboard":
    st.markdown("""
    <style>
    .header{
        padding: 20px;
        background-color: #1a535c;
        color: #ffffff;
        text-align: center;
        margin-bottom: 20px;
    }
    .content{
        background-color: #ffffff;
        margin-bottom: 20px;
    }
    h1, h2 {
        color: #4ecdc4;
    }
    .image {
            width: 100%;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(Image.open(logo_path), width=150, output_format="auto")
    with col2:
        st.title("NeuroScan - Epileptic Seizure Recognition")
        st.markdown("<p style='font-size:1rem;'>Mendeteksi Kejang Epilepsi Berbasis Data EEG dengan Teknik Otomatisasi</p>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("""
    <div class='content'>
        <h2>Tentang NeuroScan</h2>
        <p style='text-align: justify;'>
            <strong>NeuroScan</strong> merupakan implementasi sistem deteksi kejang epilepsi secara otomatis menggunakan metode Convolutional Autoencoder untuk ekstraksi fitur dan gabnungan Regresi Logistik dan SVM menggunakan soft voting untuk klasifikasi. Sistem ini merupakan hasil dari penelitian <strong>“Optimasi Regresi Logistik dan Support Vector Machine Menggunakan Convolutional Autoencoder Untuk Deteksi Kejang Epilepsi”</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
