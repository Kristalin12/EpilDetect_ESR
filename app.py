import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import joblib 
import numpy as np
import seaborn as sns
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
        options=["Dashboard", "Dataset", "Model", "Klasifikasi EEG", "Tentang Peneliti"],
        default_index=0,
        styles={
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"5px"},
            "nav-link-selected": {"background-color": "#00ace6", "color": "white"},
        }
    )

# Dashboard
if selected == "Dashboard":
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
        margin-bottom: 20px;
    }
    h1, h2 {
        color: #4ecdc4;
    }
    .image {
            width: 100%;
            border-radius: 20px;
            margin-top: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(Image.open(logo_path), width=150, output_format="auto")
    with col2:
        st.title("NeuroScan - Epileptic Seizure Recognition")
        st.markdown("<p style='font-size:1rem;'>Mendeteksi Kejang Epilepsi Berbasis EEG dengan Teknik Otomatisasi</p>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("""
    <div class='content'>
        <h2>Tentang NeuroScan</h2>
        <p style='text-align: justify;'>
            <strong>NeuroScan</strong> merupakan implementasi sistem deteksi kejang epilepsi secara otomatis dan merupakan hasil dari penelitian <strong>“Optimasi Regresi Logistik dan Support Vector Machine Menggunakan Convolutional Autoencoder Untuk Deteksi Kejang Epilepsi”</strong>.
        </p>
        <p style='text-align: justify;'>
            NeuroScan akan memproses data EEG yang memiliki fitur sebanyak 178. Sistem akan melakukan ekstraksi fitur untuk menghasilkan fitur laten dari model Convolutional Autoencoder, yang kemudian akan dimasukkan pada model klasifikasi oleh Regresi Logistik dan Support Vector Machine. Hasil prediksi merupakan nilai rata-rata dari kedua model.
        </p>
    </div>
    """, unsafe_allow_html=True)
    # ---Penjelasan Epilepsi---
    st.markdown("---")
    st.markdown("""
    <div class='content'>
        <h2>Penyakit Epilepsi</h2>
    <div>
    """, unsafe_allow_html=True)
    col1, col2 = st.columns([2, 3])
    with col1:
        st.image("asset/epilepsi.jpg", caption="", width=900, use_container_width=True)
    with col2:
        st.markdown("""
        <div class='content'>
            <p style='text-align: justify;'>
                Menurut International League Against Epilepsy (ILAE), epilepsi merupakan penyakit yang ditandai dengan kecenderungan terus-menerus untuk menimbulkan kejang epilepsi, yang secara praktis dapat dioperasionalkan dengan adanya dua kejang yang tidak diprovokasi atau tidak memiliki faktor langsung dengan jarak antar kejang lebih dari 24 jam (Fisher et al., 2014).
                Kejang epilepsi sendiri merupakan kondisi sesaat yang terjadi akibat aktivitas neuron di otak yang tidak normal, berlebihan, atau terjadi secara sinkron (Fisher et al., 2005). Menurut World Health Organization (WHO, 2024), lima puluh juta orang diperkirakan telah didiagnosis dengan penyakit epilepsi.
            </p>
        <div>
        """, unsafe_allow_html=True)
    #---Penjelasan Model---
    st.markdown("---")
    st.markdown("""
    <div class='content'>
        <h2>Model NeuroScan</h2>
    <div>
    """, unsafe_allow_html=True)
    with st.expander("**Convolutional Autoencoder**", expanded=True):
        st.markdown("""
        Algoritma ini digunakan untuk menurunkan data masukan menjadi fitur representasi yang berdimensi lebih rendah dan memungkinkan algoritma ini untuk mempelajari pola representatif dalam ruang laten.
        """)
    with st.expander("**Logistic Regression**", expanded=True):
        st.markdown("""
        Teknik statistik yang dapat menangkap dan menafsirkan hubungan antara prediktor dan tanggapan dikotomis, seperti ada atau tidaknya penyakit.
        """)
    with st.expander("**Support Vector Machine**", expanded=True):
        st.markdown("""
        Algoritma machine learning yang digunakan untuk klasifikasi dan regresi, dengan mencari hyperplane terbaik untuk memisahkan data menjadi beberapa kelas dengan margin yang maksimal.
        """)
    with st.expander("**Soft Voting Classifier**", expanded=True):
        st.markdown("""
        Teknik yang melakukan prediksi dengan penggabungan beberapa algoritma, dalam hal ini regresi logistik dan SVM, yang kemudian digabungkan untuk mendapatkan jumlah probabilitas yang tertimbang..
        """)

#---Dataset---
elif selected == 'Dataset':
    st.markdown("""
    <div class='content'>
        <h2>Epileptic Seizure Recognition Dataset</h2>
        <p style='text-align: justify;'>
            Dataset yang digunakan untuk penelitian ini adalah dataset epileptic seizure recognition dari kaggle, yang merupakan versi dataset Universitas Bonn yang telah direkonstruksi dan telah dibentuk kembali. Pada dataset ini 4097 titik data dibagi dan diacak menjadi 23 potongan. Oleh karena itu, setiap segmen pada dataset memiliki 23 potongan sehingga 23 x 500 menghasilkan 11.500 baris elemen informasi yang membentuk data, dengan setiap informasi berisikan 178 titik data selama satu detik. 
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("Dataset epileptic seizure recognition dapat diunduh pada tombol berikut")
    st.link_button("Epileptic Seizure Detection", "https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition/data")
    
    st.markdown("---")
    st.header("Tabel Data Lengkap")
    df = pd.read_csv("Epileptic_Seizure_Recognition.csv")
    st.dataframe(df)
    
    st.markdown("---")
    st.header("Visualisasi Data")
    df['y'] = df['y'].astype(int)
    df = df.select_dtypes(include=[np.number])
    class_labels = sorted(df['y'].unique())
    class_labels_str = [str(label) for label in class_labels]
    
    selected_str = st.selectbox("Pilih Kategori (y)", class_labels_str)
    selected_class = int(selected_str)
    filtered_df = df[df['y'] == selected_class]
    
    if filtered_df.empty:
        st.warning("Tidak ada data untuk kelas yang dipilih.")
    else:
        signal = filtered_df.iloc[0, :-1].astype(float)  # Exclude 'y' and convert to float
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(signal)
        ax.set_title(f"Sinyal EEG - Contoh dari Kelas {selected_class}")
        ax.set_xlabel("Index Sinyal")
        ax.set_ylabel("Amplitudo")
        st.pyplot(fig)
    st.markdown("### Visualisasi Distribusi Kelas")
    df['seizure_status'] = df['y'].apply(lambda x: 'Seizure' if x == 1 else 'Non-Seizure')
    status_counts = df['seizure_status'].value_counts()
    
    col1, col2 = st.columns([2, 2])
    with col1:
        fig_bar, ax_bar = plt.subplots(figsize=(6, 4)) 
        sns.barplot(x=status_counts.index, y=status_counts.values, palette='pastel', ax=ax_bar)
        ax_bar.set_title("Bar Chart")
        ax_bar.set_xlabel("Kategori")
        ax_bar.set_ylabel("Jumlah Sampel")
        st.pyplot(fig_bar, use_container_width=True)
    with col2:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        ax_pie.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
        ax_pie.set_title("Pie Chart")
        st.pyplot(fig_pie, use_container_width=True)
