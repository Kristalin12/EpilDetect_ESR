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
        options=["Dashboard", "Dataset", "Klasifikasi EEG", "Tentang Peneliti"],
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
        st.image("asset/epilepsi.jpg", caption="", width=600, use_container_width=True)
    with col2:
        st.markdown("""
        <div class='content'>
            <p style='text-align: justify;'>
                Menurut International League Against Epilepsy (ILAE), epilepsi merupakan penyakit yang ditandai dengan kecenderungan terus-menerus untuk menimbulkan kejang epilepsi, yang secara praktis dapat dioperasionalkan dengan adanya dua kejang yang tidak diprovokasi atau tidak memiliki faktor langsung dengan jarak antar kejang lebih dari 24 jam (Fisher et al., 2014).
            </p>
            <p style='text-align: justify;'>
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
    st.markdown("## Visualisasi Distribusi Kelas") 
    col1, col2= st.columns([4, 4])        
    with col1:
        st.image("asset/Distribusi_Kelas_Bar.png", caption="", width=900, use_container_width=True)
    with col2:
        st.markdown("""
            <div class='content'>
                <p style='text-align: justify;'>
                Dataset EEG yang digunakan pada penelitian ini memiliki distribusi kelas yang tidak seimbang antara data kejang dan non-kejang dengan data kejang dengan data <b>non-kejang</b> sebesar <b>9200</b> dan data <b>kejang</b> sebesar <b>2300</b>.
                </p>
                <p style='text-align: justify;'>
                Oleh karena itu, dataset ini juga perlu diproses melalui penyeimbangan data agar model klasifikasi tidak bias terhadap kelas mayoritas.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
#---Uji---
elif selected == 'Klasifikasi EEG':
    @st.cache_resource
    def load_artifacts():
        encoder   = load_model("encoder.keras", compile=False)      
        clf       = joblib.load("voting_model.pkl")            
        scaler    = joblib.load("scaler.pkl")                      
        return encoder, clf, scaler
    
    encoder, clf, scaler = load_artifacts()
    
    st.title("Pengujian Model")
    st.caption("Upload CSV file dengan 178 fitur EEG")
    
    uploaded_file = st.file_uploader("📂 Drag and drop file here", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if df.shape[1] != 178:
            st.error("❌ CSV harus memiliki 178 columns (fitur)")
        else:
            clr = {0: "royalblue", 1: "crimson"}
            lbl = {0: "Non‑Seizure", 1: "Seizure"}

            st.markdown("---")
            st.subheader("📈 EEG Signal")
            st.write("Input data:")
            st.write(df)

            X_scaled = scaler.transform(df.values)
            X_resh   = X_scaled.reshape((-1, 178, 1))
            latent   = encoder.predict(X_resh, verbose=0)
            X_flat   = latent.reshape((latent.shape[0], -1))
        
            y_pred_proba = clf.predict_proba(X_flat)[:, 1]
            y_pred       = (y_pred_proba >= 0.4).astype(int)
            
            fig, ax = plt.subplots(figsize=(4, 2.5))
            for i, row in df.iterrows():
                y = y_pred[i]
                ax.plot(
                    row.values,
                    color=clr[y],
                    linewidth=0.8,
                    alpha=0.6,
                    label=lbl[y] if i == 0 else "")
                
            ax.set_xlabel("Index", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.set_title("EEG Signal (from uploaded CSV)")
            st.pyplot(fig, use_container_width=False)
            
            if y_pred[0] == 1:
                st.error(" The person is experiencing a seizure!", icon="😟")
                st.markdown("---")
                st.markdown("## ⚠️ Saran Dokter yang Penting ⚠️")
                st.markdown("""
                    - Minum obat secara konsisten pada waktu yang sama setiap hari.
                    - Hindari pemicu seperti kurang tidur, lampu yang menyala, atau alkohol.
                    - Buat catatan harian kejang untuk melacak episodenya.
                    - Beri tahu kontak dekat tentang langkah pertolongan pertama selama kejang.
                    - Jangan pernah menghentikan pengobatan tanpa saran medis.
                """)
            else:
                st.success(" No seizure detected. All clear!", icon="😌")
                st.markdown("---")
                st.markdown("## ⚠️ Saran Dokter yang Penting ⚠️")
                st.markdown("""
                    - Menjaga kesehatan secara umum dengan diet yang sehat dan olahraga.
                    - Tidur yang cukup dan hindari alkohol atau narkoba.
                    - Menjalani vaksinasi yang dianjurkan.
                    - Mencegah cedera kepala dengan menggunakan pengaman dan berhati-hati.
                """)
                
elif selected == 'Tentang Peneliti':
    col1, col2 = st.columns([1, 9])
    with col1:
        st.image("asset/user.png", caption="", width=500, use_container_width=True)
    with col2:
        st.header("Kristalina Chandra Ratu")
    st.markdown("""
        <div class='content'>
            <p style='text-align: justify;'>
                Penulis bernama lengkap Kristalina Chandra Ratu lahir di Salatiga, pada tanggal 13 September 2003. Penulis merupakan anak kedua dari dua bersaudara dari pasangan Sangkan Kabut Panangsang dan Tutik Setyowati. Penulis saat ini tinggal di Ds. Tetep RT. 04 RW. 03, Kecamatan Randuacir, Kelurahan Argomulyo, Kota Salatiga, Jawa Tengah.
            </p>
            <p style='text-align: justify;'>
                Penulis menempuh pendidikan sarjana di Program Studi Teknik Informatika, Fakultas Matematika dan Ilmu Pengetahuan Alam, Universitas Negeri Semarang. Penulis mendalami beberapa bidang teknologi seperti UI/UX, hingga machine learning dan deep learning yang menjadi topik penelitian ini. Penulis juga mengikuti kegiatan-kegiatan mahasiswa, seperti manjadi bagian kepengurusan Unit Kegiatan Mahasiswa Seni Rupa Desain, Studi Independen Kampus Merdeka batch 6, serta magang PRIGEL di PT. Global Data Inspirasi dan CV. Serpihan Tech Solution.
            </p>
        </div>
        """, unsafe_allow_html=True)
