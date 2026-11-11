import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2

# ==========================================
# 1. KONFIGURASI HALAMAN WEB
# ==========================================
st.set_page_config(page_title="Sistem Penghitung Kalori", page_icon="🍲", layout="wide")

st.title("🍲 Sistem Deteksi Makanan & Estimasi Kalori")
st.write("Unggah foto makanan Anda, dan biarkan AI menghitung kalori serta gizinya secara otomatis.")

# ==========================================
# 2. LOAD MODEL & DATABASE (Caching agar tidak loading terus)
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di folder yang sama
    return YOLO('best.pt')

@st.cache_data
def load_data():
    # Pastikan file database_gizi.csv ada di folder yang sama
    return pd.read_csv('database_gizi.csv')

model = load_model()
df_gizi = load_data()

# ==========================================
# 3. TAMPILAN UPLOAD GAMBAR
# ==========================================
uploaded_file = st.file_uploader("Pilih foto makanan (format JPG/PNG)...", type=["jpg", "jpeg", "png"])

# Asumsi porsi (bisa diubah oleh pengguna melalui slider)
st.sidebar.header("Pengaturan Porsi")
asumsi_gram = st.sidebar.slider("Asumsi berat per 1 porsi deteksi (gram):", min_value=50, max_value=300, value=100, step=10)

# ==========================================
# 4. PROSES DETEKSI & HITUNG KALORI
# ==========================================
if uploaded_file is not None:
    # Buka gambar
    image = Image.open(uploaded_file)
    
    # Bagi layar menjadi 2 kolom
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("Hasil Deteksi AI")
        with st.spinner('AI sedang mendeteksi makanan...'):
            # Lakukan deteksi
            results = model(image)
            
            # Plot hasil deteksi (kotak-kotak)
            res_plotted = results[0].plot()
            # Konversi warna BGR ke RGB untuk Streamlit
            res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            st.image(res_image, use_container_width=True)

    # --- Tampilkan Hasil Teks ---
    st.divider()
    st.subheader("📊 Hasil Analisis Gizi")
    
    detected_foods = []
    # Ekstrak hasil dari YOLO
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            detected_foods.append((class_name, confidence))

    if not detected_foods:
        st.warning("AI tidak mendeteksi makanan apa pun di gambar ini.")
    else:
        total_kalori = 0
        total_karbo = 0
        total_protein = 0
        total_lemak = 0
        
        # Tampilkan data per makanan
        for food, conf in detected_foods:
            # Cari di database
            match = df_gizi[df_gizi['Nama_Makanan'].str.lower() == food.lower()]
            
            if not match.empty:
                kalori_100g = match['Kalori'].values[0]
                karbo_100g = match['Karbohidrat'].values[0]
                prot_100g = match['Protein'].values[0]
                lemak_100g = match['Lemak'].values[0]
                
                # Hitung berdasarkan asumsi porsi
                faktor = asumsi_gram / 100.0
                kalori_item = kalori_100g * faktor
                karbo_item = karbo_100g * faktor
                prot_item = prot_100g * faktor
                lemak_item = lemak_100g * faktor
                
                # Tambahkan ke total
                total_kalori += kalori_item
                total_karbo += karbo_item
                total_protein += prot_item
                total_lemak += lemak_item
                
                st.write(f"✅ **{food}** (Keyakinan AI: {conf*100:.1f}%)")
            else:
                st.write(f"⚠️ **{food}** (Terdeteksi, tapi data gizinya tidak ada di database)")

        # Tampilkan Total dalam bentuk Metrik yang menarik
        st.write("---")
        st.write(f"**Total Estimasi untuk porsi {asumsi_gram} gram per item:**")
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Kalori", f"{total_kalori:.1f} kkal")
        m2.metric("Karbohidrat", f"{total_karbo:.1f} g")
        m3.metric("Protein", f"{total_protein:.1f} g")
        m4.metric("Lemak", f"{total_lemak:.1f} g")
        
        st.info("Catatan: Nilai gizi di atas adalah estimasi berdasarkan porsi standar. Nilai sebenarnya dapat bervariasi tergantung pada cara memasak dan bahan tambahan.")