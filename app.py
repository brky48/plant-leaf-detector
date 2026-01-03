import streamlit as st
import os

# Keras 3 çakışmalarını önlemek için arka planı sabitle
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from PIL import Image
import numpy as np
import json
import pandas as pd
from huggingface_hub import hf_hub_download

# --- PAGE CONFIG ---
st.set_page_config(page_title="PlantAI - Decision Support", layout="wide", page_icon="🌿")

@st.cache_resource
def load_resources():
    REPO_ID = "berkay48/plant-leaf-detector" 
    FILENAME = "plant_disease_detector_best.keras"
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # MIXED PRECISION OLMADAN, STANDART YÜKLEME
    # compile=False: Eğitim metadatalarını yükleme
    # safe_mode=False: InceptionV3 grafik yapısına izin ver
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    with open('plant_care_guides.json', 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    performance_df = pd.read_csv('model_performance.csv')
    
    return model, class_indices, knowledge_base, performance_df

# Execute resource loading with error handling
try:
    model, class_indices, knowledge_base, performance_df = load_resources()
    labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- STEP 4: SIDEBAR & LANGUAGE SETTINGS ---
st.sidebar.title("Settings / Ayarlar")
language = st.sidebar.selectbox("Language Selection / Dil Seçimi", ["English", "Türkçe"])
lang_code = "en" if language == "English" else "tr"

# Dictionary for multi-language UI support
t = {
    "tab1": "Diagnosis" if lang_code == "en" else "Teşhis",
    "tab2": "Model Performance" if lang_code == "en" else "Model Performansı",
    "header": "Plant Health Analysis" if lang_code == "en" else "Bitki Sağlığı Analizi",
    "upload_msg": "Upload a leaf photo" if lang_code == "en" else "Bir yaprak fotoğrafı yükleyin",
    "btn_predict": "Analyze Plant" if lang_code == "en" else "Bitkiyi Analiz Et",
    "confidence_err": "⚠️ Image rejected. This does not look like a leaf from our dataset." if lang_code == "en" else "⚠️ Görsel reddedildi. Veri setimizdeki bir yaprağa benzemiyor.",
    "expander_title": "Detailed Care Guide" if lang_code == "en" else "Detaylı Bakım Rehberi",
    "status": "Status" if lang_code == "en" else "Durum",
    "treatment": "Treatment" if lang_code == "en" else "Tedavi",
    "maintenance": "Maintenance" if lang_code == "en" else "Bakım",
    "irrigation": "Irrigation" if lang_code == "en" else "Sulama",
    "fertilizer": "Fertilizer" if lang_code == "en" else "Gübreleme",
    "perf_title": "Training Curves" if lang_code == "en" else "Eğitim Grafikleri",
    "csv_title": "Class-wise Statistics" if lang_code == "en" else "Sınıf Bazlı İstatistikler",
    "graph_file": "model_graph_en.png" if lang_code == "en" else "model_graph_tr.png"
}

# --- STEP 5: APP TABS ---
tab1, tab2 = st.tabs([f"🔍 {t['tab1']}", f"📊 {t['tab2']}"])

# --- TAB 1: DIAGNOSIS & RECOMMENDATIONS ---
with tab1:
    st.header(f"🌿 {t['header']}")
    uploaded_file = st.file_uploader(t["upload_msg"], type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='User Upload', use_container_width=True)
        
        if st.button(t["btn_predict"]):
            # Preprocessing for InceptionV3
            img = image.resize((299, 299))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner('Analyzing...' if lang_code == "en" else 'Analiz ediliyor...'):
                # Prediction using modern Keras API
                preds = model.predict(img_array)
                confidence = np.max(preds)
                predicted_label = labels[np.argmax(preds)]

            # Confidence Threshold Logic
            if confidence < 0.75:
                st.warning(t["confidence_err"])
                st.info(f"System Confidence: {confidence:.2f}")
            else:
                st.success(f"### Result: {predicted_label.replace('___', ' - ')}")
                st.progress(float(confidence))
                
                # Knowledge Base Integration
                info = knowledge_base.get(predicted_label, {}).get(lang_code)
                if info:
                    with st.expander(f"💡 {t['expander_title']}", expanded=True):
                        st.markdown(f"**{t['status']}:** {info['status']}")
                        if "treatment" in info:
                            st.error(f"💊 **{t['treatment']}:** {info['treatment']}")
                        else:
                            st.success(f"✨ **{t['maintenance']}:** {info['maintenance']}")
                        
                        st.info(f"💧 **{t['irrigation']}:** {info['irrigation']}")
                        st.info(f"🧪 **{t['fertilizer']}:** {info['fertilizer']}")

# --- TAB 2: ANALYTICS & PERFORMANCE ---
with tab2:
    st.header(f"📊 {t['tab2']}")
    
    # Visualizing Training Curves
    st.subheader(t["perf_title"])
    if os.path.exists(t["graph_file"]):
        st.image(t["graph_file"], use_container_width=True)
    
    st.divider()
    
    # Detailed CSV Data Table
    st.subheader(t["csv_title"])
    st.dataframe(
        performance_df.style.background_gradient(cmap='YlGn', subset=['f1-score']), 
        use_container_width=True
    )

# --- STEP 6: FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("👤 **Developer:** Berkay")
st.sidebar.caption("MIS Graduation Project - 2026")
