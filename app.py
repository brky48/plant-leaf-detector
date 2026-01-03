import os
# --- STEP 1: FORCE LEGACY KERAS ---
# This must be set BEFORE importing tensorflow to avoid input tensor conflicts
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import tf_keras as keras  # Use the legacy loader for stability
from PIL import Image
import numpy as np
import json
import pandas as pd
from huggingface_hub import hf_hub_download

# --- STEP 2: MIXED PRECISION POLICY ---
# Matching your training environment for float16 weights
from tensorflow.keras import mixed_precision
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
except:
    pass 

# --- STEP 3: PAGE CONFIGURATION ---
st.set_page_config(page_title="PlantAI - Decision Support", layout="wide", page_icon="🌿")

# --- STEP 4: RESOURCE LOADING (CACHED) ---
@st.cache_resource
def load_resources():
    """
    Downloads the model from Hugging Face and loads local metadata.
    """
    REPO_ID = "berkay48/plant-leaf-detector" 
    FILENAME = "plant_disease_detector_best.keras"
    
    # Download from Hugging Face
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    
    # CRITICAL FIX: Use 'keras' (tf_keras) instead of 'tf.keras' 
    # and compile=False to bypass training-specific input errors
    model = keras.models.load_model(model_path, compile=False)
    
    # Load class indices
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    
    # Load care guides
    with open('plant_care_guides.json', 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    
    # Load performance metrics
    performance_df = pd.read_csv('model_performance.csv')
    
    return model, class_indices, knowledge_base, performance_df

# Execute loading
try:
    model, class_indices, knowledge_base, performance_df = load_resources()
    labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- STEP 5: LANGUAGE SETTINGS ---
language = st.sidebar.selectbox("Language Selection / Dil Seçimi", ["English", "Türkçe"])
lang_code = "en" if language == "English" else "tr"

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

# --- STEP 6: APP TABS ---
tab1, tab2 = st.tabs([f"🔍 {t['tab1']}", f"📊 {t['tab2']}"])

# --- TAB 1: DIAGNOSIS ---
with tab1:
    st.header(f"🌿 {t['header']}")
    uploaded_file = st.file_uploader(t["upload_msg"], type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='User Upload', use_container_width=True)
        
        if st.button(t["btn_predict"]):
            # Resize for InceptionV3
            img = image.resize((299, 299))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            with st.spinner('Analyzing...' if lang_code == "en" else 'Analiz ediliyor...'):
                preds = model.predict(img_array)
                confidence = np.max(preds)
                predicted_label = labels[np.argmax(preds)]

            if confidence < 0.75:
                st.warning(t["confidence_err"])
                st.info(f"System Confidence: {confidence:.2f}")
            else:
                st.success(f"### Result: {predicted_label.replace('___', ' - ')}")
                st.progress(float(confidence))
                
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

# --- TAB 2: ANALYTICS ---
with tab2:
    st.header(f"📊 {t['tab2']}")
    st.subheader(t["perf_title"])
    if os.path.exists(t["graph_file"]):
        st.image(t["graph_file"], use_container_width=True)
    
    st.divider()
    st.subheader(t["csv_title"])
    st.dataframe(
        performance_df.style.background_gradient(cmap='YlGn', subset=['f1-score']), 
        use_container_width=True
    )

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("👤 **Developer:** Berkay")
st.sidebar.caption("MIS Graduation Project - 2026")
