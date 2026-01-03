#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 13:37:52 2026

@author: berkay
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import pandas as pd
import os
from huggingface_hub import hf_hub_download

# --- 1. SAYFA AYARLARI ---
st.set_page_config(page_title="PlantAI - Decision Support", layout="wide", page_icon="🌿")

# --- 2. HUGGING FACE'DEN MODELİ ÇEKME ---
@st.cache_resource
def load_model_from_hf():
    # REPO_ID: 'kullanici_adin/model_depo_adin' şeklinde güncelle
    REPO_ID = "berkay48/plant-leaf-detector" 
    FILENAME = "plant_disease_detector_best.keras"
    
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    model = tf.keras.models.load_model(model_path)
    return model

@st.cache_resource
def load_data():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    with open('plant_care_guides.json', 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    performance_df = pd.read_csv('model_performance.csv')
    return class_indices, knowledge_base, performance_df

# Kaynakları Yükle
try:
    model = load_model_from_hf()
    class_indices, knowledge_base, performance_df = load_data()
    labels = {v: k for k, v in class_indices.items()}
except Exception as e:
    st.error(f"Initialization Error: {e}")
    st.stop()

# --- 3. DİL SEÇİMİ VE SÖZLÜK ---
language = st.sidebar.selectbox("Language / Dil", ["English", "Türkçe"])
lang_code = "en" if language == "English" else "tr"

t = {
    "tab1": "Diagnosis" if lang_code == "en" else "Teşhis",
    "tab2": "Model Performance" if lang_code == "en" else "Model Performansı",
    "perf_title": "Training & Validation Metrics" if lang_code == "en" else "Eğitim ve Doğrulama Metrikleri",
    "csv_title": "Detailed Class Statistics" if lang_code == "en" else "Detaylı Sınıf İstatistikleri",
    "graph_file": "model_graph_en.png" if lang_code == "en" else "model_graph_tr.png"
}

# --- 4. TABS (SEKMELER) ---
tab1, tab2 = st.tabs([t["tab1"], t["tab2"]])

# --- TAB 1: TEŞHİS EKRANI ---
with tab1:
    st.header("🌿 " + ("Plant Health Diagnosis" if lang_code == "en" else "Bitki Sağlığı Teşhisi"))
    # (Önceki yazdığımız teşhis kodlarını buraya yapıştırabilirsin)
    # ... (Görsel yükleme, tahmin ve bilgi tabanı gösterimi)

# --- TAB 2: MODEL PERFORMANSI ---
with tab2:
    st.header("📊 " + t["tab2"])
    
    # A. Grafik Gösterimi (CSV'nin üzerinde)
    st.subheader(t["perf_title"])
    if os.path.exists(t["graph_file"]):
        st.image(t["graph_file"], use_container_width=True)
    else:
        st.warning("Graphic file not found / Grafik dosyası bulunamadı.")
    
    st.divider()
    
    # B. CSV Veri Tablosu
    st.subheader(t["csv_title"])
    st.dataframe(performance_df.style.background_gradient(cmap='Greens', subset=['f1-score']), use_container_width=True)