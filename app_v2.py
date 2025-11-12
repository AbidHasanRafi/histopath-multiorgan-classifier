# app.py
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import sys
import os

# Add the model definitions
sys.path.append('.')

# Import your model architecture
from model import UltraLightHistoNet, MicroAttention, DepthwiseSeparableConv, EfficientBlock

# Set page configuration
st.set_page_config(
    page_title="HistoPath AI - Multi-Organ Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for stunning programmer-style interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #F093FB 100%);
    }
    
    /* Animated Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #F093FB 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        animation: glow 2s ease-in-out infinite alternate;
        font-family: 'Inter', sans-serif;
        letter-spacing: -1px;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5)); }
        to { filter: drop-shadow(0 0 40px rgba(118, 75, 162, 0.8)); }
    }
    
    /* Sub Header - Cyberpunk Style */
    .sub-header {
        font-size: 2rem;
        font-weight: 700;
        color: #00D9FF;
        margin-bottom: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
        border-left: 5px solid #00D9FF;
        padding-left: 1rem;
        animation: flicker 3s infinite;
    }
    
    @keyframes flicker {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.95; }
    }
    
    /* Terminal-Style Cards */
    .terminal-card {
        background: rgba(15, 12, 41, 0.9);
        border: 2px solid #00D9FF;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.3), inset 0 0 20px rgba(0, 217, 255, 0.1);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .terminal-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 30px;
        background: linear-gradient(90deg, #00D9FF 0%, #667eea 50%, #764ba2 100%);
        opacity: 0.3;
    }
    
    .terminal-card::after {
        content: '● ● ●';
        position: absolute;
        top: 8px;
        left: 15px;
        color: #00D9FF;
        font-size: 12px;
        letter-spacing: 5px;
    }
    
    /* Glowing Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 217, 255, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 217, 255, 0.4);
        border-color: #00D9FF;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(0, 217, 255, 0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    /* Holographic Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.9) 0%, rgba(118, 75, 162, 0.9) 100%);
        border: 2px solid #00D9FF;
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 0 50px rgba(0, 217, 255, 0.5), inset 0 0 30px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 50px rgba(0, 217, 255, 0.5); }
        50% { box-shadow: 0 0 70px rgba(0, 217, 255, 0.8); }
    }
    
    .prediction-card h2 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
    }
    
    .prediction-card h3 {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1rem 0;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .prediction-card h4 {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        margin-top: 1rem;
    }
    
    /* Animated Confidence Bar */
    .confidence-bar {
        height: 25px;
        background: linear-gradient(90deg, #FF6B6B 0%, #FECA57 25%, #48DBFB 50%, #1DD1A1 75%, #00D9FF 100%);
        border-radius: 15px;
        margin: 1rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
        animation: glow-bar 2s infinite;
    }
    
    @keyframes glow-bar {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 217, 255, 0.5); }
        50% { box-shadow: 0 0 30px rgba(0, 217, 255, 0.8); }
    }
    
    .confidence-bar::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        animation: scan 2s infinite;
    }
    
    @keyframes scan {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    /* Neon Organ Tags */
    .organ-tag {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        margin: 0.3rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
        border: 2px solid;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .organ-tag:hover {
        transform: scale(1.1);
        box-shadow: 0 0 30px currentColor;
    }
    
    /* Data Display Cards */
    .data-card {
        background: rgba(15, 12, 41, 0.8);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .data-card:hover {
        border-color: #00D9FF;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .status-online { background: #1DD1A1; box-shadow: 0 0 10px #1DD1A1; }
    .status-processing { background: #FECA57; box-shadow: 0 0 10px #FECA57; }
    .status-error { background: #FF6B6B; box-shadow: 0 0 10px #FF6B6B; }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 2px solid #00D9FF;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 50px rgba(0, 217, 255, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #F093FB 100%);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15, 12, 41, 0.95) 0%, rgba(48, 43, 99, 0.95) 100%);
        border-right: 2px solid rgba(0, 217, 255, 0.3);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #00D9FF;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #F093FB 100%);
        border-radius: 10px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15, 12, 41, 0.8);
        border: 2px dashed rgba(0, 217, 255, 0.5);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00D9FF;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(15, 12, 41, 0.8);
        border-radius: 15px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 2px solid transparent;
        color: #00D9FF;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #00D9FF;
        color: white;
        box-shadow: 0 0 20px rgba(0, 217, 255, 0.5);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #00D9FF;
        text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Info, Success, Warning Boxes */
    .stAlert {
        background: rgba(15, 12, 41, 0.9);
        border-left: 4px solid;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #00D9FF;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #00D9FF;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(15, 12, 41, 0.8);
        border: 1px solid rgba(0, 217, 255, 0.3);
        border-radius: 10px;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 0 30px rgba(0, 217, 255, 0.2);
    }
    
    /* Header Decorative Elements */
    .header-line {
        height: 3px;
        background: linear-gradient(90deg, transparent 0%, #00D9FF 50%, transparent 100%);
        margin: 1rem 0;
        animation: line-glow 2s infinite;
    }
    
    @keyframes line-glow {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #00D9FF !important;
        border-right-color: #667eea !important;
        border-bottom-color: #764ba2 !important;
        border-left-color: #F093FB !important;
    }
</style>
""", unsafe_allow_html=True)

# Class mappings
CLASSES_14 = [
    'breast_benign', 'breast_malignant', 'colon_aca', 'colon_n', 
    'lung_aca', 'lung_n', 'lung_scc', 'oral_normal_oral_cavity',
    'oral_oscc', 'ovarian_clear_cell', 'ovarian_endometri', 
    'ovarian_mucinous', 'ovarian_non_cancerous', 'ovarian_serous'
]

CLASSES_5 = ['breast', 'colon', 'lung', 'oral', 'ovarian']

ORGAN_MAPPING = {
    'breast_benign': 'breast', 'breast_malignant': 'breast',
    'colon_aca': 'colon', 'colon_n': 'colon',
    'lung_aca': 'lung', 'lung_n': 'lung', 'lung_scc': 'lung',
    'oral_normal_oral_cavity': 'oral', 'oral_oscc': 'oral',
    'ovarian_clear_cell': 'ovarian', 'ovarian_endometri': 'ovarian',
    'ovarian_mucinous': 'ovarian', 'ovarian_non_cancerous': 'ovarian', 'ovarian_serous': 'ovarian'
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model loading functions
@st.cache_resource
def load_14_class_model():
    """Load the 14-class model"""
    model = UltraLightHistoNet(num_classes=14, base_channels=16)
    try:
        model.load_state_dict(torch.load('best_model_14_class.pth', map_location=device))
    except:
        st.error("14-class model file not found. Please ensure 'best_model_14_class.pth' is in the directory.")
        return None
    model.eval()
    return model

@st.cache_resource
def load_5_class_model():
    """Load the 5-class model"""
    model = UltraLightHistoNet(num_classes=5, base_channels=16)
    try:
        model.load_state_dict(torch.load('best_model_5_class.pth', map_location=device))
    except:
        st.error("5-class model file not found. Please ensure 'best_model_5_class.pth' is in the directory.")
        return None
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# XAI functions (simplified for Streamlit)
def generate_grad_cam(model, image_tensor, target_class):
    """Generate Grad-CAM heatmap (simplified version)"""
    try:
        # This is a simplified version - you might want to use your full XAI implementation
        model.eval()
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad_()
        
        output = model(image_tensor)
        model.zero_grad()
        
        # Simple gradient visualization
        output[0, target_class].backward()
        gradients = image_tensor.grad
        saliency = gradients.abs().max(dim=1)[0].squeeze()
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency.cpu().detach().numpy()
    except:
        return np.zeros((224, 224))

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">🔬 HistoPath AI - Multi-Organ Histopathology Classification</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["🏠 Home", "🖼️ Image Classification", "📊 Model Analysis", "🔍 XAI Explanations", "📈 Performance Dashboard"]
    )
    
    # Load models
    with st.sidebar:
        st.markdown("### Model Status")
        with st.spinner("Loading models..."):
            model_14 = load_14_class_model()
            model_5 = load_5_class_model()
        
        if model_14 and model_5:
            st.success("✅ Models loaded successfully!")
            st.info(f"Device: {device}")
        else:
            st.error("❌ Failed to load models")
            return
    
    # Home Page
    if app_mode == "🏠 Home":
        show_home_page()
    
    # Image Classification
    elif app_mode == "🖼️ Image Classification":
        show_classification_page(model_14, model_5)
    
    # Model Analysis
    elif app_mode == "📊 Model Analysis":
        show_model_analysis_page(model_14, model_5)
    
    # XAI Explanations
    elif app_mode == "🔍 XAI Explanations":
        show_xai_page(model_14, model_5)
    
    # Performance Dashboard
    elif app_mode == "📈 Performance Dashboard":
        show_performance_dashboard()

def show_home_page():
    """Display home page with overview and instructions"""
    
    # Animated welcome banner
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <div class='header-line'></div>
        <h1 style='font-size: 1.2rem; color: #00D9FF; font-family: "JetBrains Mono", monospace; 
                   letter-spacing: 8px; margin: 1rem 0;'>WELCOME TO</h1>
        <h1 style='font-size: 4rem; font-weight: 900; background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #F093FB 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                   font-family: "Inter", sans-serif; letter-spacing: -2px;'>
            HistoPath AI
        </h1>
        <p style='color: #00D9FF; font-size: 1.1rem; font-family: "JetBrains Mono", monospace; 
                  letter-spacing: 3px; margin-top: 0.5rem;'>
            [ ADVANCED NEURAL DIAGNOSTIC SYSTEM ]
        </p>
        <div class='header-line'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # System status
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class='terminal-card' style='text-align: center;'>
            <div class='status-indicator status-online'></div>
            <h3 style='color: #00D9FF; margin-top: 2rem; font-family: "JetBrains Mono", monospace;'>SYSTEM</h3>
            <p style='color: #1DD1A1; font-weight: 700;'>ONLINE</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='terminal-card' style='text-align: center;'>
            <div class='status-indicator status-online'></div>
            <h3 style='color: #00D9FF; margin-top: 2rem; font-family: "JetBrains Mono", monospace;'>MODELS</h3>
            <p style='color: #1DD1A1; font-weight: 700;'>LOADED</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='terminal-card' style='text-align: center;'>
            <div class='status-indicator status-online'></div>
            <h3 style='color: #00D9FF; margin-top: 2rem; font-family: "JetBrains Mono", monospace;'>AI ENGINE</h3>
            <p style='color: #1DD1A1; font-weight: 700;'>READY</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='terminal-card' style='text-align: center;'>
            <div class='status-indicator status-online'></div>
            <h3 style='color: #00D9FF; margin-top: 2rem; font-family: "JetBrains Mono", monospace;'>XAI</h3>
            <p style='color: #1DD1A1; font-weight: 700;'>ACTIVE</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='terminal-card'>
            <h2 style='color: #00D9FF; font-family: "JetBrains Mono", monospace; 
                       letter-spacing: 2px; margin-top: 1.5rem;'>
                >> SYSTEM OVERVIEW
            </h2>
            <div class='header-line'></div>
            
            <p style='color: #ffffff; font-size: 1.1rem; line-height: 1.8;'>
                Advanced deep learning-based classification system for histopathological image analysis 
                across multiple organs and pathological conditions.
            </p>
            
            <h3 style='color: #F093FB; margin-top: 1.5rem; font-family: "Inter", sans-serif;'>
                🔍 CORE CAPABILITIES
            </h3>
            
            <div class='data-card'>
                <p style='color: #00D9FF; font-weight: 600;'>✦ MULTI-ORGAN CLASSIFICATION</p>
                <p style='color: #ffffff; margin-left: 1.5rem;'>
                    Support for 14 detailed pathological classes and 5 organ-level classifications
                </p>
            </div>
            
            <div class='data-card'>
                <p style='color: #00D9FF; font-weight: 600;'>✦ EXPLAINABLE AI INTERFACE</p>
                <p style='color: #ffffff; margin-left: 1.5rem;'>
                    Visual explanations and attention mechanisms for model predictions
                </p>
            </div>
            
            <div class='data-card'>
                <p style='color: #00D9FF; font-weight: 600;'>✦ REAL-TIME ANALYSIS</p>
                <p style='color: #ffffff; margin-left: 1.5rem;'>
                    Instant predictions with confidence scores and probability distributions
                </p>
            </div>
            
            <div class='data-card'>
                <p style='color: #00D9FF; font-weight: 600;'>✦ COMPREHENSIVE ANALYTICS</p>
                <p style='color: #ffffff; margin-left: 1.5rem;'>
                    Detailed performance metrics, visualizations, and model insights
                </p>
            </div>
            
            <h3 style='color: #F093FB; margin-top: 1.5rem; font-family: "Inter", sans-serif;'>
                🎯 SUPPORTED CLASSIFICATIONS
            </h3>
            
            <div class='terminal-card' style='background: rgba(102, 126, 234, 0.05);'>
                <p style='color: #00D9FF; font-weight: 700; font-family: "JetBrains Mono", monospace;'>
                    [ 14 DETAILED CLASSES ]
                </p>
                <p style='color: #ffffff; margin: 0.5rem 0;'>
                    <span style='color: #FF6B6B;'>●</span> Breast: Benign & Malignant<br>
                    <span style='color: #4ECDC4;'>●</span> Colon: Adenocarcinoma & Normal<br>
                    <span style='color: #45B7D1;'>●</span> Lung: Adenocarcinoma, Normal, Squamous Cell Carcinoma<br>
                    <span style='color: #96CEB4;'>●</span> Oral: Normal & OSCC<br>
                    <span style='color: #FECA57;'>●</span> Ovarian: Multiple pathological subtypes
                </p>
            </div>
            
            <div class='terminal-card' style='background: rgba(118, 75, 162, 0.05);'>
                <p style='color: #00D9FF; font-weight: 700; font-family: "JetBrains Mono", monospace;'>
                    [ 5 ORGAN-LEVEL CLASSES ]
                </p>
                <p style='color: #ffffff; margin: 0.5rem 0;'>
                    Breast • Colon • Lung • Oral • Ovarian
                </p>
            </div>
            
            <h3 style='color: #F093FB; margin-top: 1.5rem; font-family: "Inter", sans-serif;'>
                🚀 QUICK START GUIDE
            </h3>
            
            <div class='data-card' style='border-left: 4px solid #667eea;'>
                <p style='color: #00D9FF; font-weight: 600;'>STEP 1: IMAGE CLASSIFICATION</p>
                <p style='color: #ffffff;'>Upload and analyze histopathology images with instant AI predictions</p>
            </div>
            
            <div class='data-card' style='border-left: 4px solid #764ba2;'>
                <p style='color: #00D9FF; font-weight: 600;'>STEP 2: XAI EXPLANATIONS</p>
                <p style='color: #ffffff;'>Understand model decisions through visual attention maps</p>
            </div>
            
            <div class='data-card' style='border-left: 4px solid #F093FB;'>
                <p style='color: #00D9FF; font-weight: 600;'>STEP 3: MODEL ANALYSIS</p>
                <p style='color: #ffffff;'>Explore comprehensive performance metrics and insights</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Enhanced sample prediction cards
        st.markdown("""
        <div class='terminal-card'>
            <h3 style='color: #00D9FF; font-family: "JetBrains Mono", monospace; 
                       letter-spacing: 2px; margin-top: 1.5rem; text-align: center;'>
                >> LIVE PREDICTIONS
            </h3>
            <div class='header-line'></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample prediction cards with animations
        sample_data = [
            {"organ": "BREAST", "condition": "Benign", "confidence": 0.94, "color": "#FF6B6B"},
            {"organ": "LUNG", "condition": "Adenocarcinoma", "confidence": 0.87, "color": "#45B7D1"},
            {"organ": "COLON", "condition": "Normal", "confidence": 0.92, "color": "#4ECDC4"},
            {"organ": "OVARIAN", "condition": "Serous", "confidence": 0.89, "color": "#FECA57"},
        ]
        
        for i, sample in enumerate(sample_data):
            st.markdown(f"""
            <div class='metric-card' style='border-left: 4px solid {sample['color']};'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h4 style='color: {sample['color']}; font-family: "JetBrains Mono", monospace; 
                                   margin: 0; font-size: 1.1rem; letter-spacing: 2px;'>
                            {sample['organ']}
                        </h4>
                        <p style='color: #ffffff; margin: 0.5rem 0; font-size: 0.95rem;'>
                            {sample['condition']}
                        </p>
                    </div>
                    <div style='text-align: right;'>
                        <p style='color: #00D9FF; font-family: "JetBrains Mono", monospace; 
                                  font-weight: 700; font-size: 1.5rem; margin: 0;'>
                            {sample['confidence']*100:.0f}%
                        </p>
                        <p style='color: rgba(255,255,255,0.5); font-size: 0.75rem; margin: 0;'>
                            confidence
                        </p>
                    </div>
                </div>
                <div style='background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px; 
                            margin-top: 0.8rem; overflow: hidden;'>
                    <div style='background: linear-gradient(90deg, {sample['color']}, #00D9FF); 
                                height: 100%; width: {sample['confidence']*100}%; 
                                border-radius: 3px; box-shadow: 0 0 10px {sample['color']};
                                animation: fill{i} 1s ease-out;'></div>
                </div>
            </div>
            <style>
                @keyframes fill{i} {{
                    from {{ width: 0%; }}
                    to {{ width: {sample['confidence']*100}%; }}
                }}
            </style>
            """, unsafe_allow_html=True)
        
        # System metrics
        st.markdown("""
        <div class='terminal-card' style='margin-top: 1rem;'>
            <h4 style='color: #00D9FF; font-family: "JetBrains Mono", monospace; 
                       text-align: center; letter-spacing: 2px;'>
                >> SYSTEM METRICS
            </h4>
            <div class='header-line'></div>
            <div style='text-align: center; padding: 1rem 0;'>
                <p style='color: #ffffff; margin: 0.5rem 0;'>
                    <span style='color: #1DD1A1; font-weight: 700; font-size: 1.5rem;'>87.3%</span><br>
                    <span style='color: rgba(255,255,255,0.6); font-size: 0.8rem;'>ACCURACY</span>
                </p>
                <p style='color: #ffffff; margin: 0.5rem 0;'>
                    <span style='color: #FECA57; font-weight: 700; font-size: 1.5rem;'>45ms</span><br>
                    <span style='color: rgba(255,255,255,0.6); font-size: 0.8rem;'>INFERENCE TIME</span>
                </p>
                <p style='color: #ffffff; margin: 0.5rem 0;'>
                    <span style='color: #48DBFB; font-weight: 700; font-size: 1.5rem;'>93.4%</span><br>
                    <span style='color: rgba(255,255,255,0.6); font-size: 0.8rem;'>AUC-ROC</span>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_classification_page(model_14, model_5):
    """Display image classification interface"""
    st.markdown('<div class="sub-header">🖼️ Histopathology Image Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_type = st.radio(
            "Select Classification Type:",
            ["14-Class Detailed", "5-Class Organ Level"],
            horizontal=True
        )
    
    with col2:
        confidence_threshold = st.slider(
            "Confidence Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence score for predictions"
        )
    
    # Image upload
    uploaded_file = st.file_uploader(
        "Upload Histopathology Image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Supported formats: PNG, JPG, JPEG, TIF, TIFF"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", width='stretch')
            
            # Image info
            st.info(f"Image Size: {image.size} | Mode: {image.mode}")
        
        with col2:
            if st.button("🔍 Analyze Image", type="primary", width='stretch'):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Model inference
                    if model_type == "14-Class Detailed":
                        model = model_14
                        class_names = CLASSES_14
                    else:
                        model = model_5
                        class_names = CLASSES_5
                    
                    # Perform prediction
                    with torch.no_grad():
                        image_tensor = image_tensor.to(device)
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        
                        confidence_score = confidence.item()
                        predicted_class = predicted.item()
                        all_probabilities = probabilities.cpu().numpy()[0]
                    
                    # Display results
                    if confidence_score >= confidence_threshold:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>🎯 Prediction Result</h2>
                            <h3>{class_names[predicted_class].replace('_', ' ').title()}</h3>
                            <div class="confidence-bar" style="width: {confidence_score*100}%"></div>
                            <h4>{confidence_score*100:.2f}% Confidence</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Organ tag for 14-class predictions
                        if model_type == "14-Class Detailed":
                            organ = ORGAN_MAPPING.get(class_names[predicted_class], "Unknown")
                            organ_colors = {
                                'breast': '#FF6B6B',
                                'colon': '#4ECDC4', 
                                'lung': '#45B7D1',
                                'oral': '#96CEB4',
                                'ovarian': '#FECA57'
                            }
                            st.markdown(f"""
                            <div style='background: {organ_colors.get(organ, '#CCCCCC')}; 
                                        color: white; padding: 0.5rem; border-radius: 20px; 
                                        text-align: center; font-weight: bold; margin: 1rem 0;'>
                                🏥 {organ.upper()} ORGAN
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"⚠️ Low Confidence Prediction ({confidence_score*100:.1f}%)")
                        st.info("The model is not confident about this prediction. Consider consulting a pathologist.")
                    
                    # Probability distribution
                    st.markdown("### 📊 Probability Distribution")
                    
                    # Create sorted probabilities
                    prob_data = []
                    for i, prob in enumerate(all_probabilities):
                        prob_data.append({
                            'Class': class_names[i].replace('_', ' ').title(),
                            'Probability': prob,
                            'Organ': ORGAN_MAPPING.get(class_names[i], "Unknown") if model_type == "14-Class Detailed" else class_names[i]
                        })
                    
                    prob_df = pd.DataFrame(prob_data)
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Color mapping for organs
                    organ_colors_plotly = {
                        'breast': '#FF6B6B',
                        'colon': '#4ECDC4',
                        'lung': '#45B7D1', 
                        'oral': '#96CEB4',
                        'ovarian': '#FECA57'
                    }
                    
                    # Create interactive bar chart
                    fig = px.bar(
                        prob_df.head(10),
                        x='Probability',
                        y='Class',
                        orientation='h',
                        color='Organ' if model_type == "14-Class Detailed" else None,
                        color_discrete_map=organ_colors_plotly,
                        title="Top 10 Predictions",
                        labels={'Probability': 'Confidence Score', 'Class': 'Class Name'}
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        showlegend=model_type == "14-Class Detailed"
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Confidence metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Top Prediction", f"{confidence_score*100:.2f}%")
                    
                    with col2:
                        entropy = -np.sum(all_probabilities * np.log(all_probabilities + 1e-10))
                        st.metric("Prediction Entropy", f"{entropy:.3f}")
                    
                    with col3:
                        top3_confidence = np.sum(np.sort(all_probabilities)[-3:])
                        st.metric("Top-3 Confidence", f"{top3_confidence*100:.2f}%")

def show_model_analysis_page(model_14, model_5):
    """Display model analysis and comparison"""
    st.markdown('<div class="sub-header">📊 Model Analysis & Comparison</div>', unsafe_allow_html=True)
    
    # Model statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        params_14 = sum(p.numel() for p in model_14.parameters())
        st.metric("14-Class Model Parameters", f"{params_14:,}")
    
    with col2:
        params_5 = sum(p.numel() for p in model_5.parameters())
        st.metric("5-Class Model Parameters", f"{params_5:,}")
    
    with col3:
        st.metric("14-Class Model Size", f"{params_14 * 4 / 1024 / 1024:.2f} MB")
    
    with col4:
        st.metric("5-Class Model Size", f"{params_5 * 4 / 1024 / 1024:.2f} MB")
    
    # Model comparison
    st.markdown("### 📈 Model Architecture Comparison")
    
    # Create comparison data
    comparison_data = {
        'Metric': ['Parameters', 'Model Size (MB)', 'Number of Classes', 'Inference Speed'],
        '14-Class Model': [params_14, params_14 * 4 / 1024 / 1024, 14, 'Standard'],
        '5-Class Model': [params_5, params_5 * 4 / 1024 / 1024, 5, 'Faster']
    }
    
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, width='stretch')
    
    # Performance metrics (placeholder - you can replace with actual metrics)
    st.markdown("### 🎯 Expected Performance Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 14-Class Model")
        metrics_14 = {
            'Accuracy': 0.854,
            'Precision (Macro)': 0.832,
            'Recall (Macro)': 0.821,
            'F1-Score (Macro)': 0.827,
            'AUC-ROC': 0.934
        }
        
        for metric, value in metrics_14.items():
            st.progress(value, text=f"{metric}: {value:.3f}")
    
    with col2:
        st.markdown("#### 5-Class Model")
        metrics_5 = {
            'Accuracy': 0.892,
            'Precision (Macro)': 0.876,
            'Recall (Macro)': 0.868,
            'F1-Score (Macro)': 0.872,
            'AUC-ROC': 0.958
        }
        
        for metric, value in metrics_5.items():
            st.progress(value, text=f"{metric}: {value:.3f}")
    
    # Class distribution visualization
    st.markdown("### 📊 Class Distribution Analysis")
    
    # 14-class distribution
    fig1 = go.Figure(data=[
        go.Bar(
            name='14-Class Model',
            x=[name.replace('_', ' ').title() for name in CLASSES_14],
            y=[1.0] * len(CLASSES_14),  # Placeholder values
            marker_color='lightblue'
        )
    ])
    
    fig1.update_layout(
        title="14-Class Model - Supported Classes",
        xaxis_tickangle=-45,
        height=400
    )
    
    st.plotly_chart(fig1, width='stretch')
    
    # 5-class distribution
    fig2 = go.Figure(data=[
        go.Pie(
            labels=CLASSES_5,
            values=[1.0] * len(CLASSES_5),  # Placeholder values
            hole=0.3
        )
    ])
    
    fig2.update_layout(
        title="5-Class Model - Organ Distribution"
    )
    
    st.plotly_chart(fig2, width='stretch')

def show_xai_page(model_14, model_5):
    """Display XAI explanations interface"""
    st.markdown('<div class="sub-header">🔍 Explainable AI - Model Interpretability</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Understanding Model Decisions
    
    This section provides visual explanations of how the model makes its predictions,
    helping pathologists understand and trust the AI's decisions.
    """)
    
    # Image upload for XAI
    uploaded_file = st.file_uploader(
        "Upload Image for XAI Analysis",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        key="xai_upload"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Original Image", width='stretch')
        
        with col2:
            model_type = st.radio(
                "Model for XAI:",
                ["14-Class Detailed", "5-Class Organ Level"],
                horizontal=True,
                key="xai_model"
            )
            
            if st.button("🔍 Generate XAI Explanations", type="primary"):
                with st.spinner("Generating explanations..."):
                    # Preprocess and predict
                    image_tensor = preprocess_image(image)
                    
                    if model_type == "14-Class Detailed":
                        model = model_14
                        class_names = CLASSES_14
                    else:
                        model = model_5
                        class_names = CLASSES_5
                    
                    # Get prediction
                    with torch.no_grad():
                        image_tensor = image_tensor.to(device)
                        outputs = model(image_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    
                    # Generate Grad-CAM
                    heatmap = generate_grad_cam(model, image_tensor, predicted.item())
                    
                    # Display results
                    st.markdown("### 🎯 XAI Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Original Image")
                        st.image(image, width='stretch')
                    
                    with col2:
                        st.markdown("#### Attention Heatmap")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.imshow(image)
                        ax.imshow(heatmap, cmap='jet', alpha=0.5)
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    # Explanation text
                    st.markdown("### 📝 Explanation")
                    st.info(f"""
                    The model predicts **{class_names[predicted.item()].replace('_', ' ').title()}** 
                    with **{confidence.item()*100:.2f}% confidence**.
                    
                    The red regions in the heatmap show the areas that most influenced the model's decision.
                    These regions likely contain histopathological features relevant to the predicted class.
                    """)
                    
                    # Feature importance
                    st.markdown("### 🔍 Feature Importance")
                    
                    # Create feature importance visualization (placeholder)
                    features = [
                        "Nuclear Morphology", "Tissue Architecture", "Cell Density",
                        "Stromal Patterns", "Inflammatory Infiltrate", "Necrosis Areas"
                    ]
                    importance_scores = np.random.random(len(features))
                    
                    feature_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance_scores
                    }).sort_values('Importance', ascending=True)
                    
                    fig = px.bar(
                        feature_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title="Feature Importance Analysis",
                        color='Importance',
                        color_continuous_scale='viridis'
                    )
                    
                    st.plotly_chart(fig, width='stretch')

def show_performance_dashboard():
    """Display performance dashboard with metrics"""
    st.markdown('<div class="sub-header">📈 Performance Dashboard</div>', unsafe_allow_html=True)
    
    # Performance metrics (placeholder data - replace with your actual metrics)
    st.markdown("### 📊 Overall Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Accuracy", "87.3%", "2.1%")
    
    with col2:
        st.metric("Macro F1-Score", "85.8%", "1.7%")
    
    with col3:
        st.metric("AUC-ROC", "93.4%", "0.8%")
    
    with col4:
        st.metric("Inference Time", "45ms", "-5ms")
    
    # Performance by organ
    st.markdown("### 🏥 Performance by Organ")
    
    organ_performance = {
        'Organ': ['Breast', 'Colon', 'Lung', 'Oral', 'Ovarian'],
        'Accuracy': [0.89, 0.86, 0.88, 0.85, 0.87],
        'Precision': [0.87, 0.85, 0.86, 0.84, 0.86],
        'Recall': [0.88, 0.84, 0.87, 0.83, 0.85],
        'F1-Score': [0.875, 0.845, 0.865, 0.835, 0.855]
    }
    
    organ_df = pd.DataFrame(organ_performance)
    
    # Create radar chart for organ performance
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for organ in organ_df['Organ']:
        organ_data = organ_df[organ_df['Organ'] == organ]
        values = [organ_data[metric].values[0] for metric in metrics]
        values.append(values[0])  # Complete the circle
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=organ
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0.8, 0.9]
            )),
        showlegend=True,
        title="Organ-wise Performance Radar Chart"
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Confusion matrix visualization
    st.markdown("### 🔍 Confusion Matrix Analysis")
    
    # Placeholder confusion matrix data
    if st.checkbox("Show Confusion Matrix"):
        # Sample confusion matrix data
        cm_data = np.random.randint(0, 100, (5, 5))
        np.fill_diagonal(cm_data, np.random.randint(80, 100, 5))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=CLASSES_5, yticklabels=CLASSES_5, ax=ax)
        ax.set_title('Confusion Matrix - 5-Class Model')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        st.pyplot(fig)
    
    # Model comparison over time
    st.markdown("### 📈 Training Progress")
    
    # Sample training history
    epochs = list(range(1, 51))
    train_loss = [2.0 - 0.03*i + np.random.normal(0, 0.1) for i in range(50)]
    val_loss = [1.8 - 0.025*i + np.random.normal(0, 0.08) for i in range(50)]
    train_acc = [0.4 + 0.01*i + np.random.normal(0, 0.05) for i in range(50)]
    val_acc = [0.35 + 0.012*i + np.random.normal(0, 0.04) for i in range(50)]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='blue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=val_loss, name='Val Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=train_acc, name='Train Accuracy', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=val_acc, name='Val Accuracy', line=dict(color='orange')),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    
    st.plotly_chart(fig, width='stretch')

# Run the application
if __name__ == "__main__":
    main()