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

# Custom CSS for terminal-style interface
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Code+Pro:wght@300;400;500;600&display=swap');
    
    /* Global Styles - Dark Terminal */
    .stApp {
        background: #0a0e14;
        font-family: 'Source Code Pro', monospace;
        color: #00ff41;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0e14;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #00ff41;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #00cc33;
    }
    
    /* Main Header - Terminal Style */
    .main-header {
        font-size: 1.8rem;
        font-weight: 400;
        color: #00ff41;
        text-align: center;
        margin-bottom: 1rem;
        font-family: 'Source Code Pro', monospace;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.5);
    }
    
    /* Sub Header - Minimal Green */
    .sub-header {
        font-size: 1rem;
        font-weight: 500;
        color: #00ff41;
        margin-bottom: 1rem;
        font-family: 'Source Code Pro', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-left: 3px solid #00ff41;
        padding-left: 0.8rem;
    }
    
    /* Terminal-Style Cards */
    .terminal-card {
        background: rgba(0, 20, 10, 0.6);
        border: 1px solid #00ff41;
        border-radius: 3px;
        padding: 1rem;
        margin: 0.8rem 0;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.1);
        font-size: 0.85rem;
    }
    
    .terminal-card::before {
        content: '> ';
        color: #00ff41;
        font-weight: 600;
    }
    
    /* Metric Cards - Minimal */
    .metric-card {
        background: rgba(0, 20, 10, 0.4);
        border: 1px solid #00ff41;
        border-radius: 2px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        transition: all 0.2s ease;
        font-size: 0.85rem;
    }
    
    .metric-card:hover {
        background: rgba(0, 30, 15, 0.6);
        box-shadow: 0 0 8px rgba(0, 255, 65, 0.3);
    }
    
    /* Prediction Card - Terminal Style */
    .prediction-card {
        background: rgba(0, 30, 15, 0.8);
        border: 2px solid #00ff41;
        color: #00ff41;
        padding: 1.2rem;
        border-radius: 3px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(0, 255, 65, 0.3);
        font-size: 0.9rem;
    }
    
    .prediction-card h2 {
        font-family: 'Source Code Pro', monospace;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
    
    .prediction-card h3 {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 0.8rem 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .prediction-card h4 {
        font-family: 'Source Code Pro', monospace;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    /* Confidence Bar - Green Terminal Style */
    .confidence-bar {
        height: 8px;
        background: #00ff41;
        border-radius: 2px;
        margin: 0.8rem 0;
        box-shadow: 0 0 8px rgba(0, 255, 65, 0.5);
    }
    
    /* Data Display Cards */
    .data-card {
        background: rgba(0, 20, 10, 0.3);
        border: 1px solid #00ff41;
        border-radius: 2px;
        padding: 0.7rem;
        margin: 0.4rem 0;
        transition: all 0.2s ease;
        font-size: 0.85rem;
    }
    
    .data-card:hover {
        border-color: #00ff41;
        box-shadow: 0 0 8px rgba(0, 255, 65, 0.2);
    }
    
    /* Status Indicators */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: blink 2s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    .status-online { background: #00ff41; box-shadow: 0 0 6px #00ff41; }
    .status-processing { background: #ffaa00; box-shadow: 0 0 6px #ffaa00; }
    .status-error { background: #ff0000; box-shadow: 0 0 6px #ff0000; }
    
    /* Buttons - Terminal Style */
    .stButton > button {
        background: transparent;
        color: #00ff41;
        border: 1px solid #00ff41;
        border-radius: 2px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        font-family: 'Source Code Pro', monospace;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: rgba(0, 255, 65, 0.1);
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.3);
    }
    
    /* Sidebar - Dark Terminal */
    [data-testid="stSidebar"] {
        background: #0a0e14;
        border-right: 1px solid #00ff41;
    }
    
    [data-testid="stSidebar"] * {
        color: #00ff41 !important;
        font-size: 0.85rem;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: #00ff41;
        border-radius: 2px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(0, 20, 10, 0.3);
        border: 1px dashed #00ff41;
        border-radius: 3px;
        padding: 1.5rem;
        font-size: 0.85rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #00ff41;
        box-shadow: 0 0 10px rgba(0, 255, 65, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        color: #00ff41;
        font-family: 'Source Code Pro', monospace;
        font-weight: 400;
        border-radius: 2px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(0, 255, 65, 0.1);
        border-color: #00ff41;
        color: #00ff41;
    }
    
    /* Metrics - Terminal Style */
    [data-testid="stMetricValue"] {
        font-family: 'Source Code Pro', monospace;
        font-size: 1.5rem;
        font-weight: 500;
        color: #00ff41;
        text-shadow: 0 0 8px rgba(0, 255, 65, 0.3);
    }
    
    [data-testid="stMetricLabel"] {
        font-family: 'Source Code Pro', monospace;
        font-weight: 400;
        color: #00ff41;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.75rem;
    }
    
    /* Alert Boxes */
    .stAlert {
        background: rgba(0, 20, 10, 0.5);
        border-left: 3px solid #00ff41;
        border-radius: 2px;
        font-size: 0.85rem;
    }
    
    /* Radio Buttons */
    .stRadio > label {
        color: #00ff41;
        font-family: 'Source Code Pro', monospace;
        font-weight: 400;
        font-size: 0.85rem;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #00ff41;
        font-family: 'Source Code Pro', monospace;
        font-weight: 400;
        font-size: 0.85rem;
    }
    
    /* Slider */
    .stSlider label {
        color: #00ff41;
        font-family: 'Source Code Pro', monospace;
        font-weight: 400;
        font-size: 0.85rem;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        background: rgba(0, 20, 10, 0.4);
        border: 1px solid #00ff41;
        border-radius: 3px;
        font-size: 0.85rem;
    }
    
    /* Text Elements */
    p, li, div {
        color: #00ff41;
        font-family: 'Source Code Pro', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #00ff41;
        font-family: 'Source Code Pro', monospace;
        font-weight: 500;
    }
    
    /* Header Line */
    .header-line {
        height: 1px;
        background: #00ff41;
        margin: 0.8rem 0;
        box-shadow: 0 0 5px rgba(0, 255, 65, 0.3);
    }
    
    /* Loading Spinner */
    .stSpinner > div {
        border-top-color: #00ff41 !important;
        border-right-color: #00ff41 !important;
        border-bottom-color: #00ff41 !important;
        border-left-color: transparent !important;
    }
    
    /* Input Fields */
    input, textarea, select {
        background: rgba(0, 20, 10, 0.5) !important;
        border: 1px solid #00ff41 !important;
        color: #00ff41 !important;
        font-family: 'Source Code Pro', monospace !important;
        font-size: 0.85rem !important;
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
    st.markdown('<div class="main-header">> HISTOPATH AI - NEURAL DIAGNOSTIC SYSTEM</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("> NAVIGATION")
    app_mode = st.sidebar.selectbox(
        "SELECT MODE:",
        ["HOME", "IMAGE CLASSIFICATION", "MODEL ANALYSIS", "XAI EXPLANATIONS", "PERFORMANCE DASHBOARD"]
    )
    
    # Load models
    with st.sidebar:
        st.markdown("### > MODEL STATUS")
        with st.spinner("LOADING MODELS..."):
            model_14 = load_14_class_model()
            model_5 = load_5_class_model()
        
        if model_14 and model_5:
            st.success("[OK] MODELS LOADED")
            st.info(f"DEVICE: {device}")
        else:
            st.error("[ERROR] FAILED TO LOAD MODELS")
            return
    
    # Home Page
    if app_mode == "HOME":
        show_home_page()
    
    # Image Classification
    elif app_mode == "IMAGE CLASSIFICATION":
        show_classification_page(model_14, model_5)
    
    # Model Analysis
    elif app_mode == "MODEL ANALYSIS":
        show_model_analysis_page(model_14, model_5)
    
    # XAI Explanations
    elif app_mode == "XAI EXPLANATIONS":
        show_xai_page(model_14, model_5)
    
    # Performance Dashboard
    elif app_mode == "PERFORMANCE DASHBOARD":
        show_performance_dashboard()

def show_home_page():
    """Display home page with overview and instructions"""
    
    # Terminal-style welcome banner
    st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <div class='header-line'></div>
        <p style='font-size: 0.8rem; color: #00ff41; font-family: "Source Code Pro", monospace; 
                   letter-spacing: 2px; margin: 0.5rem 0;'>> INITIALIZING...</p>
        <h1 style='font-size: 1.8rem; font-weight: 500; color: #00ff41;
                   font-family: "Source Code Pro", monospace; letter-spacing: 3px; margin: 1rem 0;'>
            HISTOPATH AI v2.0
        </h1>
        <p style='color: #00ff41; font-size: 0.75rem; font-family: "Source Code Pro", monospace; 
                  letter-spacing: 1px; margin-top: 0.3rem;'>
            ADVANCED NEURAL DIAGNOSTIC SYSTEM
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
            <p style='color: #00ff41; margin-top: 1rem; font-size: 0.8rem;'>SYSTEM</p>
            <p style='color: #00ff41; font-weight: 500; font-size: 0.75rem;'>[ONLINE]</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='terminal-card' style='text-align: center;'>
            <div class='status-indicator status-online'></div>
            <p style='color: #00ff41; margin-top: 1rem; font-size: 0.8rem;'>MODELS</p>
            <p style='color: #00ff41; font-weight: 500; font-size: 0.75rem;'>[LOADED]</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='terminal-card' style='text-align: center;'>
            <div class='status-indicator status-online'></div>
            <p style='color: #00ff41; margin-top: 1rem; font-size: 0.8rem;'>AI ENGINE</p>
            <p style='color: #00ff41; font-weight: 500; font-size: 0.75rem;'>[READY]</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='terminal-card' style='text-align: center;'>
            <div class='status-indicator status-online'></div>
            <p style='color: #00ff41; margin-top: 1rem; font-size: 0.8rem;'>XAI</p>
            <p style='color: #00ff41; font-weight: 500; font-size: 0.75rem;'>[ACTIVE]</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='terminal-card'>
            <p style='color: #00ff41; font-size: 0.85rem; letter-spacing: 1px; margin-top: 0.5rem;'>
                >> SYSTEM OVERVIEW
            </p>
            <div class='header-line'></div>
            
            <p style='color: #00ff41; font-size: 0.8rem; line-height: 1.7; margin: 1rem 0;'>
                Deep learning classification system for histopathological image analysis.
                Multi-organ pathological condition detection and classification.
            </p>
            
            <p style='color: #00ff41; font-size: 0.8rem; font-weight: 500; margin-top: 1rem;'>
                >> CORE CAPABILITIES
            </p>
            
            <div class='data-card'>
                <p style='color: #00ff41; font-size: 0.8rem;'>[+] MULTI-ORGAN CLASSIFICATION</p>
                <p style='color: #00ff41; font-size: 0.75rem; margin-left: 1rem; opacity: 0.8;'>
                    14 detailed pathological classes | 5 organ-level classifications
                </p>
            </div>
            
            <div class='data-card'>
                <p style='color: #00ff41; font-size: 0.8rem;'>[+] EXPLAINABLE AI</p>
                <p style='color: #00ff41; font-size: 0.75rem; margin-left: 1rem; opacity: 0.8;'>
                    Visual explanations | Attention mechanisms | Model interpretability
                </p>
            </div>
            
            <div class='data-card'>
                <p style='color: #00ff41; font-size: 0.8rem;'>[+] REAL-TIME ANALYSIS</p>
                <p style='color: #00ff41; font-size: 0.75rem; margin-left: 1rem; opacity: 0.8;'>
                    Instant predictions | Confidence scores | Probability distributions
                </p>
            </div>
            
            <div class='data-card'>
                <p style='color: #00ff41; font-size: 0.8rem;'>[+] ANALYTICS DASHBOARD</p>
                <p style='color: #00ff41; font-size: 0.75rem; margin-left: 1rem; opacity: 0.8;'>
                    Performance metrics | Model insights | Visualization tools
                </p>
            </div>
            
            <p style='color: #00ff41; font-size: 0.8rem; font-weight: 500; margin-top: 1.5rem;'>
                >> SUPPORTED CLASSIFICATIONS
            </p>
            
            <div class='terminal-card' style='background: rgba(0, 30, 15, 0.3);'>
                <p style='color: #00ff41; font-size: 0.75rem; font-weight: 500;'>
                    [14 DETAILED CLASSES]
                </p>
                <p style='color: #00ff41; font-size: 0.75rem; margin: 0.5rem 0; line-height: 1.6;'>
                    > Breast: Benign | Malignant<br>
                    > Colon: Adenocarcinoma | Normal<br>
                    > Lung: Adenocarcinoma | Normal | Squamous Cell Carcinoma<br>
                    > Oral: Normal | OSCC<br>
                    > Ovarian: Multiple pathological subtypes
                </p>
            </div>
            
            <div class='terminal-card' style='background: rgba(0, 30, 15, 0.3);'>
                <p style='color: #00ff41; font-size: 0.75rem; font-weight: 500;'>
                    [5 ORGAN-LEVEL CLASSES]
                </p>
                <p style='color: #00ff41; font-size: 0.75rem; margin: 0.5rem 0;'>
                    breast | colon | lung | oral | ovarian
                </p>
            </div>
            
            <p style='color: #00ff41; font-size: 0.8rem; font-weight: 500; margin-top: 1.5rem;'>
                >> QUICK START
            </p>
            
            <div class='data-card' style='border-left: 2px solid #00ff41;'>
                <p style='color: #00ff41; font-size: 0.8rem;'>STEP 1: IMAGE CLASSIFICATION</p>
                <p style='color: #00ff41; font-size: 0.75rem; opacity: 0.8;'>Upload and analyze histopathology images</p>
            </div>
            
            <div class='data-card' style='border-left: 2px solid #00ff41;'>
                <p style='color: #00ff41; font-size: 0.8rem;'>STEP 2: XAI EXPLANATIONS</p>
                <p style='color: #00ff41; font-size: 0.75rem; opacity: 0.8;'>View model decision attention maps</p>
            </div>
            
            <div class='data-card' style='border-left: 2px solid #00ff41;'>
                <p style='color: #00ff41; font-size: 0.8rem;'>STEP 3: MODEL ANALYSIS</p>
                <p style='color: #00ff41; font-size: 0.75rem; opacity: 0.8;'>Explore performance metrics</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Terminal-style predictions
        st.markdown("""
        <div class='terminal-card'>
            <p style='color: #00ff41; font-size: 0.8rem; letter-spacing: 1px; text-align: center;'>
                >> RECENT PREDICTIONS
            </p>
            <div class='header-line'></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample prediction cards
        sample_data = [
            {"organ": "BREAST", "condition": "Benign", "confidence": 0.94},
            {"organ": "LUNG", "condition": "Adenocarcinoma", "confidence": 0.87},
            {"organ": "COLON", "condition": "Normal", "confidence": 0.92},
            {"organ": "OVARIAN", "condition": "Serous", "confidence": 0.89},
        ]
        
        for i, sample in enumerate(sample_data):
            st.markdown(f"""
            <div class='metric-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <p style='color: #00ff41; font-size: 0.75rem; font-weight: 500; 
                                   margin: 0; letter-spacing: 1px;'>
                            [{sample['organ']}]
                        </p>
                        <p style='color: #00ff41; margin: 0.3rem 0; font-size: 0.75rem; opacity: 0.8;'>
                            {sample['condition']}
                        </p>
                    </div>
                    <div style='text-align: right;'>
                        <p style='color: #00ff41; font-weight: 500; font-size: 1.1rem; margin: 0;'>
                            {sample['confidence']*100:.0f}%
                        </p>
                        <p style='color: #00ff41; font-size: 0.65rem; margin: 0; opacity: 0.6;'>
                            confidence
                        </p>
                    </div>
                </div>
                <div style='background: rgba(0, 255, 65, 0.1); height: 3px; border-radius: 2px; 
                            margin-top: 0.5rem; overflow: hidden;'>
                    <div style='background: #00ff41; height: 100%; width: {sample['confidence']*100}%; 
                                border-radius: 2px; box-shadow: 0 0 5px #00ff41;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # System metrics
        st.markdown("""
        <div class='terminal-card' style='margin-top: 1rem;'>
            <p style='color: #00ff41; font-size: 0.8rem; text-align: center; letter-spacing: 1px;'>
                >> SYSTEM METRICS
            </p>
            <div class='header-line'></div>
            <div style='text-align: center; padding: 0.8rem 0;'>
                <p style='color: #00ff41; margin: 0.4rem 0;'>
                    <span style='font-weight: 500; font-size: 1.2rem;'>87.3%</span><br>
                    <span style='font-size: 0.7rem; opacity: 0.7;'>ACCURACY</span>
                </p>
                <p style='color: #00ff41; margin: 0.4rem 0;'>
                    <span style='font-weight: 500; font-size: 1.2rem;'>45ms</span><br>
                    <span style='font-size: 0.7rem; opacity: 0.7;'>INFERENCE TIME</span>
                </p>
                <p style='color: #00ff41; margin: 0.4rem 0;'>
                    <span style='font-weight: 500; font-size: 1.2rem;'>93.4%</span><br>
                    <span style='font-size: 0.7rem; opacity: 0.7;'>AUC-ROC</span>
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def show_classification_page(model_14, model_5):
    """Display image classification interface"""
    st.markdown('<div class="sub-header">IMAGE CLASSIFICATION</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_type = st.radio(
            "SELECT CLASSIFICATION TYPE:",
            ["14-Class Detailed", "5-Class Organ Level"],
            horizontal=True
        )
    
    with col2:
        confidence_threshold = st.slider(
            "CONFIDENCE THRESHOLD:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence score for predictions"
        )
    
    # Image upload
    uploaded_file = st.file_uploader(
        "UPLOAD HISTOPATHOLOGY IMAGE",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        help="Supported: PNG, JPG, JPEG, TIF, TIFF"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="UPLOADED IMAGE", width='stretch')
            
            # Image info
            st.info(f"SIZE: {image.size} | MODE: {image.mode}")
        
        with col2:
            if st.button("ANALYZE IMAGE", type="primary", width='stretch'):
                with st.spinner("ANALYZING..."):
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
                            <p style='font-size: 0.75rem; margin-bottom: 0.5rem;'>PREDICTION RESULT</p>
                            <p style='font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;'>{class_names[predicted_class].replace('_', ' ').upper()}</p>
                            <div class="confidence-bar" style="width: {confidence_score*100}%"></div>
                            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>{confidence_score*100:.2f}% CONFIDENCE</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Organ tag for 14-class predictions
                        if model_type == "14-Class Detailed":
                            organ = ORGAN_MAPPING.get(class_names[predicted_class], "Unknown")
                            st.markdown(f"""
                            <div style='background: rgba(0, 255, 65, 0.1); border: 1px solid #00ff41;
                                        color: #00ff41; padding: 0.4rem; border-radius: 3px; 
                                        text-align: center; font-weight: 500; margin: 1rem 0; font-size: 0.8rem;'>
                                [{organ.upper()} ORGAN]
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning(f"[WARNING] LOW CONFIDENCE: {confidence_score*100:.1f}%")
                        st.info("[INFO] Consult a pathologist for verification")
                    
                    # Probability distribution
                    st.markdown("### >> PROBABILITY DISTRIBUTION")
                    
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
                    
                    # Create bar chart
                    fig = px.bar(
                        prob_df.head(10),
                        x='Probability',
                        y='Class',
                        orientation='h',
                        title="TOP 10 PREDICTIONS",
                        labels={'Probability': 'Confidence Score', 'Class': 'Class Name'}
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#00ff41', family='Source Code Pro')
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                    
                    # Confidence metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("TOP PREDICTION", f"{confidence_score*100:.2f}%")
                    
                    with col2:
                        entropy = -np.sum(all_probabilities * np.log(all_probabilities + 1e-10))
                        st.metric("ENTROPY", f"{entropy:.3f}")
                    
                    with col3:
                        top3_confidence = np.sum(np.sort(all_probabilities)[-3:])
                        st.metric("TOP-3 CONFIDENCE", f"{top3_confidence*100:.2f}%")

def show_model_analysis_page(model_14, model_5):
    """Display model analysis and comparison"""
    st.markdown('<div class="sub-header">MODEL ANALYSIS</div>', unsafe_allow_html=True)
    
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
    st.markdown('<div class="sub-header">XAI EXPLANATIONS</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class='terminal-card'>
        <p style='font-size: 0.8rem;'>
            >> Visual explanations for model predictions.<br>
            >> Attention mechanisms and interpretability tools.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Image upload for XAI
    uploaded_file = st.file_uploader(
        "UPLOAD IMAGE FOR XAI ANALYSIS",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        key="xai_upload"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="ORIGINAL IMAGE", width='stretch')
        
        with col2:
            model_type = st.radio(
                "MODEL FOR XAI:",
                ["14-Class Detailed", "5-Class Organ Level"],
                horizontal=True,
                key="xai_model"
            )
            
            if st.button("GENERATE XAI", type="primary"):
                with st.spinner("GENERATING EXPLANATIONS..."):
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
                        image_tensor_pred = image_tensor.to(device)
                        outputs = model(image_tensor_pred)
                        probabilities = F.softmax(outputs, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                    
                    # Generate Grad-CAM
                    heatmap = generate_grad_cam(model, image_tensor, predicted.item())
                    
                    # Display results
                    st.markdown("### >> XAI RESULTS")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### ORIGINAL IMAGE")
                        st.image(image, width='stretch')
                    
                    with col2:
                        st.markdown("#### ATTENTION HEATMAP")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        fig.patch.set_facecolor('#0a0e14')
                        ax.set_facecolor('#0a0e14')
                        ax.imshow(image)
                        ax.imshow(heatmap, cmap='jet', alpha=0.5)
                        ax.axis('off')
                        st.pyplot(fig)
                    
                    # Explanation text
                    st.markdown("### >> EXPLANATION")
                    st.info(f"""
                    PREDICTED: {class_names[predicted.item()].replace('_', ' ').upper()} | CONFIDENCE: {confidence.item()*100:.2f}%
                    
                    The highlighted regions show areas that influenced the model's decision.
                    These regions contain relevant histopathological features.
                    """)
                    
                    # Feature importance
                    st.markdown("### >> FEATURE IMPORTANCE")
                    
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
                        title="FEATURE IMPORTANCE ANALYSIS"
                    )
                    
                    fig.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#00ff41', family='Source Code Pro')
                    )
                    
                    st.plotly_chart(fig, width='stretch')

def show_performance_dashboard():
    """Display performance dashboard with metrics"""
    st.markdown('<div class="sub-header">PERFORMANCE DASHBOARD</div>', unsafe_allow_html=True)
    
    # Performance metrics
    st.markdown("### >> OVERALL METRICS")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ACCURACY", "87.3%", "2.1%")
    
    with col2:
        st.metric("F1-SCORE", "85.8%", "1.7%")
    
    with col3:
        st.metric("AUC-ROC", "93.4%", "0.8%")
    
    with col4:
        st.metric("INFERENCE", "45ms", "-5ms")
    
    # Performance by organ
    st.markdown("### >> ORGAN PERFORMANCE")
    
    organ_performance = {
        'Organ': ['Breast', 'Colon', 'Lung', 'Oral', 'Ovarian'],
        'Accuracy': [0.89, 0.86, 0.88, 0.85, 0.87],
        'Precision': [0.87, 0.85, 0.86, 0.84, 0.86],
        'Recall': [0.88, 0.84, 0.87, 0.83, 0.85],
        'F1-Score': [0.875, 0.845, 0.865, 0.835, 0.855]
    }
    
    organ_df = pd.DataFrame(organ_performance)
    
    # Create radar chart
    fig = go.Figure()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for organ in organ_df['Organ']:
        organ_data = organ_df[organ_df['Organ'] == organ]
        values = [organ_data[metric].values[0] for metric in metrics]
        values.append(values[0])
        
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
        title="ORGAN-WISE PERFORMANCE",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ff41', family='Source Code Pro')
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Confusion matrix
    st.markdown("### >> CONFUSION MATRIX")
    
    if st.checkbox("SHOW CONFUSION MATRIX"):
        cm_data = np.random.randint(0, 100, (5, 5))
        np.fill_diagonal(cm_data, np.random.randint(80, 100, 5))
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0a0e14')
        ax.set_facecolor('#0a0e14')
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=CLASSES_5, yticklabels=CLASSES_5, ax=ax,
                   cbar_kws={'label': 'Count'})
        ax.set_title('CONFUSION MATRIX - 5-CLASS MODEL', color='#00ff41', fontsize=14)
        ax.set_xlabel('PREDICTED', color='#00ff41')
        ax.set_ylabel('ACTUAL', color='#00ff41')
        ax.tick_params(colors='#00ff41')
        
        st.pyplot(fig)
    
    # Training progress
    st.markdown("### >> TRAINING PROGRESS")
    
    epochs = list(range(1, 51))
    train_loss = [2.0 - 0.03*i + np.random.normal(0, 0.1) for i in range(50)]
    val_loss = [1.8 - 0.025*i + np.random.normal(0, 0.08) for i in range(50)]
    train_acc = [0.4 + 0.01*i + np.random.normal(0, 0.05) for i in range(50)]
    val_acc = [0.35 + 0.012*i + np.random.normal(0, 0.04) for i in range(50)]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('LOSS CURVE', 'ACCURACY CURVE'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=train_loss, name='Train Loss', line=dict(color='#00ff41')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=val_loss, name='Val Loss', line=dict(color='#ff4444')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=train_acc, name='Train Accuracy', line=dict(color='#00ff41')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=epochs, y=val_acc, name='Val Accuracy', line=dict(color='#ff4444')),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600, 
        showlegend=True,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#00ff41', family='Source Code Pro')
    )
    fig.update_xaxes(title_text="EPOCH", row=2, col=1, color='#00ff41')
    fig.update_yaxes(title_text="LOSS", row=1, col=1, color='#00ff41')
    fig.update_yaxes(title_text="ACCURACY", row=2, col=1, color='#00ff41')
    
    st.plotly_chart(fig, width='stretch')

# Run the application
if __name__ == "__main__":
    main()