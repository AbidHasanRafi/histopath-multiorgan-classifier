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
import json
from datetime import datetime
from typing import Dict, List, Any

# Add the model definitions
sys.path.append('.')

# Import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI not available. Install with: pip install google-generativeai")

# Import your model architecture - handle import errors gracefully
try:
    from model import UltraLightHistoNet, MicroAttention, DepthwiseSeparableConv, EfficientBlock
except ImportError as e:
    st.error(f"Failed to import model architecture: {e}")
    # Define placeholder classes to prevent crashes
    class UltraLightHistoNet(nn.Module):
        def __init__(self, num_classes=14, base_channels=16):
            super().__init__()
            self.num_classes = num_classes
            self.dummy = nn.Linear(10, num_classes)
        def forward(self, x):
            return self.dummy(x.view(x.size(0), -1))
    
    class MicroAttention(nn.Module):
        def __init__(self):
            super().__init__()
    
    class DepthwiseSeparableConv(nn.Module):
        def __init__(self):
            super().__init__()
    
    class EfficientBlock(nn.Module):
        def __init__(self):
            super().__init__()

# Set page configuration
st.set_page_config(
    page_title="HistoPath AI - Multi-Organ Classification",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal CSS - Clean and Simple
st.markdown("""
<style>
    .stApp {
        background: #ffffff;
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #1a202c;
        text-align: left;
        margin-bottom: 1.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #2d3748;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
    }
    
    .info-card {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    
    .prediction-box {
        background: #edf2f7;
        border-left: 4px solid #4299e1;
        padding: 1.2rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    
    .confidence-bar {
        height: 6px;
        background: #4299e1;
        border-radius: 3px;
        margin: 0.5rem 0;
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
        # Try different possible model file locations
        model_paths = [
            'best_model_14_class.pth',
            './models/best_model_14_class.pth',
            './weights/best_model_14_class.pth'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model_loaded = True
                    break
            except:
                continue
                
        if not model_loaded:
            st.warning("14-class model file not found. Using dummy model for demonstration.")
    except Exception as e:
        st.warning(f"Failed to load 14-class model: {e}. Using dummy model.")
    
    model.eval()
    return model

@st.cache_resource
def load_5_class_model():
    """Load the 5-class model"""
    model = UltraLightHistoNet(num_classes=5, base_channels=16)
    try:
        # Try different possible model file locations
        model_paths = [
            'best_model_5_class.pth',
            './models/best_model_5_class.pth',
            './weights/best_model_5_class.pth'
        ]
        
        model_loaded = False
        for model_path in model_paths:
            try:
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, map_location=device))
                    model_loaded = True
                    break
            except:
                continue
                
        if not model_loaded:
            st.warning("5-class model file not found. Using dummy model for demonstration.")
    except Exception as e:
        st.warning(f"Failed to load 5-class model: {e}. Using dummy model.")
    
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

# Multi-Agent Diagnostic System
class HistopathologyMultiAgent:
    def __init__(self, model_14, model_5, api_key=None):
        self.model_14 = model_14
        self.model_5 = model_5
        
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self.gemini = genai.GenerativeModel('gemini-2.5-flash')
                self.enabled = True
            except Exception as e:
                st.error(f"Failed to configure Gemini: {e}")
                self.gemini = None
                self.enabled = False
        else:
            self.gemini = None
            self.enabled = False
        
        # Define specialized agents
        self.agents = {
            "pathology_expert": """
            You are a senior histopathologist with 20+ years of experience.
            Specialized in multi-organ pathology and differential diagnosis.
            Provide detailed pathological insights and clinical correlations.
            Use medical terminology appropriately and be precise in your analysis.
            """,
            
            "ml_interpreter": """
            You are an AI/ML expert specializing in computer vision for healthcare.
            Explain model predictions, confidence scores, and feature importance
            in clinically relevant terms. Bridge the gap between AI outputs and
            clinical decision-making.
            """,
            
            "clinical_advisor": """
            You are a clinical oncologist specializing in treatment planning.
            Provide next steps, additional tests, and treatment considerations
            based on pathological findings. Focus on actionable recommendations
            and evidence-based guidelines.
            """,
            
            "patient_educator": """
            You are a patient educator specializing in explaining complex
            medical concepts in simple, empathetic language. Avoid jargon,
            use analogies when helpful, and be compassionate.
            """
        }
    
    def chat_response(self, message: str, context: Dict = None) -> str:
        """Generate response for chat interface"""
        if not self.enabled:
            return "[SYSTEM] Multi-Agent AI not configured. Please add your Google API key in the sidebar."
        
        try:
            if context:
                prompt = f"""
                Context:
                - Recent Diagnosis: {context.get('diagnosis', 'N/A')}
                - Confidence: {context.get('confidence', 'N/A')}
                - Organ: {context.get('organ', 'N/A')}
                
                User Query: {message}
                
                Provide a helpful, accurate response based on the context and your expertise.
                """
            else:
                prompt = f"User Query: {message}\n\nProvide a helpful medical AI assistant response."
            
            response = self.gemini.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"[ERROR] Agent unavailable: {str(e)}"
    
    def multi_agent_consultation(self, prediction_14: Dict, prediction_5: Dict, clinical_context: str = ""):
        """Multi-agent consultation system"""
        if not self.enabled:
            return {
                "pathology_insights": "[SYSTEM] Multi-Agent AI not configured. Please add your Google API key.",
                "ml_interpretation": "[SYSTEM] Multi-Agent AI not configured. Please add your Google API key.",
                "clinical_recommendations": "[SYSTEM] Multi-Agent AI not configured. Please add your Google API key.",
                "patient_explanation": "[SYSTEM] Multi-Agent AI not configured. Please add your Google API key.",
                "consensus_diagnosis": "[SYSTEM] Multi-Agent AI not configured. Please add your Google API key."
            }
        
        consultation_report = {
            "pathology_insights": "",
            "ml_interpretation": "",
            "clinical_recommendations": "",
            "patient_explanation": "",
            "consensus_diagnosis": ""
        }
        
        # Agent 1: Pathology Expert
        pathology_prompt = f"""
        As a senior pathologist, analyze this histopathology case:
        
        Image Analysis Results:
        - 14-Class Prediction: {prediction_14['class']} (Confidence: {prediction_14['confidence']:.2%})
        - Organ-Level Prediction: {prediction_5['class']} (Confidence: {prediction_5['confidence']:.2%})
        - Clinical Context: {clinical_context if clinical_context else 'Not provided'}
        
        Provide:
        1. Pathological features supporting this diagnosis
        2. Differential diagnoses to consider
        3. Key histological characteristics
        4. Potential pitfalls or confounding factors
        
        Keep response concise but comprehensive.
        """
        
        consultation_report["pathology_insights"] = self._query_agent(
            pathology_prompt, "pathology_expert"
        )
        
        # Agent 2: ML Interpreter
        ml_prompt = f"""
        As an AI expert, interpret these model predictions:
        
        Model Outputs:
        - Detailed Classification: {prediction_14['class']} ({prediction_14['confidence']:.2%})
        - Organ Classification: {prediction_5['class']} ({prediction_5['confidence']:.2%})
        
        Explain:
        1. What features the model likely detected
        2. Confidence score interpretation and reliability
        3. Model limitations for this case
        4. Suggestions for pathologist verification
        
        Be technical but accessible.
        """
        
        consultation_report["ml_interpretation"] = self._query_agent(
            ml_prompt, "ml_interpreter"
        )
        
        # Agent 3: Clinical Advisor
        clinical_prompt = f"""
        As a clinical oncologist, provide recommendations:
        
        Diagnosis: {prediction_14['class']}
        Organ: {prediction_5['class']}
        Confidence: {prediction_14['confidence']:.2%}
        
        Suggest:
        1. Next diagnostic steps
        2. Additional tests needed
        3. Treatment considerations
        4. Prognostic implications
        
        Focus on actionable clinical guidance.
        """
        
        consultation_report["clinical_recommendations"] = self._query_agent(
            clinical_prompt, "clinical_advisor"
        )
        
        # Agent 4: Patient Educator
        patient_prompt = f"""
        Explain this diagnosis to a patient:
        
        Medical Diagnosis: {prediction_14['class']}
        Organ: {prediction_5['class']}
        
        Provide:
        1. Simple explanation of the condition
        2. What to expect next
        3. Questions to ask their doctor
        4. Emotional support guidance
        
        Use simple language, be compassionate and reassuring.
        """
        
        consultation_report["patient_explanation"] = self._query_agent(
            patient_prompt, "patient_educator"
        )
        
        # Generate consensus
        consensus_prompt = f"""
        Generate a consensus diagnostic report integrating expert opinions:
        
        Summary:
        - Diagnosis: {prediction_14['class']}
        - Confidence: {prediction_14['confidence']:.2%}
        - Organ: {prediction_5['class']}
        
        Create a unified diagnostic summary with:
        1. Primary diagnosis and confidence assessment
        2. Key supporting features
        3. Critical next steps
        4. Overall recommendation
        
        Keep it concise and actionable.
        """
        
        consultation_report["consensus_diagnosis"] = self._query_agent(
            consensus_prompt, "pathology_expert"
        )
        
        return consultation_report
    
    def _query_agent(self, prompt: str, agent_role: str) -> str:
        """Query specific agent with role-based context"""
        if not self.enabled:
            return "[SYSTEM] Agent not available"
        
        full_prompt = f"{self.agents[agent_role]}\n\n{prompt}"
        
        try:
            response = self.gemini.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"[ERROR] Agent {agent_role}: {str(e)}"

def generate_multi_agent_consultation(gemini_model, prediction_14: Dict, prediction_5: Dict, clinical_context: str = ""):
    """Generate multi-agent consultation using provided Gemini model"""
    
    consultation_report = {
        "pathology_insights": "",
        "ml_interpretation": "",
        "clinical_recommendations": "",
        "patient_explanation": "",
        "consensus_diagnosis": ""
    }
    
    # Agent roles
    agents = {
        "pathology_expert": """
        You are a senior histopathologist with 20+ years of experience.
        Specialized in multi-organ pathology and differential diagnosis.
        Provide detailed pathological insights and clinical correlations.
        Use medical terminology appropriately and be precise in your analysis.
        """,
        
        "ml_interpreter": """
        You are an AI/ML expert specializing in computer vision for healthcare.
        Explain model predictions, confidence scores, and feature importance
        in clinically relevant terms. Bridge the gap between AI outputs and
        clinical decision-making.
        """,
        
        "clinical_advisor": """
        You are a clinical oncologist specializing in treatment planning.
        Provide next steps, additional tests, and treatment considerations
        based on pathological findings. Focus on actionable recommendations
        and evidence-based guidelines.
        """,
        
        "patient_educator": """
        You are a patient educator specializing in explaining complex
        medical concepts in simple, empathetic language. Avoid jargon,
        use analogies when helpful, and be compassionate.
        """
    }
    
    # Agent 1: Pathology Expert
    pathology_prompt = f"""
    {agents["pathology_expert"]}
    
    As a senior pathologist, analyze this histopathology case:
    
    Image Analysis Results:
    - 14-Class Prediction: {prediction_14['class']} (Confidence: {prediction_14['confidence']:.2%})
    - Organ-Level Prediction: {prediction_5['class']} (Confidence: {prediction_5['confidence']:.2%})
    - Clinical Context: {clinical_context if clinical_context else 'Not provided'}
    
    Provide:
    1. Pathological features supporting this diagnosis
    2. Differential diagnoses to consider
    3. Key histological characteristics
    4. Potential pitfalls or confounding factors
    
    Keep response concise but comprehensive.
    """
    
    try:
        response = gemini_model.generate_content(pathology_prompt)
        consultation_report["pathology_insights"] = response.text
    except Exception as e:
        consultation_report["pathology_insights"] = f"[ERROR] Pathology Expert: {str(e)}"
    
    # Agent 2: ML Interpreter
    ml_prompt = f"""
    {agents["ml_interpreter"]}
    
    As an AI expert, interpret these model predictions:
    
    Model Outputs:
    - Detailed Classification: {prediction_14['class']} ({prediction_14['confidence']:.2%})
    - Organ Classification: {prediction_5['class']} ({prediction_5['confidence']:.2%})
    
    Explain:
    1. What features the model likely detected
    2. Confidence score interpretation and reliability
    3. Model limitations for this case
    4. Suggestions for pathologist verification
    
    Be technical but accessible.
    """
    
    try:
        response = gemini_model.generate_content(ml_prompt)
        consultation_report["ml_interpretation"] = response.text
    except Exception as e:
        consultation_report["ml_interpretation"] = f"[ERROR] ML Interpreter: {str(e)}"
    
    # Agent 3: Clinical Advisor
    clinical_prompt = f"""
    {agents["clinical_advisor"]}
    
    As a clinical oncologist, provide recommendations:
    
    Diagnosis: {prediction_14['class']}
    Organ: {prediction_5['class']}
    Confidence: {prediction_14['confidence']:.2%}
    Clinical Context: {clinical_context if clinical_context else 'Not provided'}
    
    Suggest:
    1. Next diagnostic steps
    2. Additional tests needed
    3. Treatment considerations
    4. Prognostic implications
    
    Focus on actionable clinical guidance.
    """
    
    try:
        response = gemini_model.generate_content(clinical_prompt)
        consultation_report["clinical_recommendations"] = response.text
    except Exception as e:
        consultation_report["clinical_recommendations"] = f"[ERROR] Clinical Advisor: {str(e)}"
    
    # Agent 4: Patient Educator
    patient_prompt = f"""
    {agents["patient_educator"]}
    
    Explain this diagnosis to a patient:
    
    Medical Diagnosis: {prediction_14['class']}
    Organ: {prediction_5['class']}
    
    Provide:
    1. Simple explanation of the condition
    2. What to expect next
    3. Questions to ask their doctor
    4. Emotional support guidance
    
    Use simple language, be compassionate and reassuring.
    """
    
    try:
        response = gemini_model.generate_content(patient_prompt)
        consultation_report["patient_explanation"] = response.text
    except Exception as e:
        consultation_report["patient_explanation"] = f"[ERROR] Patient Educator: {str(e)}"
    
    # Generate consensus
    consensus_prompt = f"""
    {agents["pathology_expert"]}
    
    Generate a consensus diagnostic report integrating expert opinions:
    
    Summary:
    - Diagnosis: {prediction_14['class']}
    - Confidence: {prediction_14['confidence']:.2%}
    - Organ: {prediction_5['class']}
    
    Create a unified diagnostic summary with:
    1. Primary diagnosis and confidence assessment
    2. Key supporting features
    3. Critical next steps
    4. Overall recommendation
    
    Keep it concise and actionable.
    """
    
    try:
        response = gemini_model.generate_content(consensus_prompt)
        consultation_report["consensus_diagnosis"] = response.text
    except Exception as e:
        consultation_report["consensus_diagnosis"] = f"[ERROR] Consensus: {str(e)}"
    
    return consultation_report

# Initialize multi_agent as None, will be set after models are loaded
multi_agent = None

# Main application
def main():
    global multi_agent
    
    # Header
    st.markdown('<div class="main-header">HistoPath AI - Multi-Organ Classification System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Select Mode:",
        ["HOME", "IMAGE CLASSIFICATION", "MULTI-AGENT DIAGNOSTIC"]
    )
    
    # Load models
    with st.sidebar:
        st.markdown("### Model Status")
        with st.spinner("Loading models..."):
            model_14 = load_14_class_model()
            model_5 = load_5_class_model()
        
        if model_14 and model_5:
            st.success("✓ Models Loaded")
            st.info(f"Device: {device}")
            
            # Initialize multi-agent system
            if multi_agent is None:
                multi_agent = HistopathologyMultiAgent(model_14, model_5)
        else:
            st.error("✗ Failed to load models")
            return
    
    # Home Page
    if app_mode == "HOME":
        show_home_page()
    
    # Image Classification
    elif app_mode == "IMAGE CLASSIFICATION":
        show_classification_page(model_14, model_5)
    
    # Multi-Agent Diagnostic
    elif app_mode == "MULTI-AGENT DIAGNOSTIC":
        show_multi_agent_page(model_14, model_5)

def show_home_page():
    """Display simplified home page"""

    st.markdown("""
    - **14-Class Classification**: Detailed tissue classification across multiple organs
    - **5-Class Organ Detection**: Breast, Colon, Lung, Oral, Ovarian
    - **AI-Powered Consultation**: Multi-agent diagnostic assistant using Google Gemini
    
    **Supported Organs:** Breast, Colon, Lung, Oral Cavity, Ovarian
    
    **Classification Types:** Benign/Malignant, Normal/Diseased, Specific Cancer Subtypes
    """)
    
    st.markdown("---")
    
    # Quick Guide
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Image Classification")
        st.markdown("""
        1. Go to **Image Classification** page
        2. Upload histopathology image
        3. Get instant predictions from both models
        4. View confidence scores
        """)
    
    with col2:
        st.markdown("### Multi-Agent Consultation")
        st.markdown("""
        1. Go to **Multi-Agent Diagnostic** page
        2. Enter your Google Gemini API key
        3. Upload image for analysis
        4. Get AI-powered consultation from specialized agents
        """)

def show_classification_page(model_14, model_5):
    """Display image classification interface"""
    st.markdown('<div class="sub-header">Image Classification</div>', unsafe_allow_html=True)
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    
    with col1:
        model_type = st.radio(
            "Classification Type:",
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
        help="Supported: PNG, JPG, JPEG, TIF, TIFF"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="UPLOADED IMAGE", width='content')
            
            # Image info
            st.info(f"SIZE: {image.size} | MODE: {image.mode}")
        
        with col2:
            analyze_col, agent_col = st.columns(2)
            
            with analyze_col:
                analyze_button = st.button("ANALYZE IMAGE", type="primary", width='content')
            
            with agent_col:
                use_multi_agent = st.checkbox("ENABLE MULTI-AGENT", value=False, help="Use AI agents for comprehensive analysis")
            
            if analyze_button:
                with st.spinner("ANALYZING..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Model inference for both models
                    with torch.no_grad():
                        # 14-class prediction
                        image_tensor_cuda = image_tensor.to(device)
                        outputs_14 = model_14(image_tensor_cuda)
                        probs_14 = F.softmax(outputs_14, dim=1)
                        conf_14, pred_14 = torch.max(probs_14, 1)
                        
                        # 5-class prediction
                        outputs_5 = model_5(image_tensor_cuda)
                        probs_5 = F.softmax(outputs_5, dim=1)
                        conf_5, pred_5 = torch.max(probs_5, 1)
                        
                        confidence_score = conf_14.item()
                        predicted_class = pred_14.item()
                        all_probabilities = probs_14.cpu().numpy()[0]
                        
                        # Prepare predictions for multi-agent
                        prediction_14 = {
                            'class': CLASSES_14[pred_14.item()],
                            'confidence': conf_14.item()
                        }
                        
                        prediction_5 = {
                            'class': CLASSES_5[pred_5.item()],
                            'confidence': conf_5.item()
                        }
                    
                    # Store in session state for multi-agent
                    st.session_state['last_prediction_14'] = prediction_14
                    st.session_state['last_prediction_5'] = prediction_5
                    st.session_state['last_image'] = image
                    
                    # Display results
                    if confidence_score >= confidence_threshold:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <p style='font-size: 0.75rem; margin-bottom: 0.5rem;'>PREDICTION RESULT</p>
                            <p style='font-size: 1.2rem; font-weight: 600; margin: 0.5rem 0;'>{CLASSES_14[predicted_class].replace('_', ' ').upper()}</p>
                            <div class="confidence-bar" style="width: {confidence_score*100}%"></div>
                            <p style='font-size: 0.9rem; margin-top: 0.5rem;'>{confidence_score*100:.2f}% CONFIDENCE</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Organ tag
                        organ = ORGAN_MAPPING.get(CLASSES_14[predicted_class], "Unknown")
                        st.markdown(f"""
                        <div style='background: rgba(102, 126, 234, 0.1); border: 1px solid #667eea;
                                    color: #667eea; padding: 0.4rem; border-radius: 3px; 
                                    text-align: center; font-weight: 500; margin: 1rem 0; font-size: 0.8rem;'>
                            [{organ.upper()} ORGAN]
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(f"[WARNING] LOW CONFIDENCE: {confidence_score*100:.1f}%")
                        st.info("[INFO] Consult a pathologist for verification")
                    
                    # Multi-Agent Consultation
                    if use_multi_agent and multi_agent and multi_agent.enabled:
                        st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
                        st.markdown("### >> MULTI-AGENT CONSULTATION")
                        
                        with st.spinner("CONSULTING AI AGENTS..."):
                            clinical_context = st.text_area(
                                "CLINICAL CONTEXT (OPTIONAL):",
                                placeholder="Enter patient history, symptoms, or additional context...",
                                height=80,
                                key="clinical_context"
                            )
                            
                            consultation = multi_agent.multi_agent_consultation(
                                prediction_14, 
                                prediction_5, 
                                clinical_context
                            )
                            
                            # Display consultation in tabs
                            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                                "CONSENSUS", "PATHOLOGY", "AI INSIGHTS", "CLINICAL", "PATIENT INFO"
                            ])
                            
                            with tab1:
                                st.markdown('<div class="terminal-card">', unsafe_allow_html=True)
                                st.markdown(consultation['consensus_diagnosis'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with tab2:
                                st.markdown('<div class="terminal-card">', unsafe_allow_html=True)
                                st.markdown(consultation['pathology_insights'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with tab3:
                                st.markdown('<div class="terminal-card">', unsafe_allow_html=True)
                                st.markdown(consultation['ml_interpretation'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with tab4:
                                st.markdown('<div class="terminal-card">', unsafe_allow_html=True)
                                st.markdown(consultation['clinical_recommendations'])
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            with tab5:
                                st.markdown('<div class="terminal-card">', unsafe_allow_html=True)
                                st.markdown(consultation['patient_explanation'])
                                st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Probability distribution
                    st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)
                    st.markdown("### >> PROBABILITY DISTRIBUTION")
                    
                    # Create sorted probabilities
                    prob_data = []
                    for i, prob in enumerate(all_probabilities):
                        prob_data.append({
                            'Class': CLASSES_14[i].replace('_', ' ').title(),
                            'Probability': prob
                        })
                    
                    prob_df = pd.DataFrame(prob_data)
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Create bar chart
                    fig = px.bar(
                        prob_df.head(10),
                        x='Probability',
                        y='Class',
                        orientation='h',
                        title="TOP 10 PREDICTIONS"
                    )
                    
                    fig.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        height=400,
                        showlegend=False,
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#2c3e50', family='Inter')
                    )
                    
                    st.plotly_chart(fig, width='content')
                    
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

def show_multi_agent_page(model_14, model_5):
    """Display Multi-Agent Diagnostic Assistant - Models run first, then AI analyzes"""
    st.markdown('<div class="sub-header">Multi-Agent Diagnostic Assistant</div>', unsafe_allow_html=True)
    
    # Step 1: API Configuration
    st.markdown("### Step 1: Configure AI Agent")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        api_key_input = st.text_input(
            "Google Gemini API Key",
            type="password",
            placeholder="Enter your API key from https://makersuite.google.com/app/apikey",
            key="multi_agent_api_key"
        )
    
    with col2:
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
            help="Select Gemini model"
        )
    
    # Validate API
    agent_ready = False
    if api_key_input:
        try:
            if GEMINI_AVAILABLE:
                genai.configure(api_key=api_key_input)
                gemini_model = genai.GenerativeModel(model_name)
                agent_ready = True
                st.success(f"✓ AI Agent Ready ({model_name})")
            else:
                st.error("Google Generative AI library not installed. Run: pip install google-generativeai")
        except Exception as e:
            st.error(f"API Configuration Error: {str(e)}")
    else:
        st.info("Please enter your API key to continue")
    
    st.markdown("---")
    
    # Step 2: Image Upload & Model Analysis
    st.markdown("### Step 2: Upload Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload Histopathology Image",
        type=['png', 'jpg', 'jpeg', 'tif', 'tiff'],
        key="multi_agent_upload"
    )
    
    if uploaded_file and agent_ready:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", width='content')
            st.info(f"Size: {image.size[0]}x{image.size[1]}")
        
        with col2:
            if st.button("🔬 Analyze with AI Agents", type="primary", width='content'):
                with st.spinner("Running models and consulting AI agents..."):
                    # Step 1: Run the models
                    image_tensor = preprocess_image(image)
                    
                    with torch.no_grad():
                        image_tensor_cuda = image_tensor.to(device)
                        
                        # Get predictions from both models
                        outputs_14 = model_14(image_tensor_cuda)
                        probs_14 = F.softmax(outputs_14, dim=1)
                        conf_14, pred_14 = torch.max(probs_14, 1)
                        
                        outputs_5 = model_5(image_tensor_cuda)
                        probs_5 = F.softmax(outputs_5, dim=1)
                        conf_5, pred_5 = torch.max(probs_5, 1)
                    
                    prediction_14 = {
                        'class': CLASSES_14[pred_14.item()],
                        'confidence': conf_14.item()
                    }
                    
                    prediction_5 = {
                        'class': CLASSES_5[pred_5.item()],
                        'confidence': conf_5.item()
                    }
                    
                    # Store predictions
                    st.session_state['multi_agent_prediction_14'] = prediction_14
                    st.session_state['multi_agent_prediction_5'] = prediction_5
                    st.session_state['multi_agent_image'] = image
                    # Do NOT assign to 'multi_agent_api_key' here — the widget with
                    # key 'multi_agent_api_key' already manages its own session state.
                    # Assigning to it after the widget is created raises a Streamlit
                    # API exception. The user's input is already available via the
                    # widget key, so we keep it unchanged here.
                    st.session_state['multi_agent_model_name'] = model_name
                    
                    st.success("✓ Model analysis complete")
                    st.rerun()
    
    # Step 3: Display Results & AI Consultation
    if 'multi_agent_prediction_14' in st.session_state and agent_ready:
        st.markdown("---")
        st.markdown("### Step 3: Model Results & AI Consultation")
        
        pred_14 = st.session_state['multi_agent_prediction_14']
        pred_5 = st.session_state['multi_agent_prediction_5']
        
        # Display model predictions
        st.markdown(f"""
        <div class="prediction-box">
            <strong>Primary Diagnosis:</strong> {pred_14['class'].replace('_', ' ').title()}<br>
            <strong>Confidence:</strong> {pred_14['confidence']*100:.1f}%<br>
            <strong>Organ:</strong> {pred_5['class'].title()}
            <div class="confidence-bar" style="width: {pred_14['confidence']*100}%"></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical context (optional)
        clinical_context = st.text_area(
            "Additional Clinical Context (Optional)",
            placeholder="Patient age, symptoms, medical history, etc.",
            height=80,
            key="clinical_context"
        )
        
        # Generate consultation
        if st.button("🤖 Generate AI Consultation", type="primary", width='content'):
            with st.spinner("AI agents analyzing results..."):
                try:
                    # Safely obtain API key and model name from the current widgets
                    api_key = api_key_input if api_key_input else st.session_state.get('multi_agent_api_key')
                    model_to_use = model_name if model_name else st.session_state.get('multi_agent_model_name')

                    if not api_key:
                        raise ValueError("API key not provided. Enter your Google Gemini API key in the field above.")

                    genai.configure(api_key=api_key)
                    gemini_model = genai.GenerativeModel(model_to_use)

                    consultation = generate_multi_agent_consultation(
                        gemini_model, pred_14, pred_5, clinical_context
                    )

                    st.session_state['consultation_report'] = consultation
                    st.success("✓ AI consultation complete")

                except Exception as e:
                    st.error(f"Consultation failed: {str(e)}")
        
        # Display consultation
        if 'consultation_report' in st.session_state:
            st.markdown("---")
            st.markdown("### AI Consultation Report")
            
            consultation = st.session_state['consultation_report']
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Consensus", "Pathology", "AI Analysis", "Clinical", "Patient Info"
            ])
            
            with tab1:
                st.markdown(consultation['consensus_diagnosis'])
            
            with tab2:
                st.markdown(consultation['pathology_insights'])
            
            with tab3:
                st.markdown(consultation['ml_interpretation'])
            
            with tab4:
                st.markdown(consultation['clinical_recommendations'])
            
            with tab5:
                st.markdown(consultation['patient_explanation'])
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                report_json = json.dumps({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'diagnosis': pred_14,
                    'organ': pred_5,
                    'clinical_context': clinical_context,
                    'consultation': consultation
                }, indent=2)
                
                st.download_button(
                    "📄 Export JSON",
                    report_json,
                    file_name=f"consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col2:
                report_text = f"""HISTOPATH AI - CONSULTATION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

DIAGNOSIS: {pred_14['class']}
CONFIDENCE: {pred_14['confidence']*100:.2f}%
ORGAN: {pred_5['class']}

CLINICAL CONTEXT:
{clinical_context if clinical_context else 'Not provided'}

CONSENSUS DIAGNOSIS:
{consultation['consensus_diagnosis']}

PATHOLOGY INSIGHTS:
{consultation['pathology_insights']}

ML INTERPRETATION:
{consultation['ml_interpretation']}

CLINICAL RECOMMENDATIONS:
{consultation['clinical_recommendations']}

PATIENT EXPLANATION:
{consultation['patient_explanation']}
"""
                
                st.download_button(
                    "📄 Export Text",
                    report_text,
                    file_name=f"consultation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            
            with col3:
                if st.button("🔄 New Analysis"):
                    for key in ['multi_agent_prediction_14', 'multi_agent_prediction_5', 
                               'multi_agent_image', 'consultation_report']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()

# Run the application
if __name__ == "__main__":
    main()