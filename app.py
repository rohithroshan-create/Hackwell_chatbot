import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from datetime import datetime
import google.generativeai as genai
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI Wellness Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WellnessAssistant:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.load_models()
        self.setup_gemini()
    
    def setup_gemini(self):
        """Setup Google Gemini AI"""
        try:
            # Try environment variable first
            api_key = os.getenv("GOOGLE_API_KEY")
            
            # Fallback to user input
            if not api_key:
                api_key = st.sidebar.text_input("🔑 Google Gemini API Key", type="password")
            
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel(
                    "gemini-pro",
                    system_instruction=(
                        "You are a medical AI assistant specializing in chronic diseases: "
                        "Heart Disease, Type 2 Diabetes, and Hypertension. "
                        "Provide educational information only. Never prescribe medications. "
                        "Always recommend consulting healthcare professionals."
                    )
                )
                self.gemini_available = True
                st.sidebar.success("🤖 Gemini AI Active")
            else:
                self.gemini_available = False
                
        except Exception as e:
            self.gemini_available = False
            st.sidebar.error(f"Gemini setup error: {e}")
    
    def load_models(self):
        """Load trained models"""
        try:
            model_files = {
                'heart_disease': 'heart_disease_model.pkl',
                'diabetes': 'diabetes_model.pkl',
                'hypertension': 'hypertension_model.pkl'
            }
            
            scaler_files = {
                'heart_disease': 'heart_disease_scaler.pkl',
                'diabetes': 'diabetes_scaler.pkl',
                'hypertension': 'hypertension_scaler.pkl'
            }
            
            for condition, file_path in model_files.items():
                if os.path.exists(file_path):
                    self.models[condition] = joblib.load(file_path)
                    if condition in scaler_files and os.path.exists(scaler_files[condition]):
                        self.scalers[condition] = joblib.load(scaler_files[condition])
            
            if os.path.exists('performance_metrics.pkl'):
                self.performance_metrics = joblib.load('performance_metrics.pkl')
            
            if not self.models:
                st.error("⚠️ Models not found. Please upload model files.")
                
        except Exception as e:
            st.error(f"Error loading models: {e}")
    
    def collect_patient_data(self):
        """Collect patient information"""
        st.sidebar.header("📋 Health Assessment")
        
        with st.sidebar.expander("👤 Personal Info", expanded=True):
            name = st.text_input("Name")
            age = st.slider("Age", 18, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with st.sidebar.expander("🩺 Health Metrics"):
            height = st.slider("Height (cm)", 140, 220, 170)
            weight = st.slider("Weight (kg)", 40, 200, 70)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            systolic_bp = st.slider("Systolic BP", 90, 200, 120)
            diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
            glucose = st.slider("Glucose (mg/dL)", 70, 300, 100)
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
            heart_rate = st.slider("Heart Rate", 50, 120, 72)
        
        with st.sidebar.expander("🏃‍♂️ Lifestyle"):
            exercise = st.selectbox("Exercise Frequency", 
                                  ["Never", "1-2x/week", "3-4x/week", "Daily"])
            smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
            sleep_hours = st.slider("Sleep Hours", 4, 12, 8)
            stress_level = st.slider("Stress Level", 1, 10, 5)
        
        return {
            'name': name, 'age': age, 'gender': gender,
            'height': height, 'weight': weight, 'bmi': bmi,
            'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
            'glucose': glucose, 'cholesterol': cholesterol, 'heart_rate': heart_rate,
            'exercise': exercise, 'smoking': smoking,
            'sleep_hours': sleep_hours, 'stress_level': stress_level
        }
    
    def predict_risk(self, patient_data, condition):
        """Predict disease risk"""
        if condition not in self.models:
            return None
        
        try:
            # Prepare features based on condition
            if condition == 'heart_disease':
                features = [
                    patient_data['age'],
                    1 if patient_data['gender'] == 'Male' else 0,
                    1, patient_data['systolic_bp'], patient_data['cholesterol'],
                    1 if patient_data['glucose'] > 120 else 0,
                    0, patient_data['heart_rate'], 0, 0, 1, 0, 2,
                    0 if patient_data['age'] < 40 else 1 if patient_data['age'] < 55 else 2 if patient_data['age'] < 70 else 3
                ]
            
            elif condition == 'diabetes':
                features = [
                    0 if patient_data['gender'] == 'Male' else min(5, (patient_data['age'] - 20) // 5),
                    patient_data['glucose'], patient_data['diastolic_bp'],
                    20, 85, patient_data['bmi'], 0.5, patient_data['age'],
                    0 if patient_data['bmi'] < 18.5 else 1 if patient_data['bmi'] < 25 else 2 if patient_data['bmi'] < 30 else 3,
                    0 if patient_data['glucose'] < 100 else 1 if patient_data['glucose'] < 126 else 2
                ]
            
            elif condition == 'hypertension':
                features = [
                    patient_data['age'], 1 if patient_data['gender'] == 'Male' else 0,
                    1 if patient_data['smoking'] == 'Current' else 0,
                    0, 2 if patient_data['exercise'] == 'Daily' else 1 if '3-4' in patient_data['exercise'] else 0,
                    0, 0, 1 if patient_data['bmi'] > 30 else 0,
                    2 if patient_data['stress_level'] > 7 else 1 if patient_data['stress_level'] > 4 else 0,
                    2, patient_data['sleep_hours'], 8,
                    0 if patient_data['age'] < 35 else 1 if patient_data['age'] < 50 else 2 if patient_data['age'] < 65 else 3
                ]
            
            features_array = np.array(features).reshape(1, -1)
            
            # Scale if scaler available
            if condition in self.scalers:
                features_array = self.scalers[condition].transform(features_array)
            
            # Predict
            model = self.models[condition]
            probability = model.predict_proba(features_array)
            
            if probability < 0.3:
                risk_level, color = "Low", "#4CAF50"
            elif probability < 0.6:
                risk_level, color = "Moderate", "#FF9800"
            else:
                risk_level, color = "High", "#F44336"
            
            return {
                'probability': probability,
                'risk_level': risk_level,
                'color': color
            }
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def create_risk_gauge(self, probability, condition, color):
        """Create risk gauge visualization"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': f"{condition.replace('_', ' ').title()} Risk"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(height=300)
        return fig
    
    def gemini_chat(self, message, context=""):
        """Chat with Gemini AI"""
        if not self.gemini_available:
            return "Gemini AI not available. Please provide API key."
        
        try:
            full_message = f"Patient context: {context}\nQuestion: {message}"
            response = self.gemini_model.generate_content(full_message)
            return response.text
        except Exception as e:
            return f"Error: {e}"

def main():
    st.title("🏥 AI Wellness Assistant")
    st.markdown("**Powered by Google Gemini AI**")
    
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 1rem; border-radius: 5px; margin: 1rem 0;">
        <strong>⚕️ Medical Disclaimer:</strong> This tool provides educational information only. 
        Always consult healthcare professionals for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    assistant = WellnessAssistant()
    
    # Display model performance
    if assistant.performance_metrics:
        st.subheader("📊 Model Performance")
        cols = st.columns(3)
        
        conditions = ['heart_disease', 'diabetes', 'hypertension']
        names = ['Heart Disease', 'Diabetes', 'Hypertension']
        
        for i, (condition, name) in enumerate(zip(conditions, names)):
            if condition in assistant.performance_metrics:
                metrics = assistant.performance_metrics[condition]
                with cols[i]:
                    st.metric(
                        name,
                        f"{metrics['accuracy']:.1%}",
                        f"AUC: {metrics['auc']:.3f}"
                    )
    
    # Collect patient data
    patient_data = assistant.collect_patient_data()
    
    # Analyze button
    if st.sidebar.button("🔍 Analyze Health Risks", type="primary"):
        if not patient_data['name']:
            st.sidebar.error("Please enter your name")
        else:
            with st.spinner("Analyzing health data..."):
                # Predict risks
                predictions = {}
                for condition in ['heart_disease', 'diabetes', 'hypertension']:
                    pred = assistant.predict_risk(patient_data, condition)
                    if pred:
                        predictions[condition] = pred
                
                st.session_state.predictions = predictions
                st.session_state.patient_data = patient_data
    
    # Display results
    if 'predictions' in st.session_state:
        predictions = st.session_state.predictions
        patient_data = st.session_state.patient_data
        
        st.subheader("🎯 Health Risk Assessment")
        
        cols = st.columns(len(predictions))
        for i, (condition, pred) in enumerate(predictions.items()):
            with cols[i]:
                st.markdown(f"""
                <div style="padding: 1rem; border-radius: 10px; margin: 1rem 0; 
                           background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                           border-left: 5px solid {pred['color']};">
                    <h3>{condition.replace('_', ' ').title()}</h3>
                    <h1 style="color: {pred['color']}">{pred['probability']:.1%}</h1>
                    <h4>{pred['risk_level']} Risk</h4>
                </div>
                """, unsafe_allow_html=True)
                
                fig = assistant.create_risk_gauge(pred['probability'], condition, pred['color'])
                st.plotly_chart(fig, use_container_width=True)
        
        # AI Chatbot
        st.subheader("🤖 AI Health Assistant")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Quick questions
        col1, col2, col3 = st.columns(3)
        questions = [
            "What's my biggest risk?",
            "How can I improve?",
            "What should I watch for?"
        ]
        
        for i, question in enumerate(questions):
            if [col1, col2, col3][i].button(question):
                context = f"Age {patient_data['age']}, {patient_data['gender']}, BMI {patient_data['bmi']:.1f}"
                response = assistant.gemini_chat(question, context)
                st.session_state.chat_history.append((question, response))
        
        # Chat input
        user_input = st.text_input("Ask your health question:")
        if st.button("Send") and user_input:
            context = f"Age {patient_data['age']}, risks: {', '.join([f'{k}: {v['probability']:.1%}' for k, v in predictions.items()])}"
            response = assistant.gemini_chat(user_input, context)
            st.session_state.chat_history.append((user_input, response))
        
        # Display chat
        for question, response in st.session_state.chat_history[-5:]:
            st.markdown(f"**You:** {question}")
            st.markdown(f"**🤖 AI:** {response}")
            st.markdown("---")
    
    else:
        st.markdown("""
        ## 👋 Welcome!
        
        Get started by:
        1. Adding your Google Gemini API key in the sidebar
        2. Filling out your health information
        3. Clicking "Analyze Health Risks"
        
        ### Features:
        - 🫀 Heart Disease Risk Assessment
        - 🩸 Type 2 Diabetes Prediction  
        - 🩺 Hypertension Evaluation
        - 🤖 AI-Powered Health Guidance
        """)

if __name__ == "__main__":
    main()

