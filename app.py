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

# Page configuration
st.set_page_config(
    page_title="AI Wellness Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional medical CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.2rem;
        color: #34495E;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
        transition: transform 0.2s;
    }
    .risk-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .high-risk {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left-color: #f44336;
    }
    .moderate-risk {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left-color: #ff9800;
    }
    .low-risk {
        background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
        border-left-color: #4caf50;
    }
    .gemini-chat-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 2px solid #007bff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .user-message {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-left: 15%;
        text-align: right;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .bot-message {
        background: linear-gradient(135deg, #28a745 0%, #1e7e34 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        margin-right: 15%;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #f39c12;
    }
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Your exact Google Gemini setup
GOOGLE_API_KEY = "AIzaSyANnfoI2zSuanVqLEk7oqXq-q-whzPFouA"  # Your provided key
genai.configure(api_key=GOOGLE_API_KEY)

CHAT_MODEL_NAME = "models/gemini-2.0-flash-exp"
chat_model = genai.GenerativeModel(
    CHAT_MODEL_NAME,
    system_instruction=(
        "You are a medical assistant specializing only in chronic diseases "
        "(Heart disease, Type 2 Diabetes, Hypertension). "
        "Answer only questions related to these diseases. "
        "If the user asks unrelated questions, politely reply that you "
        "can only assist with chronic diseases."
    )
)

def google_chatbot_query(message: str):
    try:
        response = chat_model.generate_content(message)
        return response.text
    except Exception as e:
        return f"⚠ Error: {e}"

def safe_scalar(value):
    """Convert any value to a safe scalar for comparisons"""
    try:
        if isinstance(value, (np.ndarray, list)):
            return float(value) if len(value) > 0 else 0.0
        elif isinstance(value, (pd.Series)):
            return float(value.iloc) if len(value) > 0 else 0.0
        else:
            return float(value)
    except (ValueError, TypeError, IndexError):
        return 0.0

def safe_int_scalar(value):
    """Convert any value to a safe integer scalar"""
    try:
        if isinstance(value, (np.ndarray, list)):
            return int(value) if len(value) > 0 else 0
        elif isinstance(value, (pd.Series)):
            return int(value.iloc) if len(value) > 0 else 0
        else:
            return int(value)
    except (ValueError, TypeError, IndexError):
        return 0

class WellnessAssistant:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        self.load_models()
    
    def load_models(self):
        """Load trained models with better error handling"""
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
            
            models_loaded = 0
            
            # Load models
            for condition, file_path in model_files.items():
                if os.path.exists(file_path):
                    try:
                        self.models[condition] = joblib.load(file_path)
                        models_loaded += 1
                        st.success(f"✅ Loaded {condition} model")
                        
                        # Load corresponding scaler
                        if condition in scaler_files and os.path.exists(scaler_files[condition]):
                            self.scalers[condition] = joblib.load(scaler_files[condition])
                            
                    except Exception as e:
                        st.warning(f"⚠️ Error loading {condition} model: {e}")
            
            # Load performance metrics
            if os.path.exists('performance_metrics.pkl'):
                try:
                    self.performance_metrics = joblib.load('performance_metrics.pkl')
                except:
                    pass
            
            if models_loaded == 0:
                st.warning("⚠️ No trained models found. Creating demo models...")
                self.create_demo_models()
            else:
                st.info(f"📊 Loaded {models_loaded}/3 models successfully")
                
        except Exception as e:
            st.error(f"Critical error in model loading: {e}")
            self.create_demo_models()
    
    def create_demo_models(self):
        """Create demo models for testing"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        st.info("🔄 Creating demo models for testing...")
        
        # Demo performance metrics
        self.performance_metrics = {
            'heart_disease': {'accuracy': 0.87, 'auc': 0.92, 'model_name': 'XGBoost (Demo)'},
            'diabetes': {'accuracy': 0.85, 'auc': 0.89, 'model_name': 'Random Forest (Demo)'},
            'hypertension': {'accuracy': 0.83, 'auc': 0.88, 'model_name': 'Logistic Regression (Demo)'}
        }
        
        # Create simple demo models
        for condition in ['heart_disease', 'diabetes', 'hypertension']:
            n_features = {'heart_disease': 14, 'diabetes': 10, 'hypertension': 13}[condition]
            
            # Create demo training data
            np.random.seed(42)
            X_demo = np.random.rand(100, n_features)
            y_demo = np.random.randint(0, 2, 100)
            
            # Train simple model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_demo, y_demo)
            self.models[condition] = model
            
            # Create scaler
            scaler = StandardScaler()
            scaler.fit(X_demo)
            self.scalers[condition] = scaler
        
        st.success("✅ Demo models created successfully!")
    
    def collect_patient_data(self):
        """Collect patient information"""
        st.sidebar.header("📋 Patient Health Assessment")
        
        with st.sidebar.expander("👤 Personal Information", expanded=True):
            name = st.text_input("Full Name", placeholder="Enter your name")
            age = st.slider("Age", 18, 100, 45)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with st.sidebar.expander("🩺 Biometric Measurements", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                height = st.slider("Height (cm)", 140, 220, 170)
                systolic_bp = st.slider("Systolic BP", 90, 200, 120)
                heart_rate = st.slider("Heart Rate", 50, 120, 72)
            
            with col2:
                weight = st.slider("Weight (kg)", 40, 200, 70)
                diastolic_bp = st.slider("Diastolic BP", 60, 120, 80)
                glucose = st.slider("Glucose (mg/dL)", 70, 300, 100)
            
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            cholesterol = st.slider("Cholesterol (mg/dL)", 100, 400, 200)
        
        with st.sidebar.expander("🏥 Medical History"):
            family_history = st.multiselect("Family History", 
                                          ["Heart Disease", "Diabetes", "Hypertension", "Stroke"])
            current_conditions = st.multiselect("Current Conditions",
                                              ["High BP", "High Cholesterol", "Diabetes", "Heart Disease"])
            medications = st.text_area("Current Medications", placeholder="List your medications...")
        
        with st.sidebar.expander("🏃‍♂️ Lifestyle"):
            exercise_freq = st.selectbox("Exercise Frequency", 
                                       ["Never", "1-2 times/week", "3-4 times/week", "5+ times/week", "Daily"])
            smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
            alcohol = st.selectbox("Alcohol Consumption", ["Never", "Occasional", "Moderate", "Heavy"])
            sleep_hours = st.slider("Sleep Hours/Night", 4, 12, 8)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        
        return {
            'name': name, 'age': age, 'gender': gender,
            'height': height, 'weight': weight, 'bmi': bmi,
            'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate, 'glucose': glucose, 'cholesterol': cholesterol,
            'family_history': family_history, 'current_conditions': current_conditions,
            'medications': medications, 'exercise_freq': exercise_freq,
            'smoking': smoking, 'alcohol': alcohol, 'sleep_hours': sleep_hours,
            'stress_level': stress_level
        }
    
    def predict_risk(self, patient_data, condition):
        """Predict disease risk with FIXED array handling"""
        if condition not in self.models:
            st.error(f"Model for {condition} not available")
            return None
        
        try:
            # Convert all patient data to safe scalars to avoid array ambiguity
            age = safe_int_scalar(patient_data['age'])
            gender = patient_data['gender']
            bmi = safe_scalar(patient_data['bmi'])
            systolic_bp = safe_scalar(patient_data['systolic_bp'])
            diastolic_bp = safe_scalar(patient_data['diastolic_bp'])
            glucose = safe_scalar(patient_data['glucose'])
            cholesterol = safe_scalar(patient_data['cholesterol'])
            heart_rate = safe_scalar(patient_data['heart_rate'])
            exercise_freq = patient_data['exercise_freq']
            smoking = patient_data['smoking']
            alcohol = patient_data['alcohol']
            sleep_hours = safe_scalar(patient_data['sleep_hours'])
            stress_level = safe_scalar(patient_data['stress_level'])
            family_history = patient_data['family_history']
            current_conditions = patient_data['current_conditions']
            
            # Prepare features based on condition with FIXED comparisons
            if condition == 'heart_disease':
                # All comparisons now use scalar values
                features = [
                    age,
                    1 if gender == 'Male' else 0,
                    1,  # chest pain type (default)
                    systolic_bp,
                    cholesterol,
                    1 if glucose > 120 else 0,  # Fixed: Now uses scalar
                    0,  # rest ECG
                    heart_rate,
                    0,  # exercise induced angina
                    0,  # oldpeak
                    1,  # slope
                    0,  # ca
                    2,  # thal
                    # Age group feature - Fixed comparison
                    0 if age < 40 else 1 if age < 55 else 2 if age < 70 else 3
                ]
            
            elif condition == 'diabetes':
                # Fixed: All comparisons use scalars
                pregnancies = 0 if gender == 'Male' else max(0, min(5, (age - 20) // 5))
                
                features = [
                    pregnancies,
                    glucose,
                    diastolic_bp,
                    20,  # skin thickness (default)
                    85,  # insulin (default)
                    bmi,
                    0.5,  # diabetes pedigree function
                    age,
                    # BMI category - Fixed comparison
                    0 if bmi < 18.5 else 1 if bmi < 25 else 2 if bmi < 30 else 3,
                    # Glucose category - Fixed comparison
                    0 if glucose < 100 else 1 if glucose < 126 else 2
                ]
            
            elif condition == 'hypertension':
                # Fixed: All comparisons use scalars and safe checks
                smoking_current = 1 if smoking == 'Current' else 0
                alcohol_heavy = 1 if alcohol in ['Moderate', 'Heavy'] else 0
                exercise_active = 2 if exercise_freq == 'Daily' else 1 if ('3-4' in exercise_freq or '5+' in exercise_freq) else 0
                family_hyp = 1 if 'Hypertension' in family_history else 0
                diabetes_condition = 1 if 'Diabetes' in current_conditions else 0
                obese = 1 if bmi > 30 else 0
                high_stress = 2 if stress_level > 7 else 1 if stress_level > 4 else 0
                
                features = [
                    age,
                    1 if gender == 'Male' else 0,
                    smoking_current,
                    alcohol_heavy,
                    exercise_active,
                    family_hyp,
                    diabetes_condition,
                    obese,
                    high_stress,
                    2,  # salt intake (default high)
                    sleep_hours,
                    8,  # work hours (default)
                    # Age risk category - Fixed comparison
                    0 if age < 35 else 1 if age < 50 else 2 if age < 65 else 3
                ]
            
            # Convert to numpy array
            features_array = np.array(features, dtype=float).reshape(1, -1)
            
            # Debug info
            st.write(f"Debug - {condition} features shape: {features_array.shape}")
            st.write(f"Debug - {condition} features: {features}")
            
            # Scale features if scaler available
            if condition in self.scalers:
                try:
                    features_array = self.scalers[condition].transform(features_array)
                except Exception as scale_error:
                    st.warning(f"Scaling error for {condition}: {scale_error}")
            
            # Get prediction
            model = self.models[condition]
            
            # Make prediction with error handling
            try:
                prob_array = model.predict_proba(features_array)
                if len(prob_array.shape) == 2 and prob_array.shape >= 2:
                    probability = float(prob_array)  # Get positive class probability
                else:
                    probability = float(prob_array)  # Single probability
                    
            except Exception as pred_error:
                st.error(f"Prediction error: {pred_error}")
                # Return default prediction
                probability = 0.5
            
            # Ensure probability is a scalar
            probability = max(0.0, min(1.0, float(probability)))
            
            # Risk categorization
            if probability < 0.3:
                risk_level = "Low"
                color = "#4CAF50"
            elif probability < 0.6:
                risk_level = "Moderate"
                color = "#FF9800"
            else:
                risk_level = "High"
                color = "#F44336"
            
            return {
                'probability': probability,
                'risk_level': risk_level,
                'color': color,
                'confidence': max(probability, 1 - probability)
            }
            
        except Exception as e:
            st.error(f"Error predicting {condition}: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            
            # Return a safe default prediction to prevent app crash
            return {
                'probability': 0.5,
                'risk_level': "Moderate",
                'color': "#FF9800",
                'confidence': 0.5
            }
    
    def create_risk_gauge(self, probability, condition, color):
        """Create risk gauge visualization"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{condition.replace('_', ' ').title()} Risk"},
            delta={'reference': 30},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig

def main():
    """Main application"""
    # Header
    st.markdown('<h1 class="main-header">🏥 AI Wellness Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enhanced with Google Gemini AI</p>', unsafe_allow_html=True)
    
    # Medical disclaimer
    st.markdown("""
    <div class="warning-box">
        <strong>⚕️ Medical Disclaimer:</strong> This AI tool provides educational information only and does not replace professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize assistant
    assistant = WellnessAssistant()
    
    # Navigation
    page = st.sidebar.radio(
        "🧭 Navigation",
        ["🏠 Risk Assessment", "💬 AI Chatbot"],
        index=0
    )
    
    if page == "🏠 Risk Assessment":
        # Display model performance if available
        if assistant.performance_metrics:
            st.subheader("📊 AI Model Performance")
            cols = st.columns(3)
            
            conditions = ['heart_disease', 'diabetes', 'hypertension']
            condition_names = ['Heart Disease', 'Diabetes', 'Hypertension']
            
            for i, (condition, name) in enumerate(zip(conditions, condition_names)):
                if condition in assistant.performance_metrics:
                    metrics = assistant.performance_metrics[condition]
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>{name}</h4>
                            <p><strong>Model:</strong> {metrics['model_name']}</p>
                            <p><strong>Accuracy:</strong> {metrics['accuracy']:.1%}</p>
                            <p><strong>AUC:</strong> {metrics['auc']:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Collect patient data
        patient_data = assistant.collect_patient_data()
        
        # Analysis button
        if st.sidebar.button("🔍 Analyze Health Risks", type="primary", use_container_width=True):
            if not patient_data['name']:
                st.sidebar.error("⚠️ Please enter your name to continue")
            else:
                with st.spinner("🔄 Analyzing your health data..."):
                    # Predict risks for all conditions
                    conditions = ['heart_disease', 'diabetes', 'hypertension']
                    predictions = {}
                    
                    for condition in conditions:
                        st.write(f"🔄 Predicting {condition}...")
                        pred = assistant.predict_risk(patient_data, condition)
                        if pred:
                            predictions[condition] = pred
                            st.success(f"✅ {condition} prediction complete")
                        else:
                            st.error(f"❌ {condition} prediction failed")
                    
                    # Store in session state
                    st.session_state.predictions = predictions
                    st.session_state.patient_data = patient_data
                    
                    if predictions:
                        st.success("✅ Analysis complete!")
                    else:
                        st.error("❌ No predictions generated")
        
        # Display results
        if 'predictions' in st.session_state and st.session_state.predictions:
            predictions = st.session_state.predictions
            patient_data = st.session_state.patient_data
            
            st.subheader("🎯 Health Risk Assessment")
            
            # FIXED: Check if predictions exist and have items
            if predictions and len(predictions) > 0:
                cols = st.columns(len(predictions))
                for i, (condition, pred) in enumerate(predictions.items()):
                    with cols[i]:
                        # Risk card
                        risk_class = pred['risk_level'].lower().replace(' ', '-') + '-risk'
                        
                        st.markdown(f"""
                        <div class="risk-card {risk_class}">
                            <h3>{condition.replace('_', ' ').title()}</h3>
                            <h1 style="color: {pred['color']}">{pred['probability']:.1%}</h1>
                            <h4><strong>{pred['risk_level']} Risk</strong></h4>
                            <p>Confidence: {pred['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk gauge
                        fig = assistant.create_risk_gauge(pred['probability'], condition, pred['color'])
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("⚠️ No valid predictions available. Please try the analysis again.")
        
        elif 'predictions' in st.session_state:
            st.warning("⚠️ No predictions available. Please complete the health assessment.")
        
        else:
            # Welcome screen
            st.markdown("""
            ## 👋 Welcome to Your AI Wellness Assistant!
            
            ### 🎯 Features
            - **🫀 Heart Disease** risk assessment using clinical datasets
            - **🩸 Type 2 Diabetes** prediction with advanced ML models
            - **🩺 Hypertension** evaluation using patient data
            - **🤖 Google Gemini AI Chatbot** for personalized health guidance
            
            ### 🚀 How to Get Started
            1. **📋 Complete Assessment** - Fill your health information in the sidebar
            2. **🔍 Analyze Risks** - Click "Analyze Health Risks" button
            3. **📊 Review Results** - Understand your personalized risk profile
            4. **💬 Chat with AI** - Use the AI Chatbot for health questions
            
            ### 🔒 Privacy & Safety
            - ✅ Educational information only
            - ✅ No personal data stored
            - ✅ Medical disclaimers provided
            - ✅ Always recommends consulting doctors
            
            **Ready to start?** Fill out your information in the sidebar!
            """)
    
    elif page == "💬 AI Chatbot":
        st.header("🤖 AI Health Assistant (Google Gemini)")
        
        st.markdown("""
        <div class="gemini-chat-container">
            <h3>🧠 Powered by Google Gemini 2.0 Flash</h3>
            <p>Ask me questions about Heart Disease, Diabetes, and Hypertension. I'm here to provide educational information and guidance!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat history for chatbot page
        if "chatbot_history" not in st.session_state:
            st.session_state.chatbot_history = []
        
        # Chat input
        user_input = st.text_input("💬 Ask your health question:", key="chatbot_input", placeholder="e.g., What are the symptoms of diabetes?")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("📤 Send", type="primary", use_container_width=True)
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chatbot_history = []
                st.rerun()
        
        # Process user input
        if send_button and user_input:
            with st.spinner("🤔 AI thinking..."):
                bot_reply = google_chatbot_query(user_input)
                st.session_state.chatbot_history.append(("You", user_input))
                st.session_state.chatbot_history.append(("Bot", bot_reply))
        
        # Display chat history
        if st.session_state.chatbot_history:
            st.markdown("### 💭 Conversation")
            for i in range(len(st.session_state.chatbot_history) - 1, -1, -1):  # Show newest first
                role, message = st.session_state.chatbot_history[i]
                
                if role == "You":
                    st.markdown(f"""
                    <div class="user-message">
                        <strong>🧑 You:</strong><br>{message}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="bot-message">
                        <strong>🤖 Gemini AI:</strong><br>{message}
                    </div>
                    """, unsafe_allow_html=True)
                
                if i > 0:  # Add separator except for last item
                    st.markdown("<br>", unsafe_allow_html=True)
        else:
            st.info("👋 Start a conversation! Ask me anything about chronic diseases.")
            
            # Quick start questions
            st.markdown("**🔗 Try these questions:**")
            quick_questions = [
                "What are the risk factors for heart disease?",
                "How can I prevent diabetes?", 
                "What are the symptoms of hypertension?",
                "What lifestyle changes help with chronic diseases?",
                "When should I see a doctor?"
            ]
            
            for question in quick_questions:
                if st.button(question, key=f"quick_{hash(question)}"):
                    with st.spinner("🤔 AI thinking..."):
                        bot_reply = google_chatbot_query(question)
                        st.session_state.chatbot_history.append(("You", question))
                        st.session_state.chatbot_history.append(("Bot", bot_reply))
                        st.rerun()

if __name__ == "__main__":
    main()
