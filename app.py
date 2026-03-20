import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { padding: 0rem 1rem; }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #667eea;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
        display: inline-block;
    }
    
    /* Prediction Result Boxes */
    .prediction-box {
        text-align: center;
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        animation: fadeIn 0.5s ease-in;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .probability-bar {
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        height: 10px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .probability-fill {
        height: 100%;
        background: white;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .badge-risk {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .badge-safe {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Database functions
def init_database():
    conn = sqlite3.connect('diabetes.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            patient_id TEXT UNIQUE,
            age INTEGER,
            gender TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            record_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pregnancies INTEGER,
            glucose INTEGER,
            blood_pressure INTEGER,
            skin_thickness INTEGER,
            insulin INTEGER,
            bmi REAL,
            diabetes_pedigree REAL,
            age INTEGER,
            prediction INTEGER,
            probability REAL,
            risk_level TEXT,
            FOREIGN KEY (patient_id) REFERENCES patients (patient_id)
        )
    ''')
    conn.commit()
    conn.close()

def add_patient(patient_name, patient_id, age, gender):
    try:
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO patients (patient_name, patient_id, age, gender)
            VALUES (?, ?, ?, ?)
        ''', (patient_name, patient_id, age, gender))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def get_patients():
    conn = sqlite3.connect('diabetes.db')
    df = pd.read_sql("SELECT * FROM patients ORDER BY created_at DESC", conn)
    conn.close()
    return df

def save_medical_record(patient_id, data, prediction, probability):
    conn = sqlite3.connect('diabetes.db')
    cursor = conn.cursor()
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    cursor.execute('''
        INSERT INTO medical_records 
        (patient_id, pregnancies, glucose, blood_pressure, skin_thickness, 
         insulin, bmi, diabetes_pedigree, age, prediction, probability, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, data[0], data[1], data[2], data[3], 
          data[4], data[5], data[6], data[7], int(prediction), float(probability), risk_level))
    conn.commit()
    conn.close()

def get_patient_history(patient_id):
    conn = sqlite3.connect('diabetes.db')
    df = pd.read_sql(f'''
        SELECT * FROM medical_records 
        WHERE patient_id = '{patient_id}'
        ORDER BY record_date DESC
    ''', conn)
    conn.close()
    return df

# Load model and scaler silently
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('model (3).pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        try:
            with open('model.pkl', 'rb') as f:
                model = pickle.load(f)
        except:
            model = None
    
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except:
        scaler = None
    
    return model, scaler

@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv('diabetes.csv')
        return df
    except:
        return None

# Initialize
init_database()
df = load_dataset()
model, scaler = load_model_and_scaler()

if df is None or model is None or scaler is None:
    st.error("❌ Required files not found. Please ensure diabetes.csv, model (3).pkl, and scaler.pkl are in the directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966322.png", width=80)
    st.markdown("## 🏥 Diabetes Prediction System")
    st.markdown("---")
    
    # Patient Management
    st.markdown("### 👤 Add New Patient")
    with st.expander("➕ New Patient", expanded=False):
        patient_name = st.text_input("Full Name")
        patient_id = st.text_input("Patient ID")
        patient_age = st.number_input("Age", 0, 120, 30)
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        if st.button("Add Patient", use_container_width=True):
            if patient_name and patient_id:
                if add_patient(patient_name, patient_id, patient_age, patient_gender):
                    st.success(f"✅ Patient added!")
                    st.rerun()
                else:
                    st.error("Patient ID already exists!")
            else:
                st.warning("Please fill all fields")
    
    # Patient Selection
    st.markdown("### 📋 Select Patient")
    patients_df = get_patients()
    if len(patients_df) > 0:
        patient_options = {f"{row['patient_name']} ({row['patient_id']})": row['patient_id'] 
                          for _, row in patients_df.iterrows()}
        selected_patient = st.selectbox("Choose patient", list(patient_options.keys()))
        selected_patient_id = patient_options[selected_patient]
    else:
        st.info("No patients yet")
        selected_patient_id = None
    
    st.markdown("---")
    st.markdown("### 📊 Statistics")
    st.metric("Total Patients", len(patients_df))
    st.metric("Dataset Size", len(df))
    st.metric("Diabetes Rate", f"{df['Outcome'].mean()*100:.1f}%")

# Main content
st.title("🏥 Diabetes Prediction System")
st.markdown("---")

# Stats Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(patients_df)}</div>
        <div class="metric-label">Total Patients</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Outcome'].sum()}</div>
        <div class="metric-label">Diabetic Cases</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['Glucose'].mean():.0f}</div>
        <div class="metric-label">Avg Glucose</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{df['BMI'].mean():.1f}</div>
        <div class="metric-label">Avg BMI</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Diabetes", "👥 Patient Records", "📊 Visualizations", "📋 History"])

# Tab 1: Prediction
with tab1:
    st.markdown('<span class="section-header">Patient Information</span>', unsafe_allow_html=True)
    
    if selected_patient_id:
        st.info(f"👤 Current Patient: **{selected_patient}**")
    else:
        st.warning("⚠️ Please add a patient from the sidebar first")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
        bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    if st.button("🔮 PREDICT DIABETES RISK", type="primary", use_container_width=True):
        if selected_patient_id:
            # Prepare input
            input_data = [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
            input_array = np.array(input_data).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Save record
            save_medical_record(selected_patient_id, input_data, prediction, probability)
            
            # Show prediction result
            st.markdown("---")
            st.markdown("## 📊 Prediction Result")
            
            if prediction == 1:
                # HIGH RISK - Show Diabetes
                st.markdown(f"""
                <div class="prediction-box risk-high">
                    <h1 style="font-size: 3rem;">⚠️ DIABETES RISK DETECTED</h1>
                    <p style="font-size: 1.5rem;">The patient has a <strong>{probability*100:.1f}%</strong> probability of having diabetes</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {probability*100:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">⚠️ Recommendation: Immediate consultation with healthcare provider recommended</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # LOW RISK - Show Healthy
                st.markdown(f"""
                <div class="prediction-box risk-low">
                    <h1 style="font-size: 3rem;">✅ LOW DIABETES RISK</h1>
                    <p style="font-size: 1.5rem;">The patient has a <strong>{(1-probability)*100:.1f}%</strong> probability of being healthy</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {(1-probability)*100:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">✅ Recommendation: Maintain healthy lifestyle with regular exercise and balanced diet</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk Factors Analysis
            st.markdown("### 🎯 Risk Factors Analysis")
            risk_cols = st.columns(4)
            with risk_cols[0]:
                if glucose > 140:
                    st.error(f"🔴 High Glucose: {glucose} mg/dL")
                else:
                    st.success(f"🟢 Normal Glucose: {glucose} mg/dL")
            with risk_cols[1]:
                if bmi > 30:
                    st.error(f"🔴 High BMI: {bmi}")
                else:
                    st.success(f"🟢 Normal BMI: {bmi}")
            with risk_cols[2]:
                if age > 45:
                    st.error(f"🔴 Age Risk: {age} years")
                else:
                    st.success(f"🟢 Age: {age} years")
            with risk_cols[3]:
                if dpf > 0.8:
                    st.error(f"🔴 High DPF: {dpf}")
                else:
                    st.success(f"🟢 Normal DPF: {dpf}")
            
            # Show recent history for this patient
            history = get_patient_history(selected_patient_id)
            if len(history) > 1:
                st.markdown("### 📋 Recent Predictions")
                st.dataframe(history[['record_date', 'glucose', 'bmi', 'risk_level']].head(5), use_container_width=True)
        else:
            st.error("Please add a patient first!")

# Tab 2: Patient Records
with tab2:
    st.markdown('<span class="section-header">All Patients</span>', unsafe_allow_html=True)
    
    patients = get_patients()
    if len(patients) > 0:
        for _, patient in patients.iterrows():
            with st.expander(f"👤 {patient['patient_name']} (ID: {patient['patient_id']})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Age", patient['age'])
                with col2:
                    st.metric("Gender", patient['gender'])
                with col3:
                    st.metric("Registered", patient['created_at'][:10])
                
                history = get_patient_history(patient['patient_id'])
                if len(history) > 0:
                    st.subheader("Medical History")
                    st.dataframe(history[['record_date', 'glucose', 'bmi', 'risk_level']].head(5), use_container_width=True)
                    
                    latest = history.iloc[0]
                    if latest['prediction'] == 1:
                        st.markdown(f'<span class="badge-risk">⚠️ Last Assessment: High Risk ({latest["probability"]*100:.1f}%)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-safe">✅ Last Assessment: Low Risk ({(1-latest["probability"])*100:.1f}% Healthy)</span>', unsafe_allow_html=True)
    else:
        st.info("No patients added yet. Add a patient from the sidebar.")

# Tab 3: Visualizations
with tab3:
    st.markdown('<span class="section-header">Data Visualizations</span>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Glucose Distribution")
        fig1 = px.histogram(df, x='Glucose', color='Outcome', nbins=30,
                           color_discrete_map={0: '#10b981', 1: '#ef4444'},
                           title="Glucose Levels by Diabetes Status")
        fig1.update_layout(template='plotly_white')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("BMI Distribution")
        fig2 = px.box(df, x='Outcome', y='BMI', color='Outcome',
                     color_discrete_map={0: '#10b981', 1: '#ef4444'},
                     title="BMI by Diabetes Status")
        fig2.update_layout(template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("3D Visualization: Glucose, BMI, and Age")
    fig3 = px.scatter_3d(df, x='Glucose', y='BMI', z='Age', color='Outcome',
                         color_discrete_map={0: '#10b981', 1: '#ef4444'},
                         title="3D Distribution of Key Features",
                         opacity=0.7)
    fig3.update_layout(template='plotly_white')
    st.plotly_chart(fig3, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diabetes by Age Group")
        age_groups = pd.cut(df['Age'], bins=[20,30,40,50,60,100], 
                           labels=['20-30', '30-40', '40-50', '50-60', '60+'])
        age_risk = df.groupby(age_groups)['Outcome'].mean() * 100
        fig4 = px.bar(x=age_risk.index, y=age_risk.values,
                     color=age_risk.values,
                     color_continuous_scale='Viridis',
                     title="Diabetes Rate by Age Group",
                     text=age_risk.round(1))
        fig4.update_traces(textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)
    
    with col2:
        st.subheader("Feature Importance")
        feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=True)
        
        fig5 = px.bar(importance, x='importance', y='feature', orientation='h',
                     color='importance',
                     color_continuous_scale='Viridis',
                     title="Feature Importance")
        st.plotly_chart(fig5, use_container_width=True)

# Tab 4: History
with tab4:
    st.markdown('<span class="section-header">All Predictions History</span>', unsafe_allow_html=True)
    
    try:
        conn = sqlite3.connect('diabetes.db')
        history_df = pd.read_sql('''
            SELECT m.*, p.patient_name 
            FROM medical_records m
            LEFT JOIN patients p ON m.patient_id = p.patient_id
            ORDER BY m.record_date DESC
        ''', conn)
        conn.close()
        
        if len(history_df) > 0:
            display_df = history_df[['record_date', 'patient_name', 'glucose', 'bmi', 
                                      'prediction', 'probability', 'risk_level']].copy()
            display_df['prediction'] = display_df['prediction'].map({1: '⚠️ Diabetes', 0: '✅ Healthy'})
            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(display_df, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Predictions", len(history_df))
            with col2:
                high_risk = len(history_df[history_df['prediction']==1])
                st.metric("High Risk Cases", high_risk)
            with col3:
                avg_risk = history_df['probability'].mean() * 100
                st.metric("Average Risk", f"{avg_risk:.1f}%")
        else:
            st.info("No predictions yet. Make a prediction to see history!")
    except Exception as e:
        st.info("No prediction history found")

st.markdown("---")
st.caption("🏥 Diabetes Prediction System | Powered by Machine Learning | Please consult healthcare provider for medical advice")