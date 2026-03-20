import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try importing optional visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available. Some visualizations will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Page config
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
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
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
    """Initialize database tables"""
    try:
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        
        # Patients table
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
        
        # Medical records table
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
        return True
    except Exception as e:
        st.error(f"Database error: {e}")
        return False

def add_patient(patient_name, patient_id, age, gender):
    """Add new patient to database"""
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
    except Exception as e:
        return False

def get_patients():
    """Get all patients"""
    try:
        conn = sqlite3.connect('diabetes.db')
        df = pd.read_sql("SELECT * FROM patients ORDER BY created_at DESC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def save_medical_record(patient_id, data, prediction, probability):
    """Save medical record"""
    try:
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
        return True
    except Exception as e:
        return False

def get_patient_history(patient_id):
    """Get patient medical history"""
    try:
        conn = sqlite3.connect('diabetes.db')
        df = pd.read_sql(f'''
            SELECT * FROM medical_records 
            WHERE patient_id = '{patient_id}'
            ORDER BY record_date DESC
        ''', conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and scaler"""
    try:
        with open('model (3).pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        return None, None

@st.cache_data
def load_dataset():
    """Load diabetes dataset"""
    try:
        df = pd.read_csv('diabetes.csv')
        return df
    except Exception as e:
        return None

# Initialize
init_database()
df = load_dataset()
model, scaler = load_model_and_scaler()

# Check if all required files are present
if df is None:
    st.error("❌ Please make sure 'diabetes.csv' is in the directory")
    st.stop()

if model is None or scaler is None:
    st.error("❌ Please make sure 'model (3).pkl' and 'scaler.pkl' are in the directory")
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
    st.metric("Total Patients", len(patients_df))
with col2:
    st.metric("Diabetic Cases", df['Outcome'].sum())
with col3:
    st.metric("Avg Glucose", f"{df['Glucose'].mean():.0f} mg/dL")
with col4:
    st.metric("Avg BMI", f"{df['BMI'].mean():.1f} kg/m²")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Diabetes", "👥 Patient Records", "📊 Visualizations", "📋 History"])

# Tab 1: Prediction
with tab1:
    st.markdown("### Enter Patient Information")
    
    if selected_patient_id:
        st.info(f"👤 Current Patient: **{selected_patient}**")
    else:
        st.warning("⚠️ Please add a patient from the sidebar first")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)
        glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1)
        bp = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70, step=1)
        skin = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20, step=1)
    
    with col2:
        insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80, step=1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01, format="%.3f")
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
    
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
            
            # Show result
            st.markdown("---")
            st.markdown("## 📊 Prediction Result")
            
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box risk-high">
                    <h1 style="font-size: 2.5rem;">⚠️ DIABETES RISK DETECTED</h1>
                    <p style="font-size: 1.8rem; font-weight: bold;">{probability*100:.1f}% Probability</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {probability*100:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">⚠️ Recommendation: Consult healthcare provider immediately</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box risk-low">
                    <h1 style="font-size: 2.5rem;">✅ LOW DIABETES RISK</h1>
                    <p style="font-size: 1.8rem; font-weight: bold;">{(1-probability)*100:.1f}% Probability of being healthy</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {(1-probability)*100:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">✅ Recommendation: Maintain healthy lifestyle</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Risk Factors
            st.markdown("### 🎯 Risk Factors Analysis")
            cols = st.columns(4)
            with cols[0]:
                if glucose > 140:
                    st.error(f"🔴 High Glucose: {glucose} mg/dL")
                else:
                    st.success(f"🟢 Normal Glucose: {glucose} mg/dL")
            with cols[1]:
                if bmi > 30:
                    st.error(f"🔴 High BMI: {bmi}")
                else:
                    st.success(f"🟢 Normal BMI: {bmi}")
            with cols[2]:
                if age > 45:
                    st.error(f"🔴 Age Risk: {age} years")
                else:
                    st.success(f"🟢 Normal Age: {age} years")
            with cols[3]:
                if dpf > 0.8:
                    st.error(f"🔴 High DPF: {dpf}")
                else:
                    st.success(f"🟢 Normal DPF: {dpf}")
            
            # Show recent history
            history = get_patient_history(selected_patient_id)
            if len(history) > 1:
                st.markdown("### 📋 Recent Predictions")
                st.dataframe(history[['record_date', 'glucose', 'bmi', 'risk_level']].head(5), use_container_width=True)
        else:
            st.error("Please add a patient first!")

# Tab 2: Patient Records
with tab2:
    st.markdown("### 👥 All Patients")
    
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
                        st.markdown(f'<span class="badge-risk">⚠️ Last: High Risk ({latest["probability"]*100:.1f}%)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-safe">✅ Last: Low Risk ({(1-latest["probability"])*100:.1f}% Healthy)</span>', unsafe_allow_html=True)
    else:
        st.info("No patients added yet. Add a patient from the sidebar.")

# Tab 3: Visualizations
with tab3:
    st.markdown("### 📊 Data Visualizations")
    
    if PLOTLY_AVAILABLE:
        # Use Plotly for interactive charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.histogram(df, x='Glucose', color='Outcome', nbins=30,
                               color_discrete_map={0: '#10b981', 1: '#ef4444'},
                               title="Glucose Distribution by Diabetes Status")
            fig1.update_layout(template='plotly_white')
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.box(df, x='Outcome', y='BMI', color='Outcome',
                         color_discrete_map={0: '#10b981', 1: '#ef4444'},
                         title="BMI Distribution by Diabetes Status")
            fig2.update_layout(template='plotly_white')
            st.plotly_chart(fig2, use_container_width=True)
        
        # 3D Visualization
        st.subheader("3D Visualization: Glucose, BMI, and Age")
        fig3 = px.scatter_3d(df, x='Glucose', y='BMI', z='Age', color='Outcome',
                            color_discrete_map={0: '#10b981', 1: '#ef4444'},
                            title="3D Distribution of Key Features")
        fig3.update_layout(template='plotly_white')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Correlation Heatmap
        st.subheader("Feature Correlation Matrix")
        corr_matrix = df.corr()
        fig4 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='Viridis',
                        title="Correlation Matrix")
        st.plotly_chart(fig4, use_container_width=True)
        
        # Feature Importance
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
                     title="Feature Importance from Model")
        fig5.update_layout(template='plotly_white')
        st.plotly_chart(fig5, use_container_width=True)
        
    else:
        # Fallback to simple charts using matplotlib
        st.warning("Plotly is not available. Showing basic visualizations.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Glucose Distribution")
            glucose_0 = df[df['Outcome']==0]['Glucose']
            glucose_1 = df[df['Outcome']==1]['Glucose']
            
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(glucose_0, bins=30, alpha=0.5, label='No Diabetes', color='green')
            ax.hist(glucose_1, bins=30, alpha=0.5, label='Diabetes', color='red')
            ax.set_xlabel('Glucose')
            ax.set_ylabel('Frequency')
            ax.set_title('Glucose Distribution')
            ax.legend()
            st.pyplot(fig)
        
        with col2:
            st.subheader("BMI Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            df.boxplot(column='BMI', by='Outcome', ax=ax)
            ax.set_title('BMI by Diabetes Status')
            ax.set_xlabel('Diabetes Status (0=No, 1=Yes)')
            ax.set_ylabel('BMI')
            st.pyplot(fig)
        
        # Simple correlation table
        st.subheader("Feature Correlations with Outcome")
        correlations = df.corr()['Outcome'].sort_values(ascending=False)
        st.dataframe(correlations, use_container_width=True)

# Tab 4: History
with tab4:
    st.markdown("### 📋 All Predictions History")
    
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
            display_df['prediction'] = display_df['prediction'].map({1: '⚠️ High Risk', 0: '✅ Low Risk'})
            display_df['probability'] = display_df['probability'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(display_df, use_container_width=True)
            
            # Summary stats
            st.markdown("### 📊 Summary Statistics")
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
