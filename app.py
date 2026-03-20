import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    
    .risk-diabetes {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .risk-healthy {
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
    
    .badge-diabetes {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        display: inline-block;
    }
    
    .badge-healthy {
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
    try:
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
        return True
    except:
        return False

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
    try:
        conn = sqlite3.connect('diabetes.db')
        df = pd.read_sql("SELECT * FROM patients ORDER BY created_at DESC", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def save_medical_record(patient_id, data, prediction, probability):
    try:
        conn = sqlite3.connect('diabetes.db')
        cursor = conn.cursor()
        risk_level = "Diabetes" if prediction == 1 else "Healthy"
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
    except:
        return False

def get_patient_history(patient_id):
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

def get_all_records():
    try:
        conn = sqlite3.connect('diabetes.db')
        df = pd.read_sql('''
            SELECT m.*, p.patient_name, p.gender
            FROM medical_records m
            LEFT JOIN patients p ON m.patient_id = p.patient_id
            ORDER BY m.record_date DESC
        ''', conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        with open('model (3).pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

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
    
    st.markdown("### 📋 Select Patient")
    patients_df = get_patients()
    if len(patients_df) > 0:
        patient_options = {f"{row['patient_name']} ({row['patient_id']})": row['patient_id'] 
                          for _, row in patients_df.iterrows()}
        selected_patient = st.selectbox("Choose patient", list(patient_options.keys()))
        selected_patient_id = patient_options[selected_patient]
        
        # Get patient gender for conditional fields
        patient_info = patients_df[patients_df['patient_id'] == selected_patient_id].iloc[0]
        patient_gender_info = patient_info['gender']
    else:
        st.info("No patients yet")
        selected_patient_id = None
        patient_gender_info = None
    
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
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Diabetes", "👥 Patient Records", "📊 Visualizations", "📋 History & Analytics"])

# Tab 1: Prediction
with tab1:
    st.markdown("### Enter Patient Information")
    
    if selected_patient_id:
        st.info(f"👤 Current Patient: **{selected_patient}** | Gender: **{patient_gender_info}**")
    else:
        st.warning("⚠️ Please add a patient from the sidebar first")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Conditional field: Pregnancies only for Female patients
        if patient_gender_info == "Female":
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1,
                                         help="Only applicable for female patients")
        else:
            pregnancies = 0
            st.info("ℹ️ Pregnancies field is not applicable for male patients. Default value 0 will be used.")
        
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
                <div class="prediction-box risk-diabetes">
                    <h1 style="font-size: 2.5rem;">🩺 DIABETES DETECTED</h1>
                    <p style="font-size: 1.8rem; font-weight: bold;">{probability*100:.1f}% Probability of Diabetes</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {probability*100:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">⚠️ Recommendation: Consult healthcare provider immediately for further evaluation</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box risk-healthy">
                    <h1 style="font-size: 2.5rem;">✅ HEALTHY</h1>
                    <p style="font-size: 1.8rem; font-weight: bold;">{(1-probability)*100:.1f}% Probability of Being Healthy</p>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: {(1-probability)*100:.0f}%;"></div>
                    </div>
                    <p style="margin-top: 1rem;">✅ Recommendation: Maintain healthy lifestyle with regular exercise and balanced diet</p>
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
                col1.metric("Age", patient['age'])
                col2.metric("Gender", patient['gender'])
                col3.metric("Registered", patient['created_at'][:10])
                
                history = get_patient_history(patient['patient_id'])
                if len(history) > 0:
                    st.subheader("Medical History")
                    display_history = history[['record_date', 'glucose', 'bmi', 'risk_level', 'probability']].copy()
                    display_history['probability'] = display_history['probability'].apply(lambda x: f"{x*100:.1f}%")
                    st.dataframe(display_history.head(5), use_container_width=True)
                    
                    latest = history.iloc[0]
                    if latest['prediction'] == 1:
                        st.markdown(f'<span class="badge-diabetes">🩺 Last: Diabetes ({latest["probability"]*100:.1f}%)</span>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<span class="badge-healthy">✅ Last: Healthy ({(1-latest["probability"])*100:.1f}%)</span>', unsafe_allow_html=True)
    else:
        st.info("No patients added yet")

# Tab 3: Visualizations
with tab3:
    st.markdown("### 📊 Data Visualizations")
    
    # Set style
    plt.style.use('default')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Glucose Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        glucose_0 = df[df['Outcome']==0]['Glucose']
        glucose_1 = df[df['Outcome']==1]['Glucose']
        ax.hist(glucose_0, bins=30, alpha=0.5, label='Healthy', color='green', edgecolor='black')
        ax.hist(glucose_1, bins=30, alpha=0.5, label='Diabetes', color='red', edgecolor='black')
        ax.set_xlabel('Glucose Level (mg/dL)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Glucose Distribution by Health Status', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.subheader("BMI Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='BMI', by='Outcome', ax=ax, patch_artist=True)
        ax.set_title('BMI Distribution by Health Status', fontsize=14, fontweight='bold')
        ax.set_xlabel('Health Status (0 = Healthy, 1 = Diabetes)', fontsize=12)
        ax.set_ylabel('BMI (kg/m²)', fontsize=12)
        st.pyplot(fig)
        plt.close()
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Matrix")
    fig, ax = plt.subplots(figsize=(12, 10))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                square=True, linewidths=1, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    plt.close()
    
    # Feature Importance
    st.subheader("Feature Importance")
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': abs(model.coef_[0])
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Viridis(importance['importance'] / importance['importance'].max())
    ax.barh(importance['feature'], importance['importance'], color=colors)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.set_title('Feature Importance from Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    for i, (feature, imp) in enumerate(zip(importance['feature'], importance['importance'])):
        ax.text(imp + 0.01, i, f'{imp:.3f}', va='center')
    
    st.pyplot(fig)
    plt.close()
    
    # Diabetes by Age Group
    st.subheader("Diabetes Rate by Age Group")
    age_groups = pd.cut(df['Age'], bins=[20, 30, 40, 50, 60, 100], 
                        labels=['20-30', '30-40', '40-50', '50-60', '60+'])
    age_risk = df.groupby(age_groups)['Outcome'].mean() * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(age_risk.index, age_risk.values, color=plt.cm.Viridis(age_risk.values / 100))
    ax.set_xlabel('Age Group', fontsize=12)
    ax.set_ylabel('Diabetes Rate (%)', fontsize=12)
    ax.set_title('Diabetes Prevalence by Age Group', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, value in zip(bars, age_risk.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}%', ha='center', va='bottom')
    
    st.pyplot(fig)
    plt.close()

# Tab 4: History & Analytics
with tab4:
    st.markdown("### 📋 All Predictions History & Analytics")
    
    all_records = get_all_records()
    
    if len(all_records) > 0:
        # Display records table
        display_df = all_records[['record_date', 'patient_name', 'gender', 'glucose', 'bmi', 
                                  'prediction', 'probability', 'risk_level']].copy()
        display_df['prediction'] = display_df['prediction'].map({1: '🩺 Diabetes', 0: '✅ Healthy'})
        display_df['probability'] = display_df['probability'].apply(lambda x: f"{x*100:.1f}%")
        st.dataframe(display_df, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Analytics Dashboard")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", len(all_records))
        with col2:
            diabetes_count = len(all_records[all_records['prediction']==1])
            st.metric("Diabetes Cases", diabetes_count)
        with col3:
            healthy_count = len(all_records[all_records['prediction']==0])
            st.metric("Healthy Cases", healthy_count)
        with col4:
            avg_risk = all_records['probability'].mean() * 100
            st.metric("Average Risk", f"{avg_risk:.1f}%")
        
        # Trend over time
        st.subheader("📈 Risk Trend Over Time")
        all_records['record_date'] = pd.to_datetime(all_records['record_date'])
        all_records['date_only'] = all_records['record_date'].dt.date
        daily_avg = all_records.groupby('date_only')['probability'].mean() * 100
        
        if len(daily_avg) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(daily_avg.index, daily_avg.values, marker='o', linewidth=2, markersize=8, color='#667eea')
            ax.fill_between(daily_avg.index, daily_avg.values, alpha=0.3, color='#667eea')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Average Risk Probability (%)', fontsize=12)
            ax.set_title('Diabetes Risk Trend Over Time', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close()
        else:
            st.info("Need more predictions to show trend")
        
        # Gender-based Analysis
        st.subheader("👥 Gender-Based Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Gender distribution of predictions
            gender_counts = all_records.groupby('gender').size()
            if len(gender_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = ['#667eea', '#f472b6', '#94a3b8']
                ax.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', colors=colors[:len(gender_counts)])
                ax.set_title('Predictions by Gender', fontsize=12, fontweight='bold')
                st.pyplot(fig)
                plt.close()
        
        with col2:
            # Risk by gender
            gender_risk = all_records.groupby('gender')['probability'].mean() * 100
            if len(gender_risk) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                bars = ax.bar(gender_risk.index, gender_risk.values, color=['#667eea', '#f472b6', '#94a3b8'])
                ax.set_xlabel('Gender', fontsize=12)
                ax.set_ylabel('Average Risk (%)', fontsize=12)
                ax.set_title('Average Diabetes Risk by Gender', fontsize=12, fontweight='bold')
                ax.set_ylim(0, 100)
                for bar, value in zip(bars, gender_risk.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{value:.1f}%', ha='center', va='bottom')
                st.pyplot(fig)
                plt.close()
        
        # Risk Distribution
        st.subheader("📊 Risk Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        all_records['risk_category'] = pd.cut(all_records['probability'] * 100, 
                                               bins=[0, 30, 50, 70, 100],
                                               labels=['Low Risk (<30%)', 'Moderate (30-50%)', 
                                                      'High (50-70%)', 'Very High (>70%)'])
        risk_counts = all_records['risk_category'].value_counts()
        colors = ['#10b981', '#f59e0b', '#f97316', '#ef4444']
        bars = ax.bar(risk_counts.index, risk_counts.values, color=colors[:len(risk_counts)])
        ax.set_xlabel('Risk Category', fontsize=12)
        ax.set_ylabel('Number of Predictions', fontsize=12)
        ax.set_title('Distribution of Risk Levels', fontsize=14, fontweight='bold')
        for bar, count in zip(bars, risk_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        st.pyplot(fig)
        plt.close()
        
        # Patient-specific analytics
        st.subheader("👤 Patient-Specific Analytics")
        selected_patient_for_analytics = st.selectbox(
            "Select patient for detailed analysis",
            all_records['patient_name'].unique() if 'patient_name' in all_records.columns else []
        )
        
        if selected_patient_for_analytics:
            patient_data = all_records[all_records['patient_name'] == selected_patient_for_analytics]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Patient risk trend
                fig, ax = plt.subplots(figsize=(10, 6))
                patient_data_sorted = patient_data.sort_values('record_date')
                ax.plot(range(len(patient_data_sorted)), patient_data_sorted['probability'] * 100, 
                       marker='o', linewidth=2, markersize=8, color='#667eea')
                ax.set_xlabel('Visit Number', fontsize=12)
                ax.set_ylabel('Risk Probability (%)', fontsize=12)
                ax.set_title(f'Risk Trend for {selected_patient_for_analytics}', fontsize=14, fontweight='bold')
                ax.set_ylim(0, 100)
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Key metrics for patient
                latest = patient_data.iloc[0]
                st.metric("Latest Result", latest['risk_level'])
                st.metric("Latest Glucose", f"{latest['glucose']} mg/dL")
                st.metric("Latest BMI", f"{latest['bmi']:.1f}")
                st.metric("Latest Risk", f"{latest['probability']*100:.1f}%")
        
    else:
        st.info("No predictions yet. Make a prediction to see history and analytics!")

st.markdown("---")
st.caption("🏥 Diabetes Prediction System | Powered by Machine Learning | Please consult healthcare provider for medical advice")
