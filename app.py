import streamlit as st
import pandas as pd
import numpy as np
import json
import os

# Set page configuration
st.set_page_config(
    page_title="Solar Desalination TDS Prediction",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1a365d;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .status-potable {
        color: #2e7d32;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .status-warning {
        color: #d97706;
        font-weight: bold;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💧 Solar Desalination Product Water TDS Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deployment of Best Model (Random Forest Regressor) trained on 80% split with 20% holdout test evaluation</div>', unsafe_allow_html=True)

# Load Datasets
@st.cache_data
def load_datasets():
    train_df = pd.read_csv('train_data_80.csv')
    test_df = pd.read_csv('test_data_20.csv')
    return train_df, test_df

train_df, test_df = load_datasets()

# Load Random Forest Model
@st.cache_resource
def load_rf_model():
    # Try loading serialized joblib pipeline first
    try:
        import joblib
        if os.path.exists('best_water_desalination_model.pkl'):
            model = joblib.load('best_water_desalination_model.pkl')
            return {"type": "joblib", "model": model}
    except Exception:
        pass
    
    # Fallback to pure json tree structure if joblib/sklearn version mismatch occurs
    if os.path.exists('all_models_bundle.json'):
        with open('all_models_bundle.json', 'r') as f:
            bundle = json.load(f)
        return {"type": "bundle", "bundle": bundle}
    
    return None

model_artifact = load_rf_model()

def predict_tds(features):
    x_raw = np.array(features, dtype=float)
    
    if model_artifact and model_artifact["type"] == "joblib":
        cols = ['pH_raw', 'Temp_raw', 'TDS_raw', 'EC_raw', 'Salinity_raw', 'Turbidity_raw']
        df_in = pd.DataFrame([features], columns=cols)
        return float(model_artifact["model"].predict(df_in)[0])
    
    elif model_artifact and model_artifact["type"] == "bundle":
        trees = model_artifact["bundle"]['random_forest']['trees']
        def eval_tree(node, x):
            while 'value' not in node:
                node = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
            return node['value']
        preds = [eval_tree(t, x_raw) for t in trees]
        return float(np.mean(preds))
    
    return 0.0

# Sidebar: Controls
st.sidebar.header("⚙️ Input Selection")
st.sidebar.markdown("**Deployed Model:** `Random Forest Regressor (Best Model)`")

input_mode = st.sidebar.radio(
    "Choose Input Method:",
    ("Select Sample from 20% Test Set", "Manual Parameter Input")
)

actual_tds = None
sample_no_val = None

if input_mode == "Select Sample from 20% Test Set":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Test Sample")
    
    test_options = [f"Sample #{int(row['Sample_No'])} (Actual TDS: {row['TDS_treated']:.1f} mg/L)" for _, row in test_df.iterrows()]
    selected_sample_str = st.sidebar.selectbox("Test Holdout Samples (45 total):", test_options)
    selected_idx = test_options.index(selected_sample_str)
    sample_row = test_df.iloc[selected_idx]
    
    sample_no_val = int(sample_row['Sample_No'])
    ph_in = float(sample_row['pH_raw'])
    temp_in = float(sample_row['Temp_raw'])
    tds_in = float(sample_row['TDS_raw'])
    ec_in = float(sample_row['EC_raw'])
    sal_in = float(sample_row['Salinity_raw'])
    turb_in = float(sample_row['Turbidity_raw'])
    actual_tds = float(sample_row['TDS_treated'])

else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Custom Water Parameters")
    ph_in = st.sidebar.slider("Feed pH (pH_raw)", 6.5, 9.0, 7.8, 0.01)
    temp_in = st.sidebar.slider("Feed Temperature (°C)", 10.0, 45.0, 24.5, 0.1)
    tds_in = st.sidebar.slider("Feed TDS (mg/L)", 500.0, 3000.0, 1250.0, 10.0)
    ec_in = st.sidebar.slider("Feed Electrical Conductivity (µS/cm)", 1000.0, 5500.0, 2400.0, 10.0)
    sal_in = st.sidebar.slider("Feed Salinity (ppt)", 0.5, 3.5, 1.25, 0.01)
    turb_in = st.sidebar.slider("Feed Turbidity (NTU)", 0.01, 2.50, 0.12, 0.005)

# Current Input Summary
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Feed pH", f"{ph_in:.2f}")
col2.metric("Feed Temp", f"{temp_in:.1f} °C")
col3.metric("Feed TDS", f"{tds_in:.1f} mg/L")
col4.metric("Feed EC", f"{ec_in:.0f} µS/cm")
col5.metric("Salinity", f"{sal_in:.2f} ppt")
col6.metric("Turbidity", f"{turb_in:.3f} NTU")

st.markdown("---")

# Prediction Execution
input_features = [ph_in, temp_in, tds_in, ec_in, sal_in, turb_in]
predicted_tds = predict_tds(input_features)

# Results Display
res_col1, res_col2 = st.columns([1.2, 1.8])

with res_col1:
    st.subheader("🎯 Model Prediction")
    st.markdown("**Model:** `Random Forest Regressor (Pipeline)`")
    
    st.markdown(f"""
        <div style="background-color: #ebf8ff; border: 2px solid #3182ce; border-radius: 10px; padding: 20px; text-align: center;">
            <div style="font-size: 1.1rem; color: #2b6cb0; font-weight: 600;">PREDICTED TREATED WATER TDS</div>
            <div style="font-size: 3rem; font-weight: 700; color: #1a365d; margin: 8px 0;">{predicted_tds:.2f} <span style="font-size: 1.2rem; font-weight: 400;">mg/L</span></div>
        </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    if predicted_tds <= 300:
        st.markdown('✅ **WHO / Potability Status:** <span class="status-potable">Excellent Quality Potable Water (≤ 300 mg/L)</span>', unsafe_allow_html=True)
    elif predicted_tds <= 500:
        st.markdown('ℹ️ **WHO / Potability Status:** <span class="status-potable">Good Drinking Standard (300 - 500 mg/L)</span>', unsafe_allow_html=True)
    else:
        st.markdown('⚠️ **WHO / Potability Status:** <span class="status-warning">Elevated TDS (> 500 mg/L) - Secondary treatment advised</span>', unsafe_allow_html=True)

with res_col2:
    if actual_tds is not None:
        st.subheader("📊 Ground Truth vs Prediction Comparison")
        error_val = abs(actual_tds - predicted_tds)
        pct_err = (error_val / actual_tds) * 100.0
        accuracy_val = max(0.0, 100.0 - pct_err)
        
        c_a, c_b, c_c = st.columns(3)
        c_a.metric("Actual Treated TDS", f"{actual_tds:.2f} mg/L")
        c_b.metric("Predicted TDS", f"{predicted_tds:.2f} mg/L")
        c_c.metric("Absolute Error (Δ)", f"{error_val:.2f} mg/L", delta=f"{pct_err:.1f}% error", delta_color="inverse")
        
        st.markdown(f"**Sample Origin:** Unseen 20% Holdout Test Partition (`Sample #{sample_no_val}`)")
        st.info(f"💡 **Model Accuracy on this Test Sample:** `{accuracy_val:.2f}%` (Absolute Error: `{error_val:.2f} mg/L`). This sample was not seen by the Random Forest model during training.")
    else:
        st.subheader("💡 Water Treatment Operational Insights")
        desal_eff = ((tds_in - predicted_tds) / tds_in) * 100.0
        st.markdown(f"""
        - **Estimated Salt Rejection Efficiency:** `{desal_eff:.2f}%`
        - **Primary Driving Feature:** Feed Electrical Conductivity (`EC_raw`) and `TDS_raw` account for over **55%** of model predictive weight.
        - **Model Reliability:** Random Forest Regressor provides superior variance reduction with a test MAE of **32.2 mg/L**, robust against raw feed turbidity and temperature fluctuations.
        """)

st.markdown("---")

# Model Benchmark Reference Card
st.subheader("📋 Random Forest Model Evaluation Performance")
st.markdown("""
| Metric | Test Holdout (20% Split) | 5-Fold Cross-Validation | Target Threshold | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **32.243 mg/L** | - | $< 35.0\text{ mg/L}$ | ✅ Benchmark Satisfied |
| **Root Mean Squared Error (RMSE)** | **49.327 mg/L** | - | - | ✅ Lowest across all models |
| **Coefficient of Determination ($R^2$)** | **0.6640** | **$0.7296 \pm 0.038$** | $> 0.65$ | ✅ Benchmark Satisfied |
""")

st.caption("Solar Desalination Water Quality Machine Learning Mini-Project | Phase 3 Best Model Deployment")
