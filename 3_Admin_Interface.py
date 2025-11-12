import streamlit as st
import pandas as pd
import time
import base64

st.set_page_config(layout="wide")
st.title("🛠️ Administrative Interface")
st.markdown("Interface for simulating data management and model governance operations.")
st.markdown("---")

st.subheader("Data Management Operations")

# --- Data Upload Simulation ---
st.markdown("#### 1. Upload New Training Data")
uploaded_file = st.file_uploader("Choose a CSV file to add to the training corpus (Simulation)", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"File '{uploaded_file.name}' successfully uploaded. Shape: {data.shape}")
        
        if st.button("Commit Data to Data Lake", key="commit_data"):
            st.session_state['data_committed'] = True
            st.success("New data committed. **Model Retraining Recommended.**")
            
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

st.markdown("---")

# --- Model Retraining Simulation ---
st.markdown("#### 2. Model Retraining Trigger")

retrain_mode = st.radio(
    "Select Retraining Mode:",
    ('Full Retrain (From Scratch)', 'Incremental Retrain (Update Model Weights)'),
    index=0
)

if st.button("Trigger Retraining Pipeline", key="trigger_retrain"):
    with st.spinner(f"Initiating {retrain_mode} process..."):
        time.sleep(3) # Simulate pipeline run time
        
        # Simulate MLflow logging and model registration
        new_accuracy = 0.9532 + (0.001 * time.time() % 0.005) # Mock improvement
        
        st.balloons()
        st.success(f"Pipeline finished successfully! New model version registered to MLflow.")
        st.markdown(f"""
        - **New Accuracy:** **{new_accuracy:.4f}**
        - **MLflow Status:** Model promoted to Staging, ready for A/B testing.
        """)
        st.warning("Please review the new model in the **Model Dashboard** before promoting it to Production.")

st.markdown("---")

# --- Model Governance Simulation ---
st.markdown("#### 3. Model Version Management (MLflow) ")
st.info("In a real environment, this section would interface directly with the MLflow Model Registry to change stages (Staging -> Production).")

model_to_promote = st.selectbox(
    "Select Model to Promote to Production:",
    ('XGBoost_Classifier', 'XGBoost_Regressor', 'None')
)

version_to_promote = st.number_input("Enter Version Number to Promote (e.g., 2)", min_value=1, value=1)

if st.button("Promote Selected Model Version", key="promote_model"):
    if model_to_promote != 'None':
        with st.spinner(f"Promoting {model_to_promote} V{version_to_promote} to Production..."):
            time.sleep(2)
            st.success(f"**{model_to_promote} Version {version_to_promote}** successfully promoted to **Production** stage in MLflow Registry!")
            st.warning("All real-time predictions will now use this version. Verify dashboard metrics.")
    else:
        st.info("Please select a model to promote.")