import streamlit as st
import pandas as pd
import mlflow
from mlflow.entities import ViewType

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
st.set_page_config(layout="wide")
st.title("📊 Model Performance Monitoring & MLflow Dashboard")
st.markdown("Review the performance of the final selected models and their history in the MLflow Registry.")
st.markdown("---")

@st.cache_data
def get_mlflow_model_data():
    """Fetches key performance metrics from the MLflow Tracking Server."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        
        # Get latest version for each model
        classifier_versions = client.get_latest_versions("XGBoost_Classifier", stages=["Production"])
        regressor_versions = client.get_latest_versions("XGBoost_Regressor", stages=["Production"])
        
        # --- Classification Metrics ---
        if classifier_versions:
            c_version = classifier_versions[0]
            c_run_id = c_version.run_id
            c_metrics = client.get_run(c_run_id).data.metrics
            c_params = client.get_run(c_run_id).data.params
            c_data = {
                "Model Name": "XGBoost Classifier",
                "MLflow Version": c_version.version,
                "Stage": c_version.current_stage,
                "Accuracy (Test)": f"{c_metrics.get('test_accuracy', 0.0) * 100:.2f}%",
                "ROC-AUC (Macro)": f"{c_metrics.get('test_roc_auc_macro', 0.0):.4f}",
                "High Risk F1": f"{c_metrics.get('test_f1_score_1', 0.0):.4f}", # Assuming 1 is High Risk
                "n_estimators": c_params.get('n_estimators', 'N/A'),
                "max_depth": c_params.get('max_depth', 'N/A'),
            }
        else:
            c_data = {"Model Name": "XGBoost Classifier", "Status": "Not Found"}
            
        # --- Regression Metrics ---
        if regressor_versions:
            r_version = regressor_versions[0]
            r_run_id = r_version.run_id
            r_metrics = client.get_run(r_run_id).data.metrics
            r_params = client.get_run(r_run_id).data.params
            r_data = {
                "Model Name": "XGBoost Regressor",
                "MLflow Version": r_version.version,
                "Stage": r_version.current_stage,
                "R-squared (Test)": f"{r_metrics.get('test_r2', 0.0):.4f}",
                "RMSE (Test)": f"₹{r_metrics.get('test_rmse', 0.0):,.0f}",
                "MAPE (Test)": f"{r_metrics.get('test_mape', 0.0):.2f}%",
                "n_estimators": r_params.get('n_estimators', 'N/A'),
                "learning_rate": r_params.get('learning_rate', 'N/A'),
            }
        else:
            r_data = {"Model Name": "XGBoost Regressor", "Status": "Not Found"}
            
        return c_data, r_data
        
    except Exception as e:
        st.error(f"MLflow Connection Error: Could not connect to the tracking server at {MLFLOW_TRACKING_URI}. {e}")
        return {"Status": "Error"}, {"Status": "Error"}

# Fetch and display data
c_data, r_data = get_mlflow_model_data()

st.subheader("Model Registry Status (Production Stage)")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Classifier Performance")
    if c_data.get("Status") == "Error":
        st.error("Classifier data unavailable.")
    else:
        st.table(pd.DataFrame(c_data, index=["Value"]).T.style.set_properties(**{'font-size': '10pt'}))
        
with col2:
    st.markdown("### Regressor Performance")
    if r_data.get("Status") == "Error":
        st.error("Regressor data unavailable.")
    else:
        st.table(pd.DataFrame(r_data, index=["Value"]).T.style.set_properties(**{'font-size': '10pt'}))

st.markdown("---")
st.subheader("Performance Visualization Placeholder")
st.markdown("In a full system, this area would display live charts, such as the Confusion Matrix for the Classifier and an Actual vs. Predicted plot for the Regressor.")

# Mock Data for Visualization
chart_data = pd.DataFrame({
    'Metric': ['Accuracy', 'ROC-AUC', 'F1-Score'],
    'Value': [0.9532, 0.9583, 0.90] 
})
st.bar_chart(chart_data.set_index('Metric'))

st.caption(f"MLflow Tracking Server URL: {MLFLOW_TRACKING_URI}")