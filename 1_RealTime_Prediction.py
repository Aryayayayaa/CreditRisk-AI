import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import joblib
import time

# Set MLflow tracking URI (must match your running MLflow server)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
st.set_page_config(layout="wide")
st.title("⚡ Real-Time Credit Prediction")
st.markdown("Input applicant details below to receive the dual prediction (Eligibility & Max EMI).")
st.markdown("---")


# --- Utility Functions ---

@st.cache_resource
def load_mlflow_models():
    """Loads the latest registered models (Classifier and Regressor) from MLflow."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # 1. Load Classifier
        classifier_uri = "models:/XGBoost_Classifier/latest"
        classifier_model = mlflow.pyfunc.load_model(classifier_uri)
        
        # 2. Load Regressor
        regressor_uri = "models:/XGBoost_Regressor/latest"
        regressor_model = mlflow.pyfunc.load_model(regressor_uri)
        
        # Load the fitted scaler and OHE list used during training (Simulated here)
        # In a real app, these would also be MLflow artifacts or a custom preprocessor wrapper.
        # For simplicity, we define the required features/OHE columns explicitly.
        
        # Define the 44 features expected by the final model (after OHE)
        # (This is a simplified version of the features derived from your project)
        features_list = [
            'age', 'monthly_salary', 'bank_balance', 'monthly_expenses', 'credit_score',
            'annual_income', 'loan_amount_requested', 'credit_utilization_ratio',
            'years_employed', 'is_log_emi_scenario', 'credit_score_zero_flag',
            'employment_type_Contract', 'employment_type_Full-Time', 'employment_type_Part-Time',
            'employment_type_Self-Employed', 'loan_purpose_Business', 'loan_purpose_Education',
            'loan_purpose_Home_Improvement', 'loan_purpose_Other', 'loan_purpose_Vehicle',
            'marital_status_Divorced', 'marital_status_Married', 'marital_status_Single',
            'property_ownership_None', 'property_ownership_Owned', 'property_ownership_Rented',
            'education_High_School', 'education_Masters', 'education_Other', 'education_PHD',
            'industry_Finance', 'industry_Healthcare', 'industry_IT', 'industry_Manufacturing',
            'industry_Retail', 'industry_Other', 'residence_type_Apartment',
            'residence_type_House', 'residence_type_Other',
            # Add placeholders for transformed features if used by the model directly (e.g., log-transformed values)
            # Since the model expects log(monthly_salary), we must transform the input.
        ]
        
        return classifier_model, regressor_model, features_list
    
    except Exception as e:
        st.error(f"Failed to load models from MLflow. Ensure MLflow server is running at {MLFLOW_TRACKING_URI}. Error: {e}")
        return None, None, None

def predict(data, classifier, regressor, features_list):
    """
    Performs feature engineering on input data and generates dual predictions.
    """
    # 1. Prepare data for model input
    df = pd.DataFrame([data])
    
    # Apply Log Transformation (Crucial step from preprocessing)
    # The Regressor was trained on log-transformed EMI, so we assume log transformations
    # were also applied to highly skewed input features like monthly_salary, bank_balance, etc.
    # We will only apply the inverse log for the final EMI output.
    
    # 2. One-Hot Encoding (OHE) for categorical features
    
    # Define categorical columns from your training set
    categorical_cols = ['employment_type', 'loan_purpose', 'marital_status', 'property_ownership', 'education', 'industry', 'residence_type']
    
    # Apply OHE to input and reindex to match the 44 features
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    # Add dummy columns that might be missing in a single applicant's input
    final_features = []
    
    # Numerical/Flag features that should always be present (excluding the OHE keys)
    numerical_features = ['age', 'monthly_salary', 'bank_balance', 'monthly_expenses', 'credit_score',
                         'annual_income', 'loan_amount_requested', 'credit_utilization_ratio',
                         'years_employed']
    
    # Simplified flags/derived features that must be created
    df_encoded['is_log_emi_scenario'] = 1 # Flag to indicate log-transformed data was used for the target
    df_encoded['credit_score_zero_flag'] = (df_encoded['credit_score'] == 0).astype(int)

    # Re-align columns to match the model's 44 features
    for col in features_list:
        if col not in df_encoded.columns:
            df_encoded[col] = 0 # Add missing OHE column with value 0
        final_features.append(df_encoded[col].iloc[0])

    # Convert to NumPy array for XGBoost prediction
    input_array = np.array(final_features).reshape(1, -1)
    
    # 3. Generate Predictions
    
    # Classification Prediction
    # The XGBoost Classifier was trained on 3 classes: 0='Eligible', 1='High Risk', 2='Not Eligible' (or similar mapping)
    class_mapping = {0: 'Eligible', 1: 'High Risk', 2: 'Not Eligible'} # Assuming this mapping
    
    classification_proba = classifier.predict_proba(input_array)[0]
    classification_index = np.argmax(classification_proba)
    eligibility_status = class_mapping.get(classification_index, 'Unknown')
    confidence = classification_proba[classification_index] * 100
    
    # Regression Prediction
    log_emi_prediction = regressor.predict(input_array)[0]
    
    # Inverse Log Transformation (Crucial step to get back to INR)
    # The formula is typically exp(prediction) - 1 or expm1(prediction)
    max_emi_amount = np.expm1(log_emi_prediction)
    
    # Format EMI to nearest Rupee
    max_emi_amount = max_emi_amount.round(0)
    
    return eligibility_status, confidence, max_emi_amount

# --- Streamlit UI Components ---

# Load models once
classifier_model, regressor_model, features_list = load_mlflow_models()

if classifier_model and regressor_model:
    
    col1, col2 = st.columns(2)
    
    # Input Data Collection (Simplified to key variables)
    with col1:
        st.header("Applicant Financials")
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        monthly_salary = st.number_input("Monthly Salary (INR)", min_value=5000.0, value=50000.0, step=1000.0)
        bank_balance = st.number_input("Bank Balance (INR)", min_value=0.0, value=150000.0, step=5000.0)
        monthly_expenses = st.number_input("Monthly Expenses (INR)", min_value=0.0, value=20000.0, step=500.0)
        credit_score = st.number_input("Credit Score", min_value=0, max_value=900, value=750)
        loan_amount_requested = st.number_input("Loan Amount Requested (INR)", min_value=1000.0, value=500000.0, step=10000.0)
        years_employed = st.number_input("Years Employed", min_value=0.0, value=5.0, step=0.5)

    with col2:
        st.header("Applicant Profile & Needs")
        employment_type = st.selectbox("Employment Type", ['Full-Time', 'Part-Time', 'Self-Employed', 'Contract'])
        loan_purpose = st.selectbox("Loan Purpose", ['Home_Improvement', 'Vehicle', 'Education', 'Business', 'Other'])
        marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
        property_ownership = st.selectbox("Property Ownership", ['Owned', 'Rented', 'None'])
        education = st.selectbox("Education Level", ['Masters', 'High_School', 'PHD', 'Other'])
        industry = st.selectbox("Industry", ['IT', 'Finance', 'Healthcare', 'Manufacturing', 'Retail', 'Other'])
        residence_type = st.selectbox("Residence Type", ['House', 'Apartment', 'Other'])

    # Prepare input dictionary for the prediction function
    input_data = {
        'age': age,
        'monthly_salary': monthly_salary,
        'bank_balance': bank_balance,
        'monthly_expenses': monthly_expenses,
        'credit_score': credit_score,
        'loan_amount_requested': loan_amount_requested,
        'years_employed': years_employed,
        'annual_income': monthly_salary * 12, # Derived feature
        'credit_utilization_ratio': loan_amount_requested / (monthly_salary * 30), # Mock derived feature
        'employment_type': employment_type,
        'loan_purpose': loan_purpose,
        'marital_status': marital_status,
        'property_ownership': property_ownership,
        'education': education,
        'industry': industry,
        'residence_type': residence_type,
    }

    st.markdown("---")
    if st.button("Generate Credit Decision", key="predict_btn"):
        with st.spinner("Analyzing applicant data and retrieving models from MLflow..."):
            time.sleep(1) # Simulated network latency
            
            # Run prediction
            status, confidence, emi_amount = predict(input_data, classifier_model, regressor_model, features_list)

        # --- Display Results ---
        st.subheader("Final Credit Decision Summary")
        
        # Color the status based on risk
        if status == 'Eligible':
            status_color = 'green'
        elif status == 'High Risk':
            status_color = 'orange'
        else:
            status_color = 'red'
            
        col3, col4 = st.columns(2)
        
        # Classification Output
        with col3:
            st.markdown(f"### Eligibility Status:")
            st.markdown(f"<p class='big-font' style='color:{status_color};'>**{status}**</p>", unsafe_allow_html=True)
            st.metric("Confidence Score", f"{confidence:.2f}%")
            
        # Regression Output
        with col4:
            st.markdown(f"### Max Monthly EMI Allowed:")
            st.markdown(f"<p class='big-font' style='color:#1E90FF;'>**₹ {emi_amount:,.0f}**</p>", unsafe_allow_html=True)
            st.caption("This is the maximum sustainable EMI based on predictive modeling.")
        
        if status == 'Eligible':
             st.success(f"Applicant is **{status}** with high confidence. Proceed with a loan offering up to a monthly EMI of **₹ {emi_amount:,.0f}**.")
        elif status == 'High Risk':
             st.warning(f"Applicant is classified as **{status}**. Recommend mandatory secondary review or offer a reduced loan (EMI: **₹ {emi_amount:,.0f}**).")
        else:
             st.error(f"Applicant is **{status}**. Loan application is **rejected** based on risk model assessment.")
        
        st.markdown("---")
        st.info(f"Prediction generated by Model **XGBoost Classifier** (version: latest) and **XGBoost Regressor** (version: latest) from MLflow.")

else:
    st.warning("Prediction interface offline. Please resolve MLflow connection issues.")