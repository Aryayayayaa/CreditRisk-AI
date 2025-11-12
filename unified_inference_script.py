import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
import os
import json

# --- 0. Configuration and Setup ---

# The names of the models registered in the previous step
CLASSIFIER_MODEL_NAME = "XGBoostEligibilityClassifier"
REGRESSOR_MODEL_NAME = "XGBoostMaxEMIRegressor"

# The required 44 features (MUST MATCH TRAINING FEATURES)
FEATURE_COLUMNS = [
    'age', 'years_of_employment', 'family_size', 'school_fees', 'college_fees', 
    'current_emi_amount', 'requested_tenure', 'credit_score_zero_flag', 
    'total_monthly_expenses', 'DTI_ratio', 'ETI_ratio', 'LIR_ratio', 
    'log_monthly_salary', 'log_bank_balance', 'log_requested_amount', 
    'log_disposable_income', 'log_income_tenure_score', 'log_credit_liquidity_score', 
    'log_monthly_rent', 'gender_Male', 'marital_status_Single', 'education_High School', 
    'education_Post Graduate', 'education_Professional', 'education_Unknown', 
    'employment_type_Private', 'employment_type_Self-employed', 'company_type_MNC', 
    'company_type_Mid-size', 'company_type_Small', 'company_type_Startup', 
    'house_type_Own', 'house_type_Rented', 'existing_loans_Yes', 
    'emi_scenario_Education EMI', 'emi_scenario_Home Appliances EMI', 
    'emi_scenario_Personal Loan EMI', 'emi_scenario_Vehicle EMI', 'credit_score_tier_Fair', 
    'credit_score_tier_Good', 'credit_score_tier_Poor', 'employment_tier_Mid', 
    'employment_tier_New', 'employment_tier_Veteran'
]

# Mapping for Eligibility Classes
ELIGIBILITY_CLASSES = {
    0: "Eligible (Low Risk)",
    1: "High Risk (Requires Review)",
    2: "Not Eligible (Denied)"
}

# --- 1. Define Dummy Input Data for Inference ---
# This dictionary represents a single, new loan application *after* feature engineering.
# NOTE: The values here are arbitrary for demonstration but MUST match the feature set.
new_applicant_data = {
    # Numerical/Log features
    'age': 35.0, 
    'years_of_employment': 10.0, 
    'family_size': 3.0, 
    'school_fees': 0.0, 
    'college_fees': 0.0, 
    'current_emi_amount': 5000.0, 
    'requested_tenure': 24.0, 
    'credit_score_zero_flag': 0.0, 
    'total_monthly_expenses': 15000.0, 
    'DTI_ratio': 0.45, 
    'ETI_ratio': 0.30, 
    'LIR_ratio': 0.15, 
    'log_monthly_salary': np.log(60000.0), # Assuming 60k salary
    'log_bank_balance': np.log(100000.0), 
    'log_requested_amount': np.log(500000.0), 
    'log_disposable_income': np.log(30000.0), 
    'log_income_tenure_score': np.log(60000.0 * 10.0), 
    'log_credit_liquidity_score': np.log(100000.0 / 5000.0), 
    'log_monthly_rent': np.log(10000.0), 
    # One-Hot Encoded features (Booleans are fine for pandas)
    'gender_Male': True, 
    'marital_status_Single': False, 
    'education_High School': False, 
    'education_Post Graduate': True, 
    'education_Professional': False, 
    'education_Unknown': False, 
    'employment_type_Private': True, 
    'employment_type_Self-employed': False, 
    'company_type_MNC': True, 
    'company_type_Mid-size': False, 
    'company_type_Small': False, 
    'company_type_Startup': False, 
    'house_type_Own': False, 
    'house_type_Rented': True, 
    'existing_loans_Yes': True, 
    'emi_scenario_Education EMI': False, 
    'emi_scenario_Home Appliances EMI': False, 
    'emi_scenario_Personal Loan EMI': True, 
    'emi_scenario_Vehicle EMI': False, 
    'credit_score_tier_Fair': False, 
    'credit_score_tier_Good': True, 
    'credit_score_tier_Poor': False, 
    'employment_tier_Mid': True, 
    'employment_tier_New': False, 
    'employment_tier_Veteran': False
}

# Convert the single data point into a DataFrame
input_df = pd.DataFrame([new_applicant_data], columns=FEATURE_COLUMNS)

print("--- Loan Risk Assessment Inference Script ---")
print(f"Loading Models from MLflow (Run ID: {mlflow.active_run().info.run_id if mlflow.active_run() else 'N/A'})")

# --- 2. Load Models from MLflow Registry ---

def load_mlflow_model(model_name):
    """Loads the latest version of a registered model from MLflow."""
    try:
        # Load the model using the standard mlflow.pyfunc module for consistency
        model_uri = f"models:/{model_name}/latest"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        print(f"✅ Successfully loaded model: {model_name} (Latest)")
        return loaded_model
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load model {model_name} from MLflow. Ensure MLflow is running and models are registered.")
        print(f"Error details: {e}")
        return None

classifier_pyfunc = load_mlflow_model(CLASSIFIER_MODEL_NAME)
regressor_pyfunc = load_mlflow_model(REGRESSOR_MODEL_NAME)

if classifier_pyfunc is None or regressor_pyfunc is None:
    print("\nInference cannot proceed due to missing models. Exiting.")
    exit()

# --- 3. Perform Inference ---

print("\n--- Generating Predictions for New Applicant ---")

# A. Eligibility Prediction (Classification)
try:
    # Pyfunc models return results directly from the predict method
    eligibility_prediction_raw = classifier_pyfunc.predict(input_df)
    
    # The output is a multi-class probability array (e.g., [[0.8, 0.1, 0.1]])
    # We take the index of the highest probability
    predicted_class_index = np.argmax(eligibility_prediction_raw[0])
    predicted_class_name = ELIGIBILITY_CLASSES.get(predicted_class_index, "Unknown Class")
    
    # Extract the probability of the predicted class
    predicted_class_prob = eligibility_prediction_raw[0][predicted_class_index]

    print(f"\n[1] Eligibility Assessment (Classifier):")
    print(f"    -> Predicted Outcome (Class Index): {predicted_class_index}")
    print(f"    -> Final Decision: {predicted_class_name} (Confidence: {predicted_class_prob:.2f})")
    
except Exception as e:
    print(f"Error during classifier prediction: {e}")


# B. Max EMI Prediction (Regression)
try:
    # The output is the log-transformed Max EMI
    log_max_emi = regressor_pyfunc.predict(input_df)[0]
    
    # Inverse transform (e^y) to get the actual INR value
    actual_max_emi = np.exp(log_max_emi)

    print(f"\n[2] Max EMI Calculation (Regressor):")
    print(f"    -> Predicted Log(Max EMI): {log_max_emi:.4f}")
    print(f"    -> Maximum Allowed Monthly Installment (INR): ₹{actual_max_emi:.2f}")

except Exception as e:
    print(f"Error during regressor prediction: {e}")

print("\n--- Unified Inference Complete ---")
print("This script successfully demonstrated loading and using both models from the MLflow Model Registry.")



'''
AT THE END: the joblib files downloaded were found to be corrupted and could not be resolved till the last moment:

Script Name                      Purpose in the ML Pipeline
1. create_working_models.py      Data Preparation/Saving: Fixed the original corrupted .joblib files 
                                 by saving new, stable models using the native XGBoost .json format.
                                 
2. log_models_to_mlflow.py      Model Registry: Loaded the stable .json files and registered both the XGBoost 
                                Classifier and XGBoost Regressor into the MLflow Model Registry.
                                
3. unified_inference_script.py  Deployment Simulation: The final piece. It simulates a production environment 
                                by connecting to the MLflow Registry, downloading the latest model versions, 
                                and using them to make dual predictions for a new applicant.
'''