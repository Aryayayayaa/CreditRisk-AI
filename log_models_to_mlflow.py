import pandas as pd
import numpy as np
import mlflow
import os
import xgboost as xgb
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient 

# --- 0. Configuration and Setup ---

# Set the experiment name for better organization in MLflow UI
MLFLOW_EXPERIMENT_NAME = "Loan_Risk_Assessment_Models"
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# These MUST exactly match the names requested by the Streamlit application:
CLASSIFIER_MODEL_NAME = "XGBoost_Classifier" 
REGRESSOR_MODEL_NAME = "XGBoost_Regressor"   

CLASSIFIER_FILE = 'classifier.json' 
REGRESSOR_FILE = 'regressor.json' 

# The required 44 features for both models (MUST MATCH TRAINING FEATURES)
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


# --- 1. Model Loading (Direct Native Load) ---
def load_booster_from_native(filepath):
    """Loads a pure XGBoost Booster object directly from a native JSON file."""
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: New stable model file not found: {filepath}")
        return None
    
    print(f"Loading stable pure Booster from: {filepath}")
    
    # 1. Create a PURE XGBoost Booster object 
    booster = xgb.Booster() 
    
    # 2. Load the state directly
    try:
        booster.load_model(filepath)
    except Exception as e:
        print(f"Error loading model {filepath}: {e}")
        return None
        
    print(f"✅ Successfully loaded PURE Booster from {filepath}")
    return booster

classifier_booster = load_booster_from_native(CLASSIFIER_FILE)
regressor_booster = load_booster_from_native(REGRESSOR_FILE)

if classifier_booster is None or regressor_booster is None:
    print("\nCannot proceed with MLflow logging. Exiting.")
    exit()

# --- 2. Create Dummy Data for Signature Generation ---
print("\nCreating dummy data and DMatrix for MLflow Model Signature...")

# Dummy input DataFrame (needs to match the 44 feature structure exactly)
dummy_X = pd.DataFrame(np.random.rand(5, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
for col in FEATURE_COLUMNS:
    # Ensure OHE columns are boolean/binary types
    if any(s in col for s in ['gender_', 'marital_status_', 'education_', 'employment_type_', 'company_type_', 'house_type_', 'existing_loans_', 'emi_scenario_', 'credit_score_tier_', 'employment_tier_']):
        # Set to boolean based on a threshold
        dummy_X[col] = (dummy_X[col] > 0.5).astype(bool) 

# Create DMatrix objects (Booster requires DMatrix for prediction)
dummy_DMatrix = xgb.DMatrix(dummy_X[FEATURE_COLUMNS])


# --- 3. MLflow Logging Process ---

# Initialize MLflow Client outside the run for stage management
client = MlflowClient()

# We use the underscore (_) to ignore the problematic return value
with mlflow.start_run() as run:
    run_id = run.info.run_id
    print(f"\nMLflow Run started with ID: {run_id}")
    
    # --- A. Log Classifier Model (Eligibility) ---
    classifier_predictions = classifier_booster.predict(dummy_DMatrix)
    final_classifier_preds = np.argmax(classifier_predictions, axis=1) 
    classifier_signature = infer_signature(dummy_X, final_classifier_preds)
    
    # Log and register the model (return value is ignored for robustness)
    _ = mlflow.xgboost.log_model(
        xgb_model=classifier_booster, 
        artifact_path="eligibility_classifier_booster",
        signature=classifier_signature,
        registered_model_name=CLASSIFIER_MODEL_NAME, 
        conda_env=None,
        metadata={
            "model_type": "Classification (Pure Booster, JSON)",
            "target_classes": "0:Eligible, 1:High_Risk, 2:Not_Eligible",
            "note": "Logged using stable native JSON file format."
        }
    )
    print(f"✅ Classifier (Booster) logged successfully. Model Name: {CLASSIFIER_MODEL_NAME}")
    
    # --- B. Log Regressor Model (Max EMI) ---
    regressor_predictions = regressor_booster.predict(dummy_DMatrix)
    regressor_signature = infer_signature(dummy_X, regressor_predictions)
    
    # Log and register the model (return value is ignored for robustness)
    _ = mlflow.xgboost.log_model(
        xgb_model=regressor_booster, 
        artifact_path="max_emi_regressor_booster",
        signature=regressor_signature,
        registered_model_name=REGRESSOR_MODEL_NAME, 
        conda_env=None,
        metadata={
            "model_type": "Regression (Pure Booster, JSON)",
            "target_scale": "Log-Transformed INR",
            "note": "Logged using stable native JSON file format."
        }
    )
    print(f"✅ Regressor (Booster) logged successfully. Model Name: {REGRESSOR_MODEL_NAME}")
    
    # --- C. Transition to Production Stage (Reliable Method) ---
    
    # 1. Transition Classifier to Production
    # Get the latest version object for the Classifier model
    # [0] safely retrieves the first (and newest) version created.
    latest_classifier_version = client.get_latest_versions(CLASSIFIER_MODEL_NAME, stages=['None', 'Staging', 'Production'])[0]
    latest_classifier_version_number = latest_classifier_version.version

    client.transition_model_version_stage(
        name=CLASSIFIER_MODEL_NAME,
        version=latest_classifier_version_number,
        stage="Production"
    )
    print(f"🔄 Classifier (v{latest_classifier_version_number}) transitioned to Production.")
    
    # 2. Transition Regressor to Production
    # Get the latest version object for the Regressor model
    latest_regressor_version = client.get_latest_versions(REGRESSOR_MODEL_NAME, stages=['None', 'Staging', 'Production'])[0]
    latest_regressor_version_number = latest_regressor_version.version

    client.transition_model_version_stage(
        name=REGRESSOR_MODEL_NAME,
        version=latest_regressor_version_number,
        stage="Production"
    )
    print(f"🔄 Regressor (v{latest_regressor_version_number}) transitioned to Production.")

# --- 4. Conclusion ---
print("\n--- MLflow Logging and Registration Complete ---")
print("The models are now registered and set to Production stage.")
