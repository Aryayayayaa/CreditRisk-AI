import pandas as pd
import numpy as np
import xgboost as xgb
import os

# --- 0. Configuration ---
CLASSIFIER_OUTPUT = 'classifier.json'
REGRESSOR_OUTPUT = 'regressor.json'

# The required 44 features (MUST MATCH the input features for your project)
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

# --- 1. Create Dummy Data ---
print("Creating dummy dataset (5 samples) for model fitting...")
# Dummy input DataFrame (needs to match the 44 feature structure exactly)
X_dummy = pd.DataFrame(np.random.rand(5, len(FEATURE_COLUMNS)), columns=FEATURE_COLUMNS)
for col in FEATURE_COLUMNS:
    # Ensure OHE columns are boolean/binary types
    if any(s in col for s in ['gender_', 'marital_status_', 'education_', 'employment_type_', 'company_type_', 'house_type_', 'existing_loans_', 'emi_scenario_', 'credit_score_tier_', 'employment_tier_']):
        X_dummy[col] = (X_dummy[col] > 0.5).astype(bool) 

# Dummy target for Classifier (3 classes: 0, 1, 2)
y_cls = np.random.randint(0, 3, size=5)
# Dummy target for Regressor (continuous log-transformed value)
y_reg = np.random.rand(5) * 5

# --- 2. Create and Save Classifier Model ---
print("Creating and fitting dummy Classifier...")
# We must use the scikit-learn wrapper to automatically handle feature names
classifier = xgb.XGBClassifier(
    objective='multi:softprob', 
    num_class=3, 
    use_label_encoder=False, 
    eval_metric='mlogloss',
    n_estimators=10 # Use a small number of estimators for speed
)
classifier.fit(X_dummy, y_cls)

# Save the model using the stable native JSON format
classifier.save_model(CLASSIFIER_OUTPUT)
print(f"✅ Dummy Classifier saved successfully to: {CLASSIFIER_OUTPUT}")


# --- 3. Create and Save Regressor Model ---
print("Creating and fitting dummy Regressor...")
regressor = xgb.XGBRegressor(
    objective='reg:squarederror', 
    n_estimators=10
)
regressor.fit(X_dummy, y_reg)

# Save the model using the stable native JSON format
regressor.save_model(REGRESSOR_OUTPUT)
print(f"✅ Dummy Regressor saved successfully to: {REGRESSOR_OUTPUT}")

print("\n--- Model Creation Complete ---")
print("You now have two stable files: classifier.json and regressor.json.")
print("Proceed to run the updated MLflow logging script.")