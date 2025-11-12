# CreditRisk-AI
MLOps project for real-time Credit Risk assessment. Uses XGBoost models, MLflow for versioning and deployment, and a Streamlit frontend to predict loan eligibility and maximum EMI.

---

💰 CreditRisk AI: MLOps for Loan Risk AssessmentThis project demonstrates a production-ready MLOps pipeline for real-time Loan Risk Assessment. It uses two interconnected XGBoost models—a Classifier and a Regressor—to determine customer eligibility and calculate the maximum affordable loan installment (EMI).The entire workflow, from model registration to live inference, is managed using MLflow and presented through an interactive Streamlit dashboard.🎯 Project GoalThe primary objective is to create a robust and scalable system where:Models are versioned and governed (MLflow Model Registry).Inference is reliable (Streamlit loads models by name and stage).Two complex models (Classification for eligibility, Regression for affordability) work together seamlessly to provide a final lending decision.🛠️ Technology StackComponentTechnologyPurposeMachine LearningXGBoostHigh-performance models for eligibility and maximum EMI prediction.MLOps & RegistryMLflowTracks experiments, registers models, and manages model versions/stages.Frontend/DeploymentStreamlitInteractive web application for real-time inference and user input.LanguagePythonPrimary development language.Data StorageSQLiteUsed by MLflow for the backend tracking store.🚀 Getting StartedFollow these steps to set up the environment, register the models, and run the Streamlit application.PrerequisitesYou need Python (3.9+) and pip.1. Environment SetupCreate and activate a virtual environment, then install all necessary dependencies.# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# OR
.\venv\Scripts\activate   # On Windows (PowerShell)

# Install dependencies (assuming you have a requirements.txt, or install them directly)
pip install mlflow streamlit xgboost pandas numpy
2. Start the MLflow Tracking ServerThe MLflow server must be running in the background to handle model registration and serving. This command uses SQLite for simplicity and stores artifacts locally in the ./mlruns directory.Open a dedicated Terminal Window (Terminal 1) and run:mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
Keep this window open and running throughout the process.3. Register and Stage the ModelsUse the provided script (log_models_to_mlflow.py) to load the pre-trained XGBoost models (classifier.json and regressor.json) and register them in the MLflow Model Registry under the names required by the Streamlit app: XGBoost_Classifier and XGBoost_Regressor.The script also transitions the newly registered versions to the 'Production' stage.Open a second Terminal Window (Terminal 2) and run:python log_models_to_mlflow.py
You should see output confirming both models are registered and transitioned to Production.4. Run the Streamlit ApplicationThe Streamlit application (app.py) will automatically connect to the MLflow server on http://127.0.0.1:5000 to load the Production models.Use the second Terminal Window (Terminal 2) and run:streamlit run app.py
A web browser will automatically open, displaying the interactive Loan Risk Assessment dashboard. You can now input customer parameters and receive real-time lending decisions.📁 Repository Structure.
├── mlruns/                        # MLflow tracking and artifact store (created after step 2)
├── classifier.json               # Pre-trained XGBoost Classification model (Eligibility)
├── regressor.json                # Pre-trained XGBoost Regression model (Max EMI)
├── log_models_to_mlflow.py       # Script to register models with MLflow
├── app.py                        # Streamlit web application for inference
└── README.md                     # This file
📝 UsageThe Streamlit application allows users to simulate a loan application process by providing key financial and personal data.Input Features: Enter required information (e.g., salary, bank balance, requested amount, credit history).Eligibility Check: The XGBoost_Classifier (Production stage) determines if the applicant is:EligibleHigh RiskNot EligibleAffordability Calculation: If eligible, the XGBoost_Regressor (Production stage) calculates the maximum affordable monthly installment (EMI), which guides the final loan offer.
