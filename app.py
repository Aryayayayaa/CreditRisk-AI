import streamlit as st
import base64 # Required for Base64 encoding

# Set the overall page configuration
st.set_page_config(
    page_title="Credit Decision System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Base64 Background Image Function ---
def get_base64_of_bin_file(bin_file):
    """Encodes a local binary file (image) to a Base64 string."""
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        st.error(f"Error: The background image file '{bin_file}' was not found. Please ensure it is in the correct directory.")
        return None

def set_background(image_path):
    """Sets the background image using Base64 encoded data URI."""
    base64_data = get_base64_of_bin_file(image_path)
    if base64_data:
        # Use image/jpeg MIME type since the file is .jpg
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{base64_data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* Ensure main content is readable over the background */
        .main {{
            /* Add a semi-transparent white background to the main content area */
            background-color: rgba(255, 255, 255, 0.85); 
            padding: 1rem;
            border-radius: 10px;
        }}
        
        /* General Streamlit component styling */
        .big-font {{
            font-size:30px !important;
            font-weight: bold;
        }}
        .sidebar .sidebar-content {{
            background-color: #ffffff;
        }}
        .stButton>button {{
            font-weight: bold;
            border-radius: 8px;
            color: white;
            background-color: #4CAF50;
            border: none;
            padding: 10px 24px;
            transition: 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #45a049;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
        
# Call the function to set the background
set_background('bg.jpg')


st.title("🏦 CreditRisk AI: Risk-Agnostic Lending Platform (MLOps Deployment)")
st.markdown("---")

st.markdown("""
This multi-page application serves as the production interface for the credit decision system. It uses **MLflow** to pull the latest **Tuned XGBoost** Classification and Regression models for real-time risk assessment and Max EMI calculation.

Use the sidebar navigation to access different application features:
- **Real-Time Prediction:** Input applicant data and get the final credit decision.
- **Model Dashboard:** Review logged performance metrics and the MLflow Model Registry status.
- **Admin Interface:** Simulate data management and retraining operations.
""")

st.info("Please ensure your MLflow Tracking Server is running at http://127.0.0.1:5000.")

# Display a high-level summary of the final models
st.sidebar.header("Final Model Summary")
st.sidebar.metric("Classifier Accuracy", "95.32%")
st.sidebar.metric("Regressor R-squared", "0.9845")
st.sidebar.markdown("**Framework:** XGBoost (Tuned)")

# Custom footer (Streamlit uses the file structure for routing, so this file acts as the welcome page)
st.markdown("---")
st.caption("Developed for Production Readiness using MLflow and Streamlit.")