import streamlit as st
import joblib
from PIL import Image
import os
import requests

# Get the absolute path to the directory where your models are stored
models_directory = os.path.join(os.getcwd(), 'models')  # Update 'models' to your actual directory name

# Ensure the models directory exists
os.makedirs(models_directory, exist_ok=True)

# URLs to the models and data on GitHub
best_model_url = 'https://github.com/anshu7345/TMLC-MegaGuide-Projects---Project3/raw/main/models/lreg_bbry_tuned_model.pkl'

# Function to download models
def download_model(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

# Function to load the best model
def load_best_model():
    model_path = os.path.join(models_directory, 'lreg_bbry_tuned_model.pkl')
    if not os.path.exists(model_path):
        st.warning("Downloading the best model. Please wait...")
        download_model(best_model_url, model_path)
    try:
        model = joblib.load(model_path)
        st.success("Best model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Error loading the best model: {e}")
        return None

# Load the best model
best_model = load_best_model()

# Define Streamlit app
def main():
    st.title("Machine Learning Model Deployment with Streamlit")

    # Sidebar with user input
    st.sidebar.header("User Input")
    user_input = get_user_input()

    # Display model predictions
    display_predictions(user_input)

    # Display SHAP force plot
    display_shap_force_plot()

# Rest of your code remains the same...

if __name__ == '__main__':
    main()
