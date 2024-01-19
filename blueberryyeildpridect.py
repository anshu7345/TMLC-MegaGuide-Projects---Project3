import streamlit as st
import joblib
from PIL import Image
import os
import requests

# Get the absolute path to the directory where your models are stored
models_directory = os.path.join(os.getcwd(), 'models')  # Update 'models' to your actual directory name

# Ensure the models directory exists
os.makedirs(models_directory, exist_ok=True)

# URLs to the models on GitHub
best_model_url = 'https://github.com/anshu7345/TMLC-MegaGuide-Projects---Project3/raw/main/models/lreg_bbry_tuned_model.pkl'

# Download models
def download_model(url, local_path):
    response = requests.get(url)
    with open(local_path, 'wb') as f:
        f.write(response.content)

download_model(best_model_url, os.path.join(models_directory, 'lreg_bbry_tuned_model.pkl'))

# Load models using joblib.load
try:
    best_model = joblib.load(os.path.join(models_directory, 'lreg_bbry_tuned_model.pkl'))
    st.success("Best model loaded successfully.")
except Exception as e:
    st.error(f"Error loading the best model: {e}")
    best_model = None

# Define Streamlit app
def main():
    st.title("Blueberry Yield Prediction")

    # Sidebar with user input
    st.sidebar.header("User Input")
    user_input = get_user_input()

    # Predict button
    predict_button = st.sidebar.button("Predict")

    # Display model predictions if the Predict button is clicked
    if predict_button:
        display_predictions(user_input)

    # Display SHAP force plot
    display_shap_force_plot()

def get_user_input():
    # Create a dictionary to store user input
    user_input = {}

    # Add Streamlit widgets for user input (customize as needed)
    user_input['feature1'] = st.sidebar.slider('Feature 1', min_value=0.0, max_value=100.0, value=50.0)
    user_input['feature2'] = st.sidebar.slider('Feature 2', min_value=0.0, max_value=100.0, value=50.0)
    # Add more features as needed

    return user_input

def display_predictions(user_input):
    # Display predictions from the models
    st.header("Model Predictions")

    try:
        if best_model is not None:
            prediction_best_model = best_model.predict([list(user_input.values())])
            st.write(f"Best Model Prediction: {prediction_best_model[0]}")
        else:
            st.error("Best model is not loaded.")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

def display_shap_force_plot():
    # Display SHAP force plot from a URL
    st.header("SHAP Force Plot")

    # URL to the image
    image_url = 'https://github.com/anshu7345/TMLC-MegaGuide-Projects---Project3/raw/main/test_force_plot1.png'

    # Display the image directly using st.image
    st.image(image_url, caption='SHAP Force Plot', use_column_width=True)

if __name__ == '__main__':
    main()
