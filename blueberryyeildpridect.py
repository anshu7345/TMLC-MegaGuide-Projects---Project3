import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from PIL import Image
""" hello
"""
import os

# Get the absolute path to the directory where your models are stored
models_directory = os.path.join(os.getcwd(), 'https://github.com/anshu7345/TMLC-MegaGuide-Projects---Project3/tree/main')

# Load models and objects
best_model_path = os.path.join(models_directory, 'lreg_bbry_tuned_model.pkl')
rf_final_path = os.path.join(models_directory, 'rf_bbry_tuned_model.pkl')
fs_rf_path = os.path.join(models_directory, 'rf_fs_object.joblib')

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

    # Use the loaded model to make predictions
    try:
        prediction_best_model = best_model_path.predict([list(user_input.values())])
        prediction_rf_final = rf_final_path.predict([list(user_input.values())])

        st.write(f"Best Model Prediction: {prediction_best_model[0]}")
        st.write(f"Random Forest Prediction: {prediction_rf_final[0]}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")

def display_shap_force_plot():
    # Display SHAP force plot
    st.header("SHAP Force Plot")

    # Load the force plot image
    image = Image.open('https://github.com/anshu7345/TMLC-MegaGuide-Projects---Project3/blob/main/test_force_plot1.png')
    st.image(image, caption='SHAP Force Plot', use_column_width=True)

if __name__ == '__main__':
    main()