import streamlit as st
import joblib
from PIL import Image
import os


best_model = joblib.load('https://github.com/anshu7345/TMLC-MegaGuide-Projects---Project3/tree/main/lreg_bbry_tuned_model.pkl')
rf_final = joblib.load('https://github.com/anshu7345/TMLC-MegaGuide-Projects---Project3/tree/main/rf_bbry_tuned_model.pkl')

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

    try:
        if best_model is not None:
            prediction_best_model = best_model.predict([list(user_input.values())])
            st.write(f"Best Model Prediction: {prediction_best_model[0]}")
        else:
            st.error("Best model is not loaded.")

        if rf_final is not None:
            prediction_rf_final = rf_final.predict([list(user_input.values())])
            st.write(f"Random Forest Prediction: {prediction_rf_final[0]}")
        else:
            st.error("Random Forest model is not loaded.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

def display_shap_force_plot():
    # Display SHAP force plot from a URL
    st.header("SHAP Force Plot")

    # URL to the image
    image_url = 'test_force_plot1.png'

    # Display the image directly using st.image
    st.image(image_url, caption='SHAP Force Plot', use_column_width=True)

if __name__ == '__main__':
    main()
