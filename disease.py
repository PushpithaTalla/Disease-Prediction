import streamlit as st
import pickle
import numpy as np
import statistics

# Load the pickle file
with open("D:\streamlit\model_data (1).pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Extract the data
final_rf_model = loaded_data["final_rf_model"]
final_nb_model = loaded_data["final_nb_model"]
final_svm_model = loaded_data["final_svm_model"]
symptom_index = loaded_data["symptom_index"]
encoder_classes = loaded_data["encoder_classes"]

# Additional data related to diseases (this can be customized)
disease_info = {
    "Fungal infection": "Fungal infections are caused by fungi and can affect skin, nails, and lungs. Treatment includes antifungal medications.",
    "Common cold": "A viral infection of the upper respiratory tract. Common symptoms include a runny nose, cough, and sore throat. Rest and hydration are important for recovery.",
    "COVID-19": "A respiratory illness caused by the SARS-CoV-2 virus. Symptoms include fever, cough, and shortness of breath. Vaccination and social distancing are recommended."
}

# Streamlit interface
st.title("Disease Prediction Based on Symptoms")
st.sidebar.title("Navigation")

# Sidebar for navigation
page = st.sidebar.radio("Select a Page", ["Home", "Disease Prediction", "Model Info"])

if page == "Home":
    st.header("Welcome to the Disease Prediction App")
    st.write("This app predicts diseases based on symptoms you input. Choose 'Disease Prediction' to begin.")

elif page == "Disease Prediction":
    st.header("Input Symptoms")
    
    # Text input for symptoms
    symptom_input = st.text_area("Enter Symptoms (comma separated)", "Itching, Skin Rash, Nodal Skin Eruptions")
    
    if st.button("Predict Disease"):
        # Process symptoms for prediction
        symptoms_list = symptom_input.split(",")
        input_data = [0] * len(symptom_index)
        
        for symptom in symptoms_list:
            symptom = symptom.strip()  # Clean up the input
            if symptom in symptom_index:
                index = symptom_index[symptom]
                input_data[index] = 1
        
        # Reshaping the input data to suitable format for model predictions
        input_data = np.array(input_data).reshape(1, -1)
        
        # Generate predictions from models
        rf_prediction = encoder_classes[final_rf_model.predict(input_data)[0]]
        nb_prediction = encoder_classes[final_nb_model.predict(input_data)[0]]
        svm_prediction = encoder_classes[final_svm_model.predict(input_data)[0]]
        
        # Final prediction using mode of all model predictions
        final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
        
        # Display the predictions
        st.subheader("Predictions:")
        st.write(f"Random Forest Model Prediction: {rf_prediction}")
        st.write(f"Naive Bayes Model Prediction: {nb_prediction}")
        st.write(f"SVM Model Prediction: {svm_prediction}")
        st.write(f"Final Prediction (based on majority voting): {final_prediction}")
        
        # Display additional information related to the disease
        st.subheader("Disease Information:")
        st.write(disease_info.get(final_prediction, "No additional information available."))

elif page == "Model Info":
    st.header("Model Information")
    st.write("""
    - **Random Forest Model**: A model that uses multiple decision trees to classify data.
    - **Naive Bayes Model**: A probabilistic model based on Bayes' theorem.
    - **SVM Model**: A supervised learning model that finds the optimal hyperplane for classification.
    """)
    st.write("All models were trained on a dataset of symptoms and diseases, and predictions are based on the majority voting of these models.")
