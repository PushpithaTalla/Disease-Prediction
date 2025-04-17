import streamlit as st
import pickle
import numpy as np
import statistics

# Load the pickle file
with open("D:\\streamlit\\model_data (1).pkl", "rb") as f:
    loaded_data = pickle.load(f)

# Extract the data
final_rf_model = loaded_data["final_rf_model"]
final_nb_model = loaded_data["final_nb_model"]
final_svm_model = loaded_data["final_svm_model"]
symptom_index = loaded_data["symptom_index"]
encoder_classes = loaded_data["encoder_classes"]

# Additional data related to diseases
disease_info = {
    "Fungal infection": "Fungal infections are caused by fungi and can affect skin, nails, and lungs. Treatment includes antifungal medications.",
    "Common cold": "A viral infection of the upper respiratory tract. Common symptoms include a runny nose, cough, and sore throat. Rest and hydration are important for recovery.",
    "COVID-19": "A respiratory illness caused by the SARS-CoV-2 virus. Symptoms include fever, cough, and shortness of breath. Vaccination and social distancing are recommended.",
}

# Streamlit interface
st.title("🩺 Disease Prediction App")
st.sidebar.title("📋 Navigation")

# Sidebar for navigation
page = st.sidebar.radio("Select a Page", ["Home", "Disease Prediction", "Model Info"])

if page == "Home":
    st.header("🏠 Welcome to the Disease Prediction App")
    st.write("This app predicts diseases based on symptoms you input. Choose **'Disease Prediction'** to begin.")
    st.image("https://via.placeholder.com/800x400.png?text=Health+is+Wealth", use_column_width=True)

elif page == "Disease Prediction":
    st.header("🔍 Input Symptoms and Predict Disease")
    
    # Text input for symptoms
    symptom_input = st.text_area("📝 Enter Symptoms (comma separated)", "Itching, Skin Rash, Nodal Skin Eruptions")
    
    # Dropdown for model selection
    selected_model = st.selectbox(
        "🤖 Select a Model for Prediction",
        ["Random Forest", "Naive Bayes", "SVM", "Majority Voting (All Models)"]
    )
    
    if st.button("🔮 Predict Disease"):
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
        
        # Generate predictions based on selected model
        if selected_model == "Random Forest":
            prediction = encoder_classes[final_rf_model.predict(input_data)[0]]
        elif selected_model == "Naive Bayes":
            prediction = encoder_classes[final_nb_model.predict(input_data)[0]]
        elif selected_model == "SVM":
            prediction = encoder_classes[final_svm_model.predict(input_data)[0]]
        else:  # Majority Voting
            rf_prediction = encoder_classes[final_rf_model.predict(input_data)[0]]
            nb_prediction = encoder_classes[final_nb_model.predict(input_data)[0]]
            svm_prediction = encoder_classes[final_svm_model.predict(input_data)[0]]
            prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
        
        # Display the prediction
        st.subheader("🩺 Prediction:")
        st.write(f"**Predicted Disease:** {prediction}")
        
        # Display additional information related to the disease
        st.subheader("ℹ️ Disease Information:")
        st.write(disease_info.get(prediction, "No additional information available."))

elif page == "Model Info":
    st.header("🧠 Model Information")
    st.write("Learn more about the models used in this application:")
    st.markdown("""
    - **🌲 Random Forest Model**: A model that uses multiple decision trees to classify data.
    - **📊 Naive Bayes Model**: A probabilistic model based on Bayes' theorem.
    - **🔗 SVM Model**: A supervised learning model that finds the optimal hyperplane for classification.
    """)
    st.write("The predictions are based on models trained on a dataset of symptoms and diseases. In the majority voting approach, all three models contribute to the final prediction.")
    st.image("https://via.placeholder.com/800x400.png?text=Machine+Learning+Models", use_column_width=True)


