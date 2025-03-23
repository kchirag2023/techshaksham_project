import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and label encoder
model = joblib.load("disease_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load dataset to get symptom names
df = pd.read_csv("/Users/kumarchirag/Desktop/techsaksham/Final_Augmented_dataset_Diseases_and_Symptoms.csv")
symptoms = list(df.columns[:-1])  # Extract symptom names

st.title("Disease Prediction App 🏥")

# Multi-select dropdown for symptoms
selected_symptoms = st.multiselect("Select Symptoms:", symptoms)

# Convert selected symptoms into model input format
input_data = np.zeros(len(symptoms))  # Initialize input as all zeros
for symptom in selected_symptoms:
    input_data[symptoms.index(symptom)] = 1  # Mark selected symptoms as 1

# Predict button
if st.button("Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom!")
    else:
        prediction = model.predict([input_data])[0]
        disease_name = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🩺 Predicted Disease: **{disease_name}**")

st.sidebar.header("About")
st.sidebar.info("This app predicts diseases based on selected symptoms using a trained ML model.")
