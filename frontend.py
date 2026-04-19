import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/predict"

st.title("🏦 Loan Approval Prediction")

st.write("Enter applicant details:")

# Inputs
ApplicantIncome = st.number_input("Applicant Income", min_value=0.0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
LoanAmount = st.number_input("Loan Amount", min_value=0.0)

Credit_History = st.selectbox("Credit History", [0, 1])
Loan_Amount_Term = st.number_input("Loan Term", min_value=1)

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])


if st.button("Predict Loan Status"):

    data = {
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Credit_History": Credit_History,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Gender": Gender,
        "Married": Married,
        "Dependents": Dependents,
        "Education": Education,
        "Self_Employed": Self_Employed,
        "Property_Area": Property_Area
    }

    response = requests.post(API_URL, json=data)

    if response.status_code == 200:
        result = response.json()

        st.success(f"Prediction: {result['loan_approval']}") 
        st.write(f"Confidence: {result['probability']}")
        st.write("Class Probabilities:", result["class_probabilities"])

    else:
        st.error("Error in prediction")