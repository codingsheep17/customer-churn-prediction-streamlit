import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model + features
model = joblib.load("churn_model.pkl")
features = joblib.load("features.pkl")

st.title("📊 Customer Churn Prediction", text_alignment='center')

st.info("ℹ️ This churn prediction model has a **recall of 91%** ✅ (good at detecting customers who may leave) and an **accuracy of 63%** ⚖️")

# User Inputs
tenure = st.slider("Tenure (Months) 🗓️", 0, 72)
monthly_charges = st.number_input("$ Monthly Charges 💵", 0.0)

contract = st.selectbox("Contract 📄", ["Month-to-month", "One year", "Two year"])
tech_support = st.selectbox("Tech Support 🛠️", ["Yes", "No"])
internet = st.selectbox("Internet Service 🌐", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method 💳", ["Credit card", "Electronic check", "Mailed check"])

# Create input dict
input_dict = {
    "Tenure_Months": tenure,
    "Monthly_Charges": monthly_charges,
    "Contract_One_year": 1 if contract == "One year" else 0,
    "Contract_Two_year": 1 if contract == "Two year" else 0,
    "Tech_Support_Yes": 1 if tech_support == "Yes" else 0,
    "Internet_Service_Fiber_optic": 1 if internet == "Fiber optic" else 0,
    "Internet_Service_No": 1 if internet == "No" else 0,
    "Payment_Method_Credit_card_automatic": 1 if payment == "Credit card" else 0,
    "Payment_Method_Electronic_check": 1 if payment == "Electronic check" else 0,
    "Payment_Method_Mailed_check": 1 if payment == "Mailed check" else 0,
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# matching columns
input_df = input_df.reindex(columns=features, fill_value=0)

# Prediction setup
if st.button("Predict 🤖"):
    prob = model.predict_proba(input_df)[0][1]
    prediction = model.predict(input_df)[0]

    st.write(f"Churn Probability: {prob:.2f} 📈")

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn ❌")
    else:
        st.success("✅ Customer is likely to stay 🎉")

# Credit
st.caption("Developed by Syed Haseeb Shah 👨‍💻", text_alignment='center')