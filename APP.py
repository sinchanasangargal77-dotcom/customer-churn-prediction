import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
with open("models/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page title
st.title("Customer Churn Prediction 🔮")
st.write("Enter customer details below to predict if they will churn or not!")
st.write("---")

st.subheader("Enter Customer Details:")

# Input fields
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0, 120, 50)
total_charges = st.slider("Total Charges", 0, 9000, 1000)

senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", [0, 1])
dependents = st.selectbox("Dependents", [0, 1])
phone_service = st.selectbox("Phone Service", [0, 1])
paperless_billing = st.selectbox("Paperless Billing", [0, 1])
gender = st.selectbox("Gender", [0, 1])

contract = st.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service",
            ["DSL", "Fiber optic", "No"])
payment_method = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)",
             "Credit card (automatic)"])

st.write("---")

if st.button("Predict Churn 🔮"):

    # Encode contract type
    contract_one_year = 1 if contract == "One year" else 0
    contract_two_year = 1 if contract == "Two year" else 0

    # Encode internet service
    internet_fiber = 1 if internet_service == "Fiber optic" else 0
    internet_no = 1 if internet_service == "No" else 0

    # Encode payment method
    payment_credit = 1 if payment_method == "Credit card (automatic)" else 0
    payment_electronic = 1 if payment_method == "Electronic check" else 0
    payment_mailed = 1 if payment_method == "Mailed check" else 0

    # Create input array
    input_data = np.array([[gender, senior_citizen, partner, dependents,
                            tenure, phone_service, paperless_billing,
                            monthly_charges, total_charges,
                            0, 0, internet_fiber, internet_no,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            contract_one_year, contract_two_year,
                            payment_credit, payment_electronic,
                            payment_mailed]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    # Show result
    if prediction[0] == 1:
        st.error(f"⚠️ High Churn Risk! Probability: {probability:.0%}")
    else:
        st.success(f"✅ Low Churn Risk! Probability: {probability:.0%}")