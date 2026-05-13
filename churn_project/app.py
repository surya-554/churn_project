import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Churn Predictor", page_icon="🎯")

st.title("🎯 Customer Churn Predictor")
st.write("Enter customer details to predict churn!")

tenure = st.slider("Months with company?", 0, 72, 12)
monthly = st.number_input("Monthly Charges ($)", 0, 150, 65)
total = monthly * tenure

st.write(f"Total Charges: ${total}")

if st.button("🔍 Predict Now!"):
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)

    features = np.zeros(30)
    features[0] = tenure
    features[1] = monthly
    features[2] = total

    prob = model.predict_proba([features])[0][1]

    st.write("---")
    if prob > 0.5:
        st.error(f"⚠️ HIGH RISK! {prob*100:.1f}%")
        st.write("👉 Call this customer immediately!")
    else:
        st.success(f"✅ LOW RISK! {prob*100:.1f}%")
        st.write("👉 No action needed!")