import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD FILES
# =========================
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_importance = pd.read_csv("feature_importance.csv")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(layout="wide")
st.title("Customer Churn Prediction Dashboard")

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns([1, 2])

# =========================
# LEFT SIDE → INPUT + GRAPH
# =========================
with col1:
    st.header("Input Customer Details")

    credit_score = st.number_input("Credit Score", 300, 900)
    age = st.number_input("Age", 18, 100)
    tenure = st.number_input("Tenure", 0, 10)
    balance = st.number_input("Balance", 0.0)
    num_products = st.number_input("Number of Products", 1, 4)
    has_card = st.selectbox("Has Credit Card", [0, 1])
    is_active = st.selectbox("Is Active Member", [0, 1])
    salary = st.number_input("Estimated Salary", 0.0)

    gender = st.selectbox("Gender", ["Male", "Female"])
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

    # FEATURE IMPORTANCE GRAPH
    st.subheader("Feature Importance")

    fig, ax = plt.subplots()
    ax.barh(feature_importance['Feature'], feature_importance['Importance'])
    ax.invert_yaxis()
    st.pyplot(fig)

# =========================
# RIGHT SIDE → OUTPUT
# =========================
with col2:
    st.header("Prediction Result")

    if st.button("Predict"):

        # ENCODING
        male, female = (1, 0) if gender == "Male" else (0, 1)

        if geography == "France":
            france, germany, spain = 1, 0, 0
        elif geography == "Germany":
            france, germany, spain = 0, 1, 0
        else:
            france, germany, spain = 0, 0, 1

        # INPUT ARRAY
        input_data = np.array([[
            credit_score, age, tenure,
            balance, num_products, has_card,
            is_active, salary,
            male, female,
            france, germany, spain
        ]])

        # SCALE
        input_scaled = scaler.transform(input_data)

        # PREDICTION
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0][1]

        # OUTPUT
        st.subheader("Prediction")

        # Show probabilities FIRST
        st.write(f"**Churn Probability:** {probability*100:.2f}%")
        st.write(f"**Retention Probability:** {(1-probability)*100:.2f}%")

        # Final decision
        if probability >= 0.5:
            st.error("⚠️ Likely to churn")
        else:
            st.success("✅ Not likely to churn")

# pip install streamlit
# streamlit run app.py