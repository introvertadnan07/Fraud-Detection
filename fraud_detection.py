import streamlit as st
import pandas as pd
import joblib

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🚨",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

# ------------------ UI ------------------
st.title("🚨 Fraud Detection Prediction App")
st.caption("Machine Learning based transaction fraud detection")

st.divider()

# ------------------ INPUT FORM ------------------
with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"]
        )
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)

    with col2:
        oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=1000.0)
        newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=900.0)

    oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

    submit = st.form_submit_button("🔍 Predict")

# ------------------ PREDICTION ------------------
if submit:
    # Input dataframe (must match training columns)
    input_data = pd.DataFrame([{
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    st.metric("Fraud Probability", f"{probability * 100:.2f}%")

    if probability > 0.7:
        st.error("🚨 High risk transaction detected")
    elif probability > 0.4:
        st.warning("⚠️ Medium risk transaction")
    else:
        st.success("✅ Low risk transaction")

    st.progress(int(probability * 100))

    # ------------------ EXPLANATION ------------------
    with st.expander("ℹ️ How this prediction works"):
        st.write("""
        The model analyzes:
        - Transaction amount
        - Sender and receiver balances
        - Balance changes
        - Transaction type
        
        Unusual balance patterns and large transfers increase fraud probability.
        """)

# ------------------ FOOTER ------------------
st.caption("Built with Streamlit & Machine Learning")
