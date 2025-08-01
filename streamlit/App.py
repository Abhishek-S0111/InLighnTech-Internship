import streamlit as st
import joblib
import pandas as pd
import time

@st.cache_resource
def load_model():
    return joblib.load('model/CreditCardFraudXGB.pkl')

model = load_model()

X = pd.read_csv('data/realtime_split.csv')

st.title("Real-Time Fraud Detection Simulator")

fraud, legit = 0,0

placeholder = st.empty()

if st.button("Start Simulation"):
    for i in range(len(X)):
        row = X.iloc[i:i+1]
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]
        
        if pred == 1:
            fraud += 1
            placeholder.write(f"Transaction #{i+1}: FRAUDULENT | Confidence: {proba:.2f}")
        else:
            legit += 1
            placeholder.write(f"Transaction #{i+1}: LEGITIMATE | Confidence: {proba:.2f}")
        
        time.sleep(0.1)  # simulate delay

    st.write("---")
    st.write(f"✅ Legitimate Transactions: {legit}")
    st.write(f"⚠️ Fraudulent Transactions: {fraud}")