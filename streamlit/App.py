import streamlit as st
import joblib
import pandas as pd
import time

def print_fraud(row):
    st.write(f"Amount : {row.iloc[0]['Amount']}")

@st.cache_resource
def load_model():
    return joblib.load('model/CreditCardFraudXGB.pkl')

model = load_model()

X = pd.read_csv('data/realtime_split.csv')

st.title("Real-Time Fraud Detection Simulator")

st.caption("The simulation will only log the fraudulent transactions on the screen. Since there are huge no. of transactions.(approx: 60000)")

fraud, legit = 0,0

placeholder = st.empty()

if st.button("Start Simulation"):
    for i in range(len(X)):
        row = X.iloc[i:i+1]
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]
        
        if pred == 1:
            fraud += 1
            st.write(f"Transaction #{i+1}: FRAUDULENT | Confidence: {proba:.2f}")
            print_fraud(row)
        else:
            legit += 1
            placeholder.write(f"Transaction #{i+1}: LEGITIMATE | Confidence: {proba:.2f}")
        
        time.sleep(0.1)  # simulate delay

    st.write("---")
    st.write(f"✅ Legitimate Transactions: {legit}")
    st.write(f"⚠️ Fraudulent Transactions: {fraud}")