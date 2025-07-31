import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load('/model/CreditCardFraudXGB.pkl')

model = load_model()

X = pd.read_csv('/data/realtime_split.csv')

st.title("Real-Time Fraud Detection Simulator")

fraud, legit = 0,0

if st.button("Start Simulation"):
    for i in range(len(X)):
        row = X.iloc[i:i+1]
        pred = model.predict(row)[0]
        proba = model.predict_proba(row)[0][1]
        
        if(pred == 1):
            fraud += 1
            st.write(f"Transaction #{i+1}: {'FRAUDULENT' if pred else 'LEGITIMATE'} | Confidence: {proba:.2f}")
        else:
            legit += 1
        
    st.write(f"===================Final Results================\nLegitimate Transactions : {legit}\nFraudulent Transactions : {fraud}")