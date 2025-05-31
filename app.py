
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("burst_predictor_model.pkl")

st.title("🔮 1xBet Crash Burst Predictor")
st.write("Predict the next round's crash point based on previous data.")

players = st.number_input("👥 Players", min_value=0, step=1)
bet_amount = st.number_input("💰 Total Bet Amount", min_value=0, step=100)
win_amount = st.number_input("🏆 Total Win Amount", min_value=0, step=100)

if st.button("Predict Burst Multiplier"):
    input_df = pd.DataFrame([[players, bet_amount, win_amount]], columns=["players", "bet_amount", "win_amount"])
    pred = model.predict(input_df)[0]
    lower, upper = max(1.0, pred - 0.3), pred + 0.3
    st.success(f"Estimated Burst: {lower:.2f}x – {upper:.2f}x")
