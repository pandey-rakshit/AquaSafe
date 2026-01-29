import streamlit as st

def show_prediction(label, confidence):
    st.success(f"Predicted Water Quality Class: **{label}**")
    st.metric("Confidence", f"{confidence:.2%}")
