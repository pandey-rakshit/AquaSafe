
import streamlit as st
import pandas as pd

def manual_input_form(feature_names):
    st.subheader("Manual Input")

    data = {}
    for col in feature_names:
        data[col] = st.number_input(col, value=0.0)

    return pd.DataFrame([data])
