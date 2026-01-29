import streamlit as st

def sidebar_upload():
    st.sidebar.header("Upload Water Sample CSV")
    return st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"]
    )
