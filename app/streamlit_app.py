import streamlit as st
from pathlib import Path

from core.model_loader import load_artifacts
from core.predictor import predict
from core.schema import get_required_features
from utils.validators import validate_input_df
from ui.sidebar import sidebar_upload
from ui.input_form import manual_input_form
from ui.output_view import show_prediction

MODELS_DIR = Path("models")

@st.cache_resource
def load_all():
    return load_artifacts(MODELS_DIR)

def main():
    st.title("AquaSafe - Water Quality Classifier")

    model, encoder, feature_names = load_all()

    uploaded_file = sidebar_upload()

    if uploaded_file:
        import pandas as pd
        df = pd.read_csv(uploaded_file)
    else:
        df = manual_input_form(feature_names)

    try:
        validate_input_df(df, feature_names)
        label, confidence, _ = predict(model, encoder, df)
        show_prediction(label, confidence)
    except Exception as e:
        st.error(str(e))

if __name__ == "__main__":
    main()
