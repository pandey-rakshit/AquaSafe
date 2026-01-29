
import pandas as pd

def predict(model, encoder, input_df: pd.DataFrame):
    probs = model.predict_proba(input_df)
    pred_class_idx = probs.argmax(axis=1)[0]

    predicted_label = encoder.inverse_transform([pred_class_idx])[0]
    confidence = probs.max(axis=1)[0]

    return predicted_label, confidence, probs
