import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# Load model development
# model = joblib.load("models/expense_forecast_model.pkl")

# Load model production (load from hugging face)
model_path = hf_hub_download(
    repo_id="salman0348/thekhatha-expense-forecast-model",
    filename="expense_forecast_model.pkl"
)

model = joblib.load(model_path)


# Load feature order
feature_columns = joblib.load("models/feature_columns.pkl")


def predict_expense(features):
    """
    features should be a dictionary containing
    all 21 feature values.
    """

    df = pd.DataFrame([features])

    # Ensure correct column order
    df = df[feature_columns]

    prediction = model.predict(df)

    return float(prediction[0])


