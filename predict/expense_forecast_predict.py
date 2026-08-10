import joblib
import pandas as pd

# Load model
model = joblib.load("models/expense_forecast_model.pkl")

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


