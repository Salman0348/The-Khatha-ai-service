import joblib

from utils.preprocess import clean_text


# Load model once
model = joblib.load(
    "models/logistic_regression_model.pkl"
)

vectorizer = joblib.load(
    "models/tfidf_vectorizer.pkl"
)


def predict_category(description: str):

    cleaned_text = clean_text(description)

    vector = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    confidence = probabilities.max()


    return {
        "category": prediction,
        "confidence": round(
            float(confidence),
            2
        )
    }