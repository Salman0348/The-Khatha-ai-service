import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from utils.preprocess import clean_text

# Load dataset
df = pd.read_csv('datasets/description_categories_dataset.csv')

# Clean text
df["description"] = df["description"].apply(clean_text)

# Features and labels
X = df["description"]
y = df["category"]

# Text into TF-IDF vectors
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Training the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)


# =========================
# MODEL EVALUATION
# =========================

# Accuracy
accuracy = accuracy_score(y_test, predictions)

# Classification report
report = classification_report(
    y_test,
    predictions,
    output_dict=True
)

# Extract weighted-average metrics for thesis
precision = report["weighted avg"]["precision"]
recall = report["weighted avg"]["recall"]
f1_score = report["weighted avg"]["f1-score"]

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score)

print("\nClassification Report:")
print(classification_report(y_test, predictions))


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)

# Create confusion matrix image
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(
    ax=ax,
    xticks_rotation=45,
    values_format="d"
)

plt.title("Confusion Matrix - Smart Expense Categorization")
plt.tight_layout()

# Save image for thesis
plt.savefig(
    "models/categorization_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================
# SAVE MODEL AND VECTORIZER
# =========================

joblib.dump(
    model,
    "models/logistic_regression_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/tfidf_vectorizer.pkl"
)


# =========================
# SAVE MODEL INFORMATION
# =========================

model_info = {
    "model_name": "Logistic Regression",
    "version": "1.0",
    "task": "Expense Description Categorization",
    "algorithm": "TF-IDF + Logistic Regression",

    "training_samples": X_train.shape[0],
    "testing_samples": X_test.shape[0],

    "accuracy": round(accuracy, 4),
    "precision_weighted": round(precision, 4),
    "recall_weighted": round(recall, 4),
    "f1_score_weighted": round(f1_score, 4),

    "categories": sorted(
        df["category"].unique().tolist()
    ),

    "ngram_range": [1, 2]
}

with open(
    "models/expense_entry_categorization_model_info.json",
    "w"
) as file:
    json.dump(model_info, file, indent=4)

print("Model, vectorizer, metrics, and confusion matrix saved successfully!")