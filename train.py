import pandas as pd
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from utils.preprocess import clean_text

#load dataset
df = pd.read_csv('datasets/description_categories_dataset.csv')

#clean text
df["description"] = df["description"].apply(clean_text)

#Featues and labels
X = df["description"]
y = df["category"]

#Text into TF-IDF vectors
vectorizer = TfidfVectorizer(
   ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

#Spliting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#training the model 
model = LogisticRegression()
model.fit(X_train, y_train)

#prediction
predections = model.predict(X_test)


# evaluation of the model

# Metrics
accuracy = accuracy_score(y_test, predections)
classification_rep = classification_report(
    y_test,
    predections,
    output_dict=True
)

#accuracy
print("Accuracy: ", accuracy)

#classification report
print("Classification Report: \n", classification_report(y_test, predections))

#confusion matrix
print("confusion matrix: \n", confusion_matrix(y_test, predections))

#saving the model and vectorizer
# Saving the model and vectorizer
joblib.dump(model, "models/logistic_regression_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Save model information
model_info = {
    "model_name": "Logistic Regression",
    "version": "1.0",
    "task": "Expense Description Categorization",
    "algorithm": "TF-IDF + Logistic Regression",
    "training_samples": len(X_train.toarray()),
    "testing_samples": len(X_test.toarray()),
    "accuracy": round(accuracy, 4),
    "categories": sorted(df["category"].unique().tolist()),
    "ngram_range": [1, 2]
}

with open("models/expense_entry_categorization_model_info.json", "w") as file:
    json.dump(model_info, file, indent=4)

print("Model, vectorizer, and model info saved successfully!")