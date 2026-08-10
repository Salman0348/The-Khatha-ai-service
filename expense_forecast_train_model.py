import pandas as pd 
import joblib 
import json


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (root_mean_squared_error, mean_absolute_error, r2_score)


# config
DATASET_PATH = "datasets/individual_expense_forecast_dataset.csv"
MODEL_PATH = "models/expense_forecast_model.pkl"

# Load the dataset
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# Split the dataset into features and target variable
X = df.drop("target_month_end_spending", axis=1)
y = df["target_month_end_spending"]


feature_columns = X.columns.tolist()
print("\nFeatures: ", feature_columns)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# model creation
model = RandomForestRegressor(
    # number of trees in the forest
    n_estimators=200,
    # maximum depth of the tree
    max_depth=15,
    # use all cpu cores for training
    n_jobs=-1,
    # random state for reproducibility
     random_state=42)

# train the model
print("\nTraining the model...")

model.fit(X_train, y_train)

print("\nModel training completed.")

# predict on the test set
y_pred = model.predict(X_test)

# evaluate the model
mae = mean_absolute_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

# print the evaluation metrics
print("\nModel Evaluation Metrics:")
print("Mean Absolute Error (MAE):",round(mae, 2))
print("Root Mean Squared Error (RMSE):",round(rmse, 2))
print("R-squared (R2):",round(r2, 2))

# save the model 
joblib.dump(model, MODEL_PATH) 
print("\nModel saved to", MODEL_PATH)

# save the feature columns
joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("Feature columns saved.")

# save the model info
model_info = {
    "model_name": "Random Forest Regressor",
    "version": "1.0",
    "target": "target_month_end_spending",
    "features": feature_columns,
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "r2": round(r2, 4)
}

with open(
    "models/expense_forecast_model_info.json",
    "w"
) as file:
    json.dump(
        model_info,
        file,
        indent=4
    )

print("Model info saved.")



