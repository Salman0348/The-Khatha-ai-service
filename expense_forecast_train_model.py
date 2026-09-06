import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = "datasets/individual_expense_forecast_dataset.csv"
MODEL_PATH = "models/expense_forecast_model.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# SPLIT FEATURES AND TARGET
# ============================================================

X = df.drop("target_month_end_spending", axis=1)
y = df["target_month_end_spending"]

feature_columns = X.columns.tolist()

print("\nFeatures:", feature_columns)


# ============================================================
# TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)


# ============================================================
# CREATE MODEL
# ============================================================

model = RandomForestRegressor(
    # Number of trees in the forest
    n_estimators=200,

    # Maximum depth of each tree
    max_depth=15,

    # Use all available CPU cores
    n_jobs=-1,

    # Reproducibility
    random_state=42
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining the model...")

model.fit(X_train, y_train)

print("\nModel training completed.")


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# EVALUATION METRICS
# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = root_mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)


# ============================================================
# PRINT EVALUATION RESULTS
# ============================================================

print("\nModel Evaluation Metrics:")

print("Mean Absolute Error (MAE):", round(mae, 2))

print("Mean Squared Error (MSE):", round(mse, 2))

print("Root Mean Squared Error (RMSE):", round(rmse, 2))

print("R-squared (R2):", round(r2, 4))


# ============================================================
# ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.6)

# Perfect prediction reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual Month-End Spending")
plt.ylabel("Predicted Month-End Spending")

plt.title("Actual vs Predicted Month-End Spending")

plt.tight_layout()

plt.savefig(
    "models/forecast_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nActual vs predicted plot saved to "
    "models/forecast_actual_vs_predicted.png"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, MODEL_PATH)

print("\nModel saved to", MODEL_PATH)


# ============================================================
# SAVE FEATURE COLUMNS
# ============================================================

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("Feature columns saved.")


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "model_name": "Random Forest Regressor",
    "version": "1.0",
    "target": "target_month_end_spending",
    "features": feature_columns,

    "training_samples": X_train.shape[0],
    "testing_samples": X_test.shape[0],

    "mae": round(mae, 2),
    "mse": round(mse, 2),
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

print("\nAll model evaluation information saved successfully!")