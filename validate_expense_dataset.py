import pandas as pd
import matplotlib.pyplot as plt


DATASET_PATH = "datasets/individual_expense_forecast_dataset.csv"


df = pd.read_csv(DATASET_PATH)


print("Dataset shape:")
print(df.shape)


print("\nColumns:")
print(df.columns.tolist())


print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

invalid_days = df[
    df["days_remaining"]
    !=
    (
        df["days_in_month"]
        -
        df["days_elapsed"]
    )
]


print(
    "\nInvalid day calculations:",
    len(invalid_days)
)

calculated_velocity = (
    df["current_month_spending"]
    /
    df["days_elapsed"]
)


difference = (
    calculated_velocity
    -
    df["spending_velocity"]
).abs()


print(
    "\nVelocity errors:",
    (difference > 0.01).sum()
)

calculated_average = (
    df["previous_month_spending"]
    +
    df["second_last_month_spending"]
    +
    df["third_last_month_spending"]
) / 3


difference = (
    calculated_average
    -
    df["average_last_3_months"]
).abs()


print(
    "\nAverage calculation errors:",
    (difference > 0.01).sum()
)

category_columns = [
    "current_food_dining",
    "current_transportation",
    "current_shopping",
    "current_groceries",
    "current_bills_utilities",
    "current_healthcare",
    "current_entertainment",
    "current_education",
    "current_travel",
    "current_miscellaneous"
]


category_total = df[
    category_columns
].sum(axis=1)


difference = (
    category_total
    -
    df["current_month_spending"]
).abs()


print(
    "\nCategory distribution errors:",
    (difference > 1).sum()
)

invalid_targets = df[
    df["target_month_end_spending"]
    <
    df["current_month_spending"]
]


print(
    "\nInvalid target rows:",
    len(invalid_targets)
)

print(
    "\nStatistics:"
)

print(
    df.describe()
)

correlation = (
    df.corr()
    [
        "target_month_end_spending"
    ]
    .sort_values(
        ascending=False
    )
)


print(
    correlation
)

# Visualizing the distribution of the target variable through a histogram in matplotlib

plt.hist(
    df["target_month_end_spending"],
    bins=50
)

plt.xlabel(
    "Target Month End Spending"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Target Distribution"
)

plt.show()