import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

NUM_SAMPLES = 20000
OUTPUT_FILE = "datasets/individual_expense_forecast_dataset.csv"

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ============================================================
# SPENDING CATEGORIES
# ============================================================

CATEGORIES = [
    "Food & Dining",
    "Transportation",
    "Shopping",
    "Groceries",
    "Bills & Utilities",
    "Healthcare",
    "Entertainment",
    "Education",
    "Travel",
    "Miscellaneous",
]


# ============================================================
# HELPER FUNCTION
# ============================================================

def generate_category_spending(
    total_spending,
    behavior
):
    """
    Distribute current spending across the 10 categories.

    The category values will add up to approximately
    current_month_spending.
    """

    # Different spending behavior produces
    # different category distributions.

    if behavior == "food_heavy":

        weights = np.random.dirichlet(
            [5, 1, 1, 2, 1, 1, 2, 1, 1, 1]
        )

    elif behavior == "shopping_heavy":

        weights = np.random.dirichlet(
            [2, 1, 6, 1, 1, 1, 2, 1, 1, 1]
        )

    elif behavior == "travel_heavy":

        weights = np.random.dirichlet(
            [2, 2, 1, 1, 1, 1, 2, 1, 6, 1]
        )

    elif behavior == "bills_heavy":

        weights = np.random.dirichlet(
            [2, 1, 1, 2, 6, 1, 1, 1, 1, 1]
        )

    elif behavior == "balanced":

        weights = np.random.dirichlet(
            [2] * 10
        )

    else:

        weights = np.random.dirichlet(
            [1] * 10
        )

    category_values = (
        weights * total_spending
    )

    return category_values


# ============================================================
# GENERATE ONE TRAINING SAMPLE
# ============================================================

def generate_sample():

    # --------------------------------------------------------
    # 1. MONTH INFORMATION
    # --------------------------------------------------------

    days_in_month = np.random.choice(
        [28, 29, 30, 31],
        p=[0.08, 0.02, 0.30, 0.60]
    )

    days_elapsed = np.random.randint(
        1,
        days_in_month + 1
    )

    days_remaining = (
        days_in_month
        - days_elapsed
    )


    # --------------------------------------------------------
    # 2. USER SPENDING BEHAVIOR
    # --------------------------------------------------------

    behavior = np.random.choice(
        [
            "food_heavy",
            "shopping_heavy",
            "travel_heavy",
            "bills_heavy",
            "balanced",
            "random",
        ],
        p=[
            0.15,
            0.15,
            0.08,
            0.12,
            0.35,
            0.15,
        ],
    )


    # --------------------------------------------------------
    # 3. HISTORICAL MONTHLY SPENDING
    # --------------------------------------------------------

    base_spending = np.random.uniform(
        3000,
        30000
    )

    previous_month_spending = max(
        0,
        np.random.normal(
            base_spending,
            base_spending * 0.15
        )
    )

    second_last_month_spending = max(
        0,
        np.random.normal(
            base_spending,
            base_spending * 0.15
        )
    )

    third_last_month_spending = max(
        0,
        np.random.normal(
            base_spending,
            base_spending * 0.15
        )
    )


    # --------------------------------------------------------
    # 4. THREE MONTH AVERAGE
    # --------------------------------------------------------

    average_last_3_months = (
        previous_month_spending
        + second_last_month_spending
        + third_last_month_spending
    ) / 3


    # --------------------------------------------------------
    # 5. CURRENT MONTH SPENDING RATE
    # --------------------------------------------------------

    # User's typical daily spending rate.

    historical_daily_rate = (
        average_last_3_months
        / days_in_month
    )


    # Add randomness to current spending behavior.

    current_daily_rate = (
        historical_daily_rate
        * np.random.uniform(
            0.6,
            1.4
        )
    )


    # --------------------------------------------------------
    # 6. CURRENT MONTH SPENDING
    # --------------------------------------------------------

    current_month_spending = (
        current_daily_rate
        * days_elapsed
    )


    # Add some random spending variation.

    current_month_spending *= np.random.uniform(
        0.85,
        1.15
    )


    current_month_spending = max(
        0,
        current_month_spending
    )


    # --------------------------------------------------------
    # 7. SPENDING VELOCITY
    # --------------------------------------------------------

    spending_velocity = (
        current_month_spending
        / days_elapsed
    )


    # --------------------------------------------------------
    # 8. ENTRY COUNT
    # --------------------------------------------------------

    # Estimate number of expenses.

    average_entry_amount = np.random.uniform(
        200,
        1500
    )

    current_month_entry_count = max(
        1,
        int(
            current_month_spending
            / average_entry_amount
        )
    )


    # --------------------------------------------------------
    # 9. PREVIOUS MONTH ENTRY COUNT
    # --------------------------------------------------------

    previous_average_entry_amount = np.random.uniform(
        200,
        1500
    )

    previous_month_entry_count = max(
        1,
        int(
            previous_month_spending
            / previous_average_entry_amount
        )
    )


    # --------------------------------------------------------
    # 10. CURRENT CATEGORY SPENDING
    # --------------------------------------------------------

    category_values = generate_category_spending(
        current_month_spending,
        behavior
    )


    # --------------------------------------------------------
    # 11. FUTURE SPENDING
    # --------------------------------------------------------

    # Estimate remaining spending based on
    # current spending velocity.

    expected_remaining_spending = (
        spending_velocity
        * days_remaining
    )


    # Historical behavior influences future spending.

    historical_influence = (
        average_last_3_months
        * np.random.uniform(
            0.05,
            0.20
        )
    )


    # Random unexpected expenses.

    unexpected_expense = 0

    if np.random.random() < 0.15:

        unexpected_expense = np.random.uniform(
            500,
            5000
        )


    # --------------------------------------------------------
    # 12. FINAL MONTH SPENDING
    # --------------------------------------------------------

    target_month_end_spending = (
        current_month_spending
        + expected_remaining_spending
        + historical_influence
        + unexpected_expense
    )


    # Add realistic randomness.

    target_month_end_spending *= np.random.uniform(
        0.90,
        1.10
    )


    # The final spending cannot be lower
    # than what has already been spent.

    target_month_end_spending = max(
        target_month_end_spending,
        current_month_spending
    )


    # --------------------------------------------------------
    # 13. CREATE RESULT
    # --------------------------------------------------------

    sample = {

        "current_month_spending":
            current_month_spending,

        "current_month_entry_count":
            current_month_entry_count,

        "previous_month_spending":
            previous_month_spending,

        "previous_month_entry_count":
            previous_month_entry_count,

        "second_last_month_spending":
            second_last_month_spending,

        "third_last_month_spending":
            third_last_month_spending,

        "average_last_3_months":
            average_last_3_months,

        "days_elapsed":
            days_elapsed,

        "days_in_month":
            days_in_month,

        "days_remaining":
            days_remaining,

        "spending_velocity":
            spending_velocity,

        "current_food_dining":
            category_values[0],

        "current_transportation":
            category_values[1],

        "current_shopping":
            category_values[2],

        "current_groceries":
            category_values[3],

        "current_bills_utilities":
            category_values[4],

        "current_healthcare":
            category_values[5],

        "current_entertainment":
            category_values[6],

        "current_education":
            category_values[7],

        "current_travel":
            category_values[8],

        "current_miscellaneous":
            category_values[9],

        "target_month_end_spending":
            target_month_end_spending,
    }

    return sample


# ============================================================
# GENERATE DATASET
# ============================================================

def generate_dataset():

    print(
        f"Generating {NUM_SAMPLES} training samples..."
    )

    data = []

    for _ in range(NUM_SAMPLES):

        sample = generate_sample()

        data.append(sample)


    # Convert to DataFrame.

    df = pd.DataFrame(data)


    # --------------------------------------------------------
    # ROUND NUMERIC VALUES
    # --------------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    df[numeric_columns] = df[
        numeric_columns
    ].round(2)


    # --------------------------------------------------------
    # SAVE DATASET
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # --------------------------------------------------------
    # DISPLAY INFORMATION
    # --------------------------------------------------------

    print(
        "\nDataset generated successfully!"
    )

    print(
        f"File: {OUTPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    print(
        "\nColumns:"
    )

    for column in df.columns:

        print(
            f"- {column}"
        )


    print(
        "\nFirst 5 records:"
    )

    print(
        df.head()
    )


    print(
        "\nDataset statistics:"
    )

    print(
        df.describe()
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    generate_dataset()