from fastapi import FastAPI
from pydantic import BaseModel

from predict.expense_category import predict_category
from predict.expense_forecast_predict import predict_expense

app = FastAPI(
    title="TheKhatha AI Service",
    version="1.0"
)


# ==================================================
# CATEGORY REQUEST
# ==================================================

class CategoryRequest(BaseModel):
    description: str



# ==================================================
# FORECAST REQUEST
# ==================================================

class ForecastRequest(BaseModel):

    current_month_spending: float
    current_month_entry_count: int

    previous_month_spending: float
    previous_month_entry_count: int

    second_last_month_spending: float
    third_last_month_spending: float

    average_last_3_months: float

    days_elapsed: int
    days_in_month: int
    days_remaining: int

    spending_velocity: float

    current_food_dining: float
    current_transportation: float
    current_shopping: float
    current_groceries: float
    current_bills_utilities: float
    current_healthcare: float
    current_entertainment: float
    current_education: float
    current_travel: float
    current_miscellaneous: float



# ==================================================
# CATEGORY ENDPOINT
# ==================================================

@app.post("/predict-category")
def predict_category_endpoint(
    expense: CategoryRequest
):

    result = predict_category(
        expense.description
    )

    return {
        "success": True,
        "data": result
    }



# ==================================================
# FORECAST ENDPOINT
# ==================================================

@app.post("/predict-forecast")
def predict_forecast_endpoint(
    features: ForecastRequest
):

    result = predict_expense(
        features.dict()
    )

    return {
        "success": True,
        "expected_monthly_spending": result
    }



# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/")
def health_check():

    return {
        "message": "TheKhatha AI Service Running"
    }