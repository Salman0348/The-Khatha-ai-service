from predict.expense_category import predict_category
from predict.expense_forecast_predict import predict_expense


print(
    predict_category(
        "KFC dinner with friends"
    )
)


sample_features = {

"current_month_spending":5000,
"current_month_entry_count":8,

"previous_month_spending":10000,
"previous_month_entry_count":15,

"second_last_month_spending":9000,
"third_last_month_spending":11000,

"average_last_3_months":10000,

"days_elapsed":10,
"days_in_month":31,
"days_remaining":21,

"spending_velocity":500,

"current_food_dining":2000,
"current_transportation":500,
"current_shopping":500,
"current_groceries":500,
"current_bills_utilities":500,
"current_healthcare":0,
"current_entertainment":500,
"current_education":0,
"current_travel":0,
"current_miscellaneous":500

}


print(
    predict_expense(
        sample_features
    )
)