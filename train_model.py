import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/student_marks.csv"
MODEL_DIR = "model"

df = pd.read_csv(DATA_PATH)

features = [
    "Study_Hours",
    "Attendance",
    "Previous_Marks",
    "Assignment_Score",
    "Internal_Test_Score"
]
target = "Final_Marks"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

results_df = pd.DataFrame(results)
print("\nModel Comparison")
print(results_df.round(3).to_string(index=False))

# Select the model with the lowest RMSE.
best_name = results_df.loc[results_df["RMSE"].idxmin(), "Model"]
best_model = models[best_name]

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(best_model, os.path.join(MODEL_DIR, "student_marks_model.pkl"))

with open(os.path.join(MODEL_DIR, "model_info.txt"), "w", encoding="utf-8") as f:
    f.write(f"Selected Model: {best_name}\n")
    f.write("Features: " + ", ".join(features) + "\n")
    f.write("Target: Final_Marks\n")

print(f"\nSaved model: {best_name}")
print("Model file: model/student_marks_model.pkl")
