
# ============================================================
# GROUP ASSIGNMENT: DIABETES REGRESSION MODEL TRAINING
# Step 4: Train Candidate Models and Evaluate
# Target: glyhb
# ============================================================

# STEP 1: IMPORT LIBRARIES

import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

from sklearn.svm import SVR

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STEP 2: LOAD TRAINING AND TESTING DATA
# ============================================================

X_train = pd.read_csv("data/X_train_processed.csv")
X_test = pd.read_csv("data/X_test_processed.csv")

y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

print("\n========== DATA LOADED ==========")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# STEP 3: DEFINE CANDIDATE REGRESSION MODELS
# ============================================================

models = {

    "Linear Regression": LinearRegression(),

    "Lasso Regression": Lasso(alpha=0.01),

    "Ridge Regression": Ridge(alpha=1.0),

    "ElasticNet Regression": ElasticNet(
        alpha=0.01,
        l1_ratio=0.5,
        max_iter=10000
    ),

    "K-Nearest Neighbors": KNeighborsRegressor(
        n_neighbors=5
    ),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1
    ),

    "CatBoost": CatBoostRegressor(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        random_seed=42,
        verbose=False,
        allow_writing_files=False
    ),

    "AdaBoost": AdaBoostRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Support Vector Regression": SVR(
        kernel="rbf",
        C=1.0,
        epsilon=0.1
    )
}


# ============================================================
# STEP 4: TRAIN MODELS AND EVALUATE PERFORMANCE
# ============================================================

results = []

predictions = {}

for model_name, model in models.items():

    print("\nTraining:", model_name)

    # Train the model using training data

    model.fit(X_train, y_train)

    # Predict using testing data

    y_pred = model.predict(X_test)

    # Store predictions

    predictions[model_name] = y_pred

    # Calculate evaluation metrics

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    # Store results

    results.append({
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print("MAE:", round(mae, 4))
    print("MSE:", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))


# ============================================================
# STEP 5: COMPARE ALL MODELS
# ============================================================

results_df = pd.DataFrame(results)

# Sort models by RMSE (lowest error first)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
).reset_index(drop=True)


print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))


# ============================================================
# STEP 6: SELECT THE BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

print("\n========== BEST MODEL ==========")

print("Best model based on test RMSE:", best_model_name)

print("MAE:", round(results_df.iloc[0]["MAE"], 4))
print("MSE:", round(results_df.iloc[0]["MSE"], 4))
print("RMSE:", round(results_df.iloc[0]["RMSE"], 4))
print("R2:", round(results_df.iloc[0]["R2"], 4))


# ============================================================
# STEP 7: SAVE RESULTS
# ============================================================

results_df.to_csv(
    "data/regression_model_results.csv",
    index=False
)

print("\nModel comparison saved successfully!")

# Save the best model's predictions

best_predictions = pd.DataFrame({
    "Actual_GlyHb": y_test.values,
    "Predicted_GlyHb": predictions[best_model_name]
})

best_predictions.to_csv(
    "data/best_model_predictions.csv",
    index=False
)

print("Best model predictions saved successfully!")

print("\n========== STEP 4 COMPLETED ==========")
