
# STEP 5A: CROSS-VALIDATION MODEL SELECTION

import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
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


# 1. Load original separated data

X = pd.read_csv("data/features.csv")
y = pd.read_csv("data/target.csv").squeeze()


# 2. Split data exactly as before

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)


# 3. Identify numerical and categorical columns

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()


# 4. Create preprocessing pipeline

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_transformer, numerical_columns),
    ("categorical", categorical_transformer, categorical_columns)
])


# 5. Define all 12 regression models

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(alpha=0.01),
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        random_state=42, objective="reg:squarederror", n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=200, learning_rate=0.05, depth=4,
        random_seed=42, verbose=False, allow_writing_files=False
    ),
    "AdaBoost": AdaBoostRegressor(
        n_estimators=100, learning_rate=0.05, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, random_state=42
    ),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1)
}


# 6. Perform 5-fold cross-validation

results = []

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

for name, model in models.items():

    print("\nEvaluating:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=1
    )

    mean_rmse = -scores["test_rmse"].mean()
    std_rmse = scores["test_rmse"].std()
    mean_r2 = scores["test_r2"].mean()

    results.append({
        "Model": name,
        "Mean RMSE": mean_rmse,
        "RMSE Std": std_rmse,
        "Mean R2": mean_r2
    })

    print("Mean RMSE:", round(mean_rmse, 4))
    print("RMSE Std:", round(std_rmse, 4))
    print("Mean R2:", round(mean_r2, 4))


# 7. Compare models

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Mean RMSE"
).reset_index(drop=True)

print("\n========== CROSS-VALIDATION RESULTS ==========")
print(results_df.to_string(index=False))


# 8. Identify the best model

best_model = results_df.iloc[0]["Model"]

print("\nBest model based on cross-validation:", best_model)


# 9. Save results

results_df.to_csv(
    "data/cross_validation_results.csv",
    index=False
)

print("\nCross-validation results saved successfully!")
