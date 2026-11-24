
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load original features and target
X = pd.read_csv("data/features.csv")
y = pd.read_csv("data/target.csv").squeeze()

# 2. Split the data (same split used in validation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 3. Identify column types
numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()

# 4. Prepare preprocessing
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

# 5. Build the selected model pipeline
final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Lasso(alpha=0.01))
])

# 6. Train using training data
final_model.fit(X_train, y_train)

# 7. Predict test data
y_pred = final_model.predict(X_test)

# 8. Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== FINAL LASSO TEST RESULTS ==========")
print(f"MAE:  {mae:.4f}")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

# 9. Save predictions
predictions = pd.DataFrame({
    "Actual_GlyHb": y_test.values,
    "Predicted_GlyHb": y_pred
})

predictions.to_csv("data/final_lasso_predictions.csv", index=False)

print("\nFinal predictions saved successfully!")
