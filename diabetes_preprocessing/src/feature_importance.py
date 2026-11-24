
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.inspection import permutation_importance

# Load data
X = pd.read_csv("data/features.csv")
y = pd.read_csv("data/target.csv").squeeze()

# Same train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# Identify column types
numerical_columns = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_columns = X_train.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessing
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

# Build and train final Lasso model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", Lasso(alpha=0.01))
])

model.fit(X_train, y_train)

# Calculate permutation importance
importance = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

# Create results table
results = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
})

results = results.sort_values(
    by="Importance", ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(results.to_string(index=False))

# Save results
results.to_csv("data/feature_importance.csv", index=False)

print("\nFeature importance saved successfully!")
