
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance

from catboost import CatBoostClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

X = pd.read_csv("data/features.csv")

y = pd.read_csv(
    "data/diabetes_classification_target.csv"
).squeeze()


# ==========================================
# 2. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 3. IDENTIFY COLUMN TYPES
# ==========================================

num_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

cat_cols = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        num_cols
    ),

    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        cat_cols
    )
])


# ==========================================
# 5. BUILD CATBOOST PIPELINE
# ==========================================

model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=4,
    auto_class_weights="Balanced",
    verbose=False,
    allow_writing_files=False,
    random_seed=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# ==========================================
# 6. TRAIN MODEL
# ==========================================

print("Training CatBoost model...")

pipeline.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 7. CALCULATE PERMUTATION IMPORTANCE
# ==========================================

print("\nCalculating feature importance...")

importance = permutation_importance(
    pipeline,
    X_test,
    y_test,
    n_repeats=30,
    random_state=42,
    scoring="roc_auc",
    n_jobs=1
)


# ==========================================
# 8. CREATE RESULTS TABLE
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# 9. DISPLAY RESULTS
# ==========================================

print("\n========== CLASSIFICATION FEATURE IMPORTANCE ==========\n")

print(importance_df.to_string(index=False))


# ==========================================
# 10. SAVE RESULTS
# ==========================================

importance_df.to_csv(
    "data/classification_feature_importance.csv",
    index=False
)

print("\nFeature importance saved successfully!")
