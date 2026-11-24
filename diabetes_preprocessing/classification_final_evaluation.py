
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from catboost import CatBoostClassifier


# ==========================================
# 1. LOAD DATA
# ==========================================

X = pd.read_csv("data/features.csv")

y = pd.read_csv(
    "data/diabetes_classification_target.csv"
).squeeze()

print("Dataset loaded successfully!")


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

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


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
# 5. SELECTED MODEL: CATBOOST
# ==========================================

catboost_model = CatBoostClassifier(
    iterations=200,
    learning_rate=0.05,
    depth=4,
    auto_class_weights="Balanced",
    verbose=False,
    allow_writing_files=False,
    random_seed=42
)


# ==========================================
# 6. CREATE AND TRAIN PIPELINE
# ==========================================

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", catboost_model)
])

print("\nTraining final CatBoost model...")

final_pipeline.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 7. MAKE PREDICTIONS
# ==========================================

y_pred = final_pipeline.predict(X_test).ravel()

y_prob = final_pipeline.predict_proba(X_test)[:, 1]


# ==========================================
# 8. EVALUATE MODEL PERFORMANCE
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

roc_auc = roc_auc_score(y_test, y_prob)


print("\n========== FINAL CATBOOST EVALUATION ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")


# ==========================================
# 9. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Not Diabetic", "Diabetic"],
        zero_division=0
    )
)


# ==========================================
# 10. SAVE PREDICTIONS
# ==========================================

predictions_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Probability": y_prob
})

predictions_df.to_csv(
    "data/final_catboost_predictions.csv",
    index=False
)


# ==========================================
# 11. SAVE METRICS
# ==========================================

metrics_df = pd.DataFrame([{
    "Model": "CatBoost",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "ROC AUC": roc_auc
}])

metrics_df.to_csv(
    "data/final_catboost_metrics.csv",
    index=False
)

print("\nFinal evaluation files saved successfully!")
