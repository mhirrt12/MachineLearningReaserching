
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Load data
X = pd.read_csv("data/features.csv")
y = pd.read_csv("data/diabetes_classification_target.csv").squeeze()

# Identify columns
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# Models
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced", max_iter=2000
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=3,
        scale_pos_weight=5.5, eval_metric="logloss",
        random_state=42, n_jobs=1
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200, learning_rate=0.05, depth=4,
        class_weights=[1, 5.5], verbose=False,
        allow_writing_files=False, random_seed=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, learning_rate=0.05, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, random_state=42
    ),
    "SVC": SVC(
        kernel="rbf", class_weight="balanced", probability=True,
        random_state=42
    )
}

# Stratified 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, classifier in models.items():
    print("Validating:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", classifier)
    ])

    scores = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=["f1", "recall", "precision", "roc_auc"],
        n_jobs=1
    )

    results.append({
        "Model": name,
        "Mean F1": scores["test_f1"].mean(),
        "Mean Recall": scores["test_recall"].mean(),
        "Mean Precision": scores["test_precision"].mean(),
        "Mean ROC AUC": scores["test_roc_auc"].mean()
    })

results_df = pd.DataFrame(results).sort_values(
    by="Mean F1", ascending=False
)

print("\n========== CLASSIFICATION CROSS-VALIDATION ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "data/classification_cross_validation.csv", index=False
)

print("\nValidation results saved successfully!")
