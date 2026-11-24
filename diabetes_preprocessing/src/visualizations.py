
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 1. Define project folders
DATA_DIR = Path("data")
FIGURES_DIR = Path("results") / "figures"

# Create the figures folder if it does not exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load the CSV result files
regression_results = pd.read_csv(
    DATA_DIR / "regression_model_results.csv"
)

classification_results = pd.read_csv(
    DATA_DIR / "classification_cross_validation.csv"
)

regression_importance = pd.read_csv(
    DATA_DIR / "feature_importance.csv"
)

classification_importance = pd.read_csv(
    DATA_DIR / "classification_feature_importance.csv"
)


# 3. Regression model comparison (RMSE)
regression_results = regression_results.sort_values("RMSE")

plt.figure(figsize=(11, 7))

plt.barh(
    regression_results["Model"],
    regression_results["RMSE"]
)

plt.xlabel("RMSE (lower is better)")
plt.ylabel("Regression Models")
plt.title("Regression Model Comparison - RMSE")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "regression_model_comparison.png",
    dpi=300
)
plt.close()


# 4. Classification model comparison (F1-score)
classification_results = classification_results.sort_values(
    "Mean F1"
)

plt.figure(figsize=(11, 7))

plt.barh(
    classification_results["Model"],
    classification_results["Mean F1"]
)

plt.xlabel("Mean F1-score (higher is better)")
plt.ylabel("Classification Models")
plt.title("Classification Model Comparison - F1-score")
plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "classification_model_comparison.png",
    dpi=300
)
plt.close()


# 5. Regression feature importance
regression_importance = regression_importance.sort_values(
    "Importance"
)

plt.figure(figsize=(11, 7))

plt.barh(
    regression_importance["Feature"],
    regression_importance["Importance"],
    xerr=regression_importance["Std"]
)

plt.xlabel("Permutation Importance (RMSE improvement)")
plt.ylabel("Features")
plt.title("Regression - Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "regression_feature_importance.png",
    dpi=300
)
plt.close()


# 6. Classification feature importance
classification_importance = classification_importance.sort_values(
    "Importance"
)

plt.figure(figsize=(11, 7))

plt.barh(
    classification_importance["Feature"],
    classification_importance["Importance"],
    xerr=classification_importance["Std"]
)

plt.xlabel("Permutation Importance (ROC-AUC improvement)")
plt.ylabel("Features")
plt.title("Classification - Feature Importance")
plt.tight_layout()

plt.savefig(
    FIGURES_DIR / "classification_feature_importance.png",
    dpi=300
)
plt.close()


print("All four charts were created successfully!")

print(f"Charts saved in: {FIGURES_DIR.resolve()}")
