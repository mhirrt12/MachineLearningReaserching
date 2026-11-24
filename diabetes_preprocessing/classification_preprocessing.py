
import pandas as pd

# Load features and GlyHb target
X = pd.read_csv("data/features.csv")
glyhb = pd.read_csv("data/target.csv").squeeze()

# Create binary classification target
y_class = (glyhb > 7.0).astype(int)

# Check class distribution
print("\n========== CLASS DISTRIBUTION ==========")
print(y_class.value_counts().rename({
    0: "Not Diabetic",
    1: "Diabetic"
}))

print("\n========== CLASS PERCENTAGES ==========")
print(
    (y_class.value_counts(normalize=True) * 100)
    .rename({0: "Not Diabetic", 1: "Diabetic"})
    .round(2)
)

# Save classification target
y_class.to_csv("data/diabetes_classification_target.csv", index=False)

print("\nClassification target saved successfully!")
