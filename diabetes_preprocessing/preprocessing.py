
import pandas as pd

# Step 1: Load the saved dataset
df = pd.read_csv("data/diabetes.csv")

# Step 2: Inspect the first five rows
print("First five rows:")
print(df.head())

# Step 3: Check the number of rows and columns
print("\nDataset shape:")
print(df.shape)

# Step 4: Check column names
print("\nColumn names:")
print(df.columns.tolist())

# Step 5: Check data types and non-missing values
print("\nDataset information:")
df.info()

# Step 6: Summary statistics
print("\nSummary statistics:")
print(df.describe(include="all"))

# Step 7: Count missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 1: Make a copy of the original data
df_clean = df.copy()

# Step 2: Remove rows with missing target values
df_clean = df_clean.dropna(subset=["glyhb"])

# Step 3: Remove columns with excessive missing values
df_clean = df_clean.drop(columns=["bp.2s", "bp.2d"])

# Step 4: Check the result
print("Shape after initial cleaning:", df_clean.shape)

print("\nRemaining missing values:")
print(df_clean.isnull().sum())

print("\nMissing values in target:")
print(df_clean["glyhb"].isnull().sum())
df_clean.to_csv("data/diabetes_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")


# Step 5: Separate features and target

# Input features: exclude target and patient ID
X = df_clean.drop(columns=["glyhb", "id"])

# Target: GlyHb concentration
y = df_clean["glyhb"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget preview:")
print(y.head())

# Save the separated data
X.to_csv("data/features.csv", index=False)
y.to_csv("data/target.csv", index=False)

print("\nFeatures and target saved successfully!")

# Step 6: Check missing values in features

print("\nMissing values in features:")
print(X.isnull().sum())

print("\nFeature data types:")
print(X.dtypes)

# Step 7: Handle missing values

# Make a copy so we preserve the original X
X_processed = X.copy()

# Find numerical and categorical columns
numerical_columns = X_processed.select_dtypes(
    include=["number"]
).columns

categorical_columns = X_processed.select_dtypes(
    exclude=["number"]
).columns

# Fill missing numerical values with the median
for column in numerical_columns:
    X_processed[column] = X_processed[column].fillna(
        X_processed[column].median()
    )

# Fill missing categorical values with the most frequent value
for column in categorical_columns:
    X_processed[column] = X_processed[column].fillna(
        X_processed[column].mode()[0]
    )

# Verify the result
print("\nMissing values after imputation:")
print(X_processed.isnull().sum())

print("\nTotal missing values:", X_processed.isnull().sum().sum())


# Step 8: Encode categorical features

import pandas as pd

X_encoded = pd.get_dummies(
    X_processed,
    columns=["location", "gender", "frame"],
    dtype=int
)

print("\nEncoded feature data:")
print(X_encoded.head())

print("\nShape before encoding:", X_processed.shape)
print("Shape after encoding:", X_encoded.shape)

print("\nRemaining text columns:")
print(X_encoded.select_dtypes(exclude=["number"]).columns.tolist())

# Step 9: Save final processed data

X_encoded.to_csv("data/features_processed.csv", index=False)
y.to_csv("data/target.csv", index=False)

# Combine features and target for a complete processed dataset
final_df = X_encoded.copy()
final_df["glyhb"] = y

final_df.to_csv("data/diabetes_final_processed.csv", index=False)

print("\nFinal processing completed!")
print("Processed features shape:", X_encoded.shape)
print("Target shape:", y.shape)
print("Final dataset shape:", final_df.shape)
print("Missing values in final dataset:", final_df.isnull().sum().sum())
print("\nFiles saved:")
print("- data/features_processed.csv")
print("- data/target.csv")
print("- data/diabetes_final_processed.csv")

