
# ============================================================
# GROUP ASSIGNMENT: DIABETES DATA PREPROCESSING
# Target: glyhb (Glycosylated hemoglobin concentration)
# Steps 1, 2, and 3
# ============================================================

# STEP 0: IMPORT LIBRARIES

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# STEP 1: DATA PREPROCESSING
# ============================================================

# Step 1.1: Load the saved dataset

df = pd.read_csv("data/diabetes.csv")


# Step 1.2: Inspect the first five rows

print("\n========== FIRST FIVE ROWS ==========")
print(df.head())


# Step 1.3: Check dataset shape

print("\n========== DATASET SHAPE ==========")
print(df.shape)


# Step 1.4: Check column names

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())


# Step 1.5: Check data types and non-missing values

print("\n========== DATASET INFORMATION ==========")
df.info()


# Step 1.6: Summary statistics

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe(include="all"))


# Step 1.7: Count missing values

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())


# ------------------------------------------------------------
# STEP 1.8: HANDLE MISSING VALUES IN INITIAL CLEANING
# ------------------------------------------------------------

# Make a copy of the original dataset

df_clean = df.copy()


# Remove rows where the target (glyhb) is missing

df_clean = df_clean.dropna(subset=["glyhb"])


# Remove columns with excessive missing values

df_clean = df_clean.drop(columns=["bp.2s", "bp.2d"])


# Check the result

print("\n========== AFTER INITIAL CLEANING ==========")

print("Shape after initial cleaning:", df_clean.shape)

print("\nRemaining missing values:")
print(df_clean.isnull().sum())

print("\nMissing values in target:")
print(df_clean["glyhb"].isnull().sum())


# Save the initially cleaned dataset

df_clean.to_csv("data/diabetes_cleaned.csv", index=False)

print("\nInitially cleaned dataset saved successfully!")


# ------------------------------------------------------------
# STEP 1.9: SEPARATE FEATURES AND TARGET
# ------------------------------------------------------------

# Input features: exclude target and patient ID

X = df_clean.drop(columns=["glyhb", "id"])


# Target: GlyHb concentration

y = df_clean["glyhb"]


print("\n========== FEATURES AND TARGET ==========")

print("Features shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget preview:")
print(y.head())


# Save the separated original features and target

X.to_csv("data/features.csv", index=False)
y.to_csv("data/target.csv", index=False)

print("\nFeatures and target saved successfully!")


# ------------------------------------------------------------
# STEP 1.10: CHECK MISSING VALUES IN FEATURES
# ------------------------------------------------------------

print("\n========== FEATURE MISSING VALUES ==========")
print(X.isnull().sum())

print("\n========== FEATURE DATA TYPES ==========")
print(X.dtypes)


# ============================================================
# STEP 2: TRAIN-TEST SPLIT
# ============================================================

# Split the data into 80% training and 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Display the split results

print("\n========== TRAIN-TEST SPLIT ==========")

print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# ============================================================
# STEP 3: FEATURE TRANSFORMATION
# ============================================================

# Step 3.1: Identify numerical and categorical columns

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()


print("\n========== FEATURE TYPES ==========")

print("Numerical columns:")
print(numerical_columns)

print("\nCategorical columns:")
print(categorical_columns)


# ------------------------------------------------------------
# STEP 3.2: TRANSFORM NUMERICAL FEATURES
# ------------------------------------------------------------

# Fill missing numerical values using the median.
# Then standardize numerical features.

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ------------------------------------------------------------
# STEP 3.3: TRANSFORM CATEGORICAL FEATURES
# ------------------------------------------------------------

# Fill missing categorical values using the most frequent value.
# Then convert categories into numerical columns.

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


# ------------------------------------------------------------
# STEP 3.4: COMBINE TRANSFORMATIONS
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_transformer, numerical_columns),
        ("categorical", categorical_transformer, categorical_columns)
    ]
)


# ------------------------------------------------------------
# STEP 3.5: FIT ON TRAINING DATA AND TRANSFORM
# ------------------------------------------------------------

# Learn transformation rules from training data only.

X_train_processed = preprocessor.fit_transform(X_train)

# Apply the learned rules to testing data.

X_test_processed = preprocessor.transform(X_test)


# Get the transformed feature names

feature_names = preprocessor.get_feature_names_out()


# Convert transformed arrays into DataFrames

X_train_processed = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_processed = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)


# ------------------------------------------------------------
# STEP 3.6: VERIFY TRANSFORMED DATA
# ------------------------------------------------------------

print("\n========== TRANSFORMED DATA ==========")

print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)

print("\nFirst five transformed training rows:")
print(X_train_processed.head())

print("\nRemaining missing values in training data:")
print(X_train_processed.isnull().sum().sum())

print("\nRemaining missing values in testing data:")
print(X_test_processed.isnull().sum().sum())


# ------------------------------------------------------------
# STEP 3.7: SAVE FINAL PROCESSED DATA
# ------------------------------------------------------------

# Save transformed training features

X_train_processed.to_csv(
    "data/X_train_processed.csv",
    index=False
)


# Save transformed testing features

X_test_processed.to_csv(
    "data/X_test_processed.csv",
    index=False
)


# Save corresponding target values

y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)


# Combine transformed features and target for training

train_final = X_train_processed.copy()
train_final["glyhb"] = y_train


# Combine transformed features and target for testing

test_final = X_test_processed.copy()
test_final["glyhb"] = y_test


# Save combined datasets

train_final.to_csv(
    "data/train_final_processed.csv",
    index=False
)

test_final.to_csv(
    "data/test_final_processed.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========")

print("Original dataset shape:", df.shape)
print("Initially cleaned dataset shape:", df_clean.shape)

print("Training dataset shape:", train_final.shape)
print("Testing dataset shape:", test_final.shape)

print("\nTotal missing values in training data:",
      train_final.isnull().sum().sum())

print("Total missing values in testing data:",
      test_final.isnull().sum().sum())


print("\n========== FILES SAVED SUCCESSFULLY ==========")

print("1. data/diabetes_cleaned.csv")
print("2. data/features.csv")
print("3. data/target.csv")
print("4. data/X_train_processed.csv")
print("5. data/X_test_processed.csv")
print("6. data/y_train.csv")
print("7. data/y_test.csv")
print("8. data/train_final_processed.csv")
print("9. data/test_final_processed.csv")

print("\nSteps 1, 2, and 3 completed successfully!")
