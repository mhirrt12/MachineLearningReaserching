Diabetes Prediction Using Machine Learning Algorithms
1. Abstract
This study applies machine learning techniques to a diabetes-related
dataset to investigate two prediction tasks: regression and
classification. The regression task predicts Glycosylated Hemoglobin
(GlyHb) percentages, while the classification task predicts a
diabetes-related category derived from GlyHb values.

The original dataset contained 403 records and 19 variables. After
preprocessing, 390 records and 15 predictor features were used. Multiple
machine learning algorithms were trained and compared. Lasso Regression
was selected for the regression task based on cross-validation
performance, while CatBoost was selected for classification.

The final regression model achieved an R² score of 0.5073. The
classification model achieved 89.74% accuracy and a 71.43% F1 score on
the test dataset. Feature-importance analysis identified stabilized
glucose as the most important predictive feature in both tasks.
2. Introduction
Diabetes is a chronic health condition that affects how the body
regulates blood glucose. Machine learning can help researchers analyze
health-related data and develop predictive models that identify patterns
associated with diabetes.

This research uses a diabetes dataset to compare different machine
learning algorithms and determine which models provide the best
predictive performance.

The study focuses on two approaches:
- Regression: Predicting GlyHb concentration percentages.
- Classification: Predicting a diabetes-related category created
  from GlyHb values.
3. Research Questions
Regression Research Questions
1. Which prediction model provides the best performance for predicting
   GlyHb percentages?
2. What are the key factors associated with GlyHb concentration?
Classification Research Questions
1. Which classification model provides the best performance for
   predicting the diabetes-related category?
2. What are the important predictive factors associated with the
   diabetes-related category?
4. Dataset Description
The dataset was obtained from:

Diabetes dataset

The original dataset contained:
- 403 records.
- 19 columns.
- Numerical and categorical variables.
- GlyHb as the main target variable.
The dataset includes variables such as age, gender, cholesterol,
stabilized glucose, height, weight, blood pressure, waist, and hip
measurements.

After removing records with missing GlyHb values and removing the two
blood pressure columns with excessive missing values (bp.2s and
bp.2d), 390 records and 17 columns remained.

The id column was excluded because it is an identifier rather than a
meaningful predictive feature.
5. Data Preprocessing
Data preprocessing was performed before training the machine learning
models.

The following steps were applied:
1. Data loading: The diabetes dataset was loaded using Pandas.
2. Missing-value handling: Records without GlyHb values were
   removed. Numerical missing values were handled using median
   imputation, while categorical missing values were handled using the
   most frequent value.
3. Feature selection: The id column was removed, and GlyHb was
   separated as the regression target.
4. Train-test splitting: The data was divided into 80% training
   data and 20% testing data.
5. Feature transformation: Numerical features were standardized
   using StandardScaler, while categorical features were converted
   into numerical representations using OneHotEncoder.
6. Data leakage prevention: Preprocessing transformations were
   fitted using training data only and then applied to testing data.
The final dataset contained 390 records and 15 predictor features. The
training set contained 312 records, while the testing set contained 78
records.
6. Regression Analysis
6.1 Regression Models
Twelve regression algorithms were trained and compared:
1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. ElasticNet Regression
5. K-Nearest Neighbors Regression
6. Decision Tree Regression
7. Random Forest Regression
8. XGBoost Regression
9. CatBoost Regression
10. AdaBoost Regression
11. Gradient Boosting Regression
12. Support Vector Regression
6.2 Model Validation
Five-fold cross-validation was used to compare the regression models.

Lasso Regression achieved the lowest mean cross-validation RMSE of
1.4054 and the highest mean cross-validation R² of 0.5426.

Therefore, Lasso Regression was selected as the final regression model.
6.3 Final Regression Evaluation
The selected Lasso Regression model was evaluated using the independent
test dataset.

  Evaluation Metric     Result
  MAE                   1.0148
MSE                   3.2851
RMSE                  1.8125
R²                    0.5073

The R² score of 0.5073 indicates that the model explains approximately
50.73% of the variation in GlyHb values in the test dataset. These
results indicate moderate predictive performance.
6.4 Regression Feature Importance
Permutation importance was used to identify the features that
contributed most to predicting GlyHb.

  Feature                             Importance
  Stabilized glucose (stab.glu)       1.079570
Age                                   0.044506
Gender                                0.019666
Cholesterol (chol)                  0.007126
Ratio                                 0.005199

Stabilized glucose was the strongest predictive feature. Its importance
was considerably higher than that of the other variables. This suggests
that stabilized glucose provides the most useful predictive information
for estimating GlyHb in this dataset.
7. Classification Analysis
7.1 Classification Target Creation
The original dataset did not contain a separate diabetes diagnosis
column. Therefore, a binary classification target was created from GlyHb
values using this project-defined rule:
- GlyHb greater than 7.0%: Class 1 (positive category).
- GlyHb less than or equal to 7.0%: Class 0 (negative category).
The resulting class distribution was:

  Category              Records   Percentage
  Negative category         330       84.62%
Positive category          60       15.38%
Total             390     100%

The dataset was imbalanced because the positive category contained fewer
records. Stratified splitting and class-balancing techniques were used
to help address this imbalance.
Important: This is a GlyHb-derived classification target, not an
independently recorded clinical diagnosis. The chosen 7.0% threshold
is a project-defined rule and should not be presented as a confirmed
diagnosis.

7.2 Classification Models
Nine classification algorithms were trained and compared:
1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. XGBoost
6. CatBoost
7. AdaBoost
8. Gradient Boosting
9. Support Vector Classifier
7.3 Model Validation
Five-fold stratified cross-validation was performed using F1 score,
recall, precision, and ROC-AUC.

  Model                      Mean F1   Mean Recall   Mean Precision   Mean ROC-AUC
  CatBoost                0.7409    0.8167           0.6878         0.9260
Random Forest               0.7195        0.7000           0.7514     0.9277
XGBoost                     0.7177        0.7667           0.6788         0.8928
AdaBoost                    0.6795        0.5833           0.8280         0.9191
Logistic Regression         0.6713        0.8000           0.5869         0.9119
SVC                         0.6587        0.6667           0.6816         0.8942
Gradient Boosting           0.6467        0.6000           0.7257         0.9106
Decision Tree               0.6406        0.7167           0.5971         0.8256
KNN                         0.5812        0.4333           0.8914         0.8019

CatBoost was selected as the final classification model because it
achieved the highest mean F1 score and recall. Random Forest had a
slightly higher mean ROC-AUC, so CatBoost was not the winner on every
metric.
7.4 Final Classification Evaluation
The selected CatBoost model was evaluated on the independent test
dataset.

  Evaluation Metric     Result
  Accuracy              89.74%
Precision             62.50%
Recall                83.33%
F1 Score              71.43%
ROC-AUC               93.69%

The model correctly classified 70 out of 78 test records.
7.5 Confusion Matrix
The confusion matrix was:
                  Predicted Negative   Predicted Positive

  Actual Negative                     60                    6
Actual Positive                      2                   10

The model correctly identified 60 negative records and 10 positive
records. It incorrectly classified 6 negative records as positive and
missed 2 positive records.

The recall score of 83.33% indicates that the model identified 10 out of
12 positive records in the test dataset.
7.6 Classification Feature Importance
Permutation importance was used to identify the features that
contributed most to classification performance.

  Feature                             Importance
  Stabilized glucose (stab.glu)       0.372433
HDL cholesterol (hdl)               0.002104
Height                                0.000253
Hip                                  -0.000084
Cholesterol (chol)                 -0.001221
Location                             -0.001305
Ratio                                -0.001473
Blood pressure (bp.1d)             -0.001641
Frame                                -0.002483
Time after meal (time.ppn)         -0.003577
Weight                               -0.003662
Waist                                -0.003746
Gender                               -0.003830
Blood pressure (bp.1s)             -0.003872
Age                                  -0.006397

Stabilized glucose was the most important predictive feature, with an
importance score of 0.372433. HDL cholesterol and height had very small
positive importance values. Most other variables had values close to
zero or slightly negative.

This suggests that stabilized glucose contributed most to the
classification model's predictive performance.
8. Discussion
The regression and classification experiments identified stabilized
glucose as the strongest predictive feature.

For regression, Lasso Regression was selected based on its
cross-validation performance, achieving a test R² of 0.5073.

For classification, CatBoost provided the strongest overall
cross-validation performance, achieving a mean F1 score of 0.7409. Its
final test accuracy was 89.74%, with a recall of 83.33%.

The classification results demonstrate that accuracy alone is not
sufficient for evaluating models on imbalanced datasets. Precision,
recall, F1 score, and ROC-AUC provide additional information about model
performance.

The feature-importance results suggest that stabilized glucose provides
substantially more predictive information than the other available
variables in both experiments.

These findings describe predictive relationships within this dataset.
They do not establish that a particular feature causes diabetes.
9. Limitations
The study has several limitations:
1. The dataset contains only 403 original records, which limits the
   amount of information available for training and evaluation.
2. The classification target was derived from GlyHb values rather than
   obtained from an independently recorded diabetes diagnosis.
3. The classification dataset is imbalanced, with fewer positive
   records.
4. Some original variables contained many missing values and had to be
   excluded.
5. The models were evaluated using one train-test split, so results may
   vary with different data samples.
6. Feature importance identifies predictive contribution, not medical
   causation.
Therefore, the models should be considered research experiments rather
than clinical diagnostic systems.
10. Conclusion
This research compared multiple machine learning algorithms for
predicting GlyHb percentages and a GlyHb-derived diabetes-related
category.

For regression, Lasso Regression was selected as the final model,
achieving an R² score of 0.5073 and an RMSE of 1.8125.

For classification, CatBoost was selected as the final model, achieving
89.74% accuracy, 71.43% F1 score, and 93.69% ROC-AUC on the test
dataset.

Feature-importance analysis identified stabilized glucose as the
strongest predictive feature in both regression and classification.

Overall, the study demonstrates how machine learning algorithms can be
used to analyze health-related data, compare predictive performance, and
identify important predictive features. However, further research using
larger datasets and independently confirmed diabetes diagnoses is
necessary before making clinical conclusions.
11. References
1. Diabetes dataset: https://hbiostat.org/data/repo/diabetes.csv
2. Scikit-learn documentation: https://scikit-learn.org/stable/
3. XGBoost documentation: https://xgboost.readthedocs.io/
4. CatBoost documentation: https://catboost.ai/docs/
5. Pandas documentation: https://pandas.pydata.org/docs/