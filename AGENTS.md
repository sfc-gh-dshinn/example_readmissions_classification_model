## AGENTS.md

### Agent Specifications: `readmission_model.py`

This document specifies the requirements and implementation details for a Python program, `readmission_model.py`, designed to predict patient readmission using time-series cross-validation on the `diabetic_data_with_dates.csv` dataset.

---

### 1. Data Loading and Preparation

* **Input Data:** The program must read the dataset from the file **`diabetic_data_with_dates.csv`**.
* **Date Handling:** The column containing date information is **`date`** and must be parsed as a **`datetime`** object.
* **Target Variable:** The target variable is the **`readmitted`** column.
    * **Class Definition:** The target must be converted into a binary classification problem:
        * **Negative Class (0):** Where `readmitted` is **'NO'**.
        * **Positive Class (1):** Where `readmitted` is **anything other than 'NO'** (e.g., '>30', '<30').
* **Data Sorting:** The dataset must be sorted by the **`date`** column in ascending order before splitting.

---

### 2. Date-Based Time-Series Cross-Validation

The program must implement a **`DateBasedTimeSeriesSplitter`** class that performs date-based time-series cross-validation. This approach ensures proper temporal splits even with multiple rows per date and missing dates.

#### 2.1 DateBasedTimeSeriesSplitter Class

**Purpose:** A custom splitter that generates date-based time-series cross-validation splits by working backwards from the last date in the dataset to ensure the final test partition ends exactly on the last date.

**Implementation Approach:**
1. **Anchor to End Date:** Calculate split positions working backwards from the maximum date in the dataset
2. **Calculate First Split Start:** Determine where the first training window must start to accommodate all n_splits
3. **Validate Date Range:** Ensure all partitions fit within the available data range, raising ValueError if not
4. **Generate Splits Forward:** Create n_splits going forward from the calculated starting position
5. **Filter Data by Dates:** For each fold, filter the actual data rows based on the calculated train/test date ranges

**Benefits:**
- Handles multiple rows per date correctly
- Handles missing dates in the data
- Provides true calendar-based time-series splits (e.g., 547 days = 18 months)
- Ensures final test partition ends exactly on the last date of the data
- Validates that all splits fit within the data range before execution

**Parameters:**
* `window_length` (int): Training window length in **days** (not rows). Set to **547 days** (18 months).
* `fh` (int): Forecast horizon in **days** - the gap between the end of training and start of testing. Set to **30 days**.
* `test_window_length` (int): Test window length in **days**. Set to **60 days**.
* `step_length` (int): Step size between consecutive training windows in **days**. Set to **90 days** to ensure non-overlapping test windows.
* `n_splits` (int): Number of splits to generate. Set to **5**.

**Method Signature:**
```python
def split(self, df, date_column='date'):
    """
    Generate train/test indices based on date ranges.
    
    Works backwards from the last date to ensure the final test partition
    ends exactly on the last date of the data.
    
    Yields:
        train_indices: np.array of row indices for training
        test_indices: np.array of row indices for testing
        
    Raises:
        ValueError: If all partitions do not fit within the date range of the data
    """
```

#### 2.2 Cross-Validation Configuration

* **Number of Folds:** **5** validation folds.
* **Training Window:** **547 days** (18 months of data).
* **Forecast Horizon:** **30 days** gap between train end and test start.
* **Test Window:** **60 days** (approximately 2 months).
* **Step Length:** **90 days** to ensure test windows do not overlap.

**Expected Behavior:**
- Works backwards from the last date to anchor the final test partition
- Each fold trains on 547 days (18 months) of data
- Waits 30 days (forecast horizon)
- Tests on the next 60 days
- Next fold starts 90 days after the previous fold's training start
- All test windows are non-overlapping
- Final test window ends exactly on the last date in the dataset
- Validates that all splits fit within the data range and raises ValueError if they don't

---

### 3. Feature Definition and Preprocessing

#### 3.1 Feature Lists

The program must define **hardcoded constant lists** at the top of the file to explicitly specify which features are used:

* **CATEGORICAL_FEATURES:** A list containing all categorical feature column names (36 features including demographics, diagnoses, medications, etc.)
* **NUMERICAL_FEATURES:** A list containing all numerical feature column names (12 features including patient_nbr, admission IDs, counts, etc.)

**Purpose:** This makes it crystal clear to readers which features are included in the model.

#### 3.2 Preprocessing Pipeline

The program must use **`sklearn.pipeline.Pipeline`** and **`sklearn.pipeline.FeatureUnion`** to create a complete preprocessing and modeling pipeline.

* **Categorical Features:**
    * **Selection:** Use lambda function to select columns from CATEGORICAL_FEATURES list
    * **Transformer:** Use **`sklearn.preprocessing.OrdinalEncoder`** with `handle_unknown='use_encoded_value'` and `unknown_value=-1` to encode categorical features as integers and handle unseen values gracefully.
* **Numerical Features:**
    * **Selection:** Use lambda function to select columns from NUMERICAL_FEATURES list
    * **Transformer:** **No transformation** (passthrough using `sklearn.preprocessing.FunctionTransformer(lambda x: x, validate=False)`).

**Full Pipeline Structure:**
```python
preprocessing = FeatureUnion([
    ('categorical', Pipeline([
        ('selector', FunctionTransformer(lambda X: X[CATEGORICAL_FEATURES].values, validate=False)),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])),
    ('numerical', Pipeline([
        ('selector', FunctionTransformer(lambda X: X[NUMERICAL_FEATURES].values, validate=False)),
        ('passthrough', FunctionTransformer(lambda x: x, validate=False))
    ]))
])

full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', XGBClassifier())
])
```

**Critical:** The full pipeline ensures preprocessing is fit only on training data within each cross-validation fold, preventing data leakage.

---

### 4. Model Training and Evaluation

#### 4.1 Cross-Validation Execution

* **Method:** Use **`sklearn.model_selection.cross_validate`** (not manual loop) to execute cross-validation.
* **Model:** **`xgboost.XGBClassifier`** (using the scikit-learn interface) within the pipeline.
* **Parameters:** **Default parameters** for the `XGBClassifier` should be used.
* **Early Stopping:** **No early stopping** should be implemented.
* **Scoring:** The primary performance metric must be **ROC AUC** using `sklearn.metrics.make_scorer(roc_auc_score, needs_proba=True)`.
* **Return Values:** Set `return_estimator=True` to retrieve trained pipeline objects for each fold.

**Implementation Pattern:**
```python
cv_results = cross_validate(
    estimator=full_pipeline,
    X=X,  # Raw data (not pre-transformed)
    y=y,
    cv=splits,  # List of (train_idx, test_idx) tuples
    scoring={'roc_auc': make_scorer(roc_auc_score, needs_proba=True)},
    return_estimator=True,
    return_train_score=False,
    verbose=1
)
```

#### 4.2 Performance Metrics

The program must output the following performance metrics:

**1. AUC Scores by Fold:**
- Print AUC score for each of the 5 validation folds
- Calculate and print average AUC and standard deviation

**2. Classification Report:**
- Consolidate predictions across all folds
- Generate `sklearn.metrics.classification_report` with target names ['No Readmission', 'Readmission']
- Display precision, recall, f1-score, and support for each class

**3. Confusion Matrix:**
- Generate `sklearn.metrics.confusion_matrix` from consolidated predictions
- Display formatted confusion matrix with row/column labels
- Include interpretation: True Negatives, False Positives, False Negatives, True Positives

**4. Lift Analysis by Fold:**
For each fold, generate a lift table showing:
- **Decile:** Risk decile (10 = highest risk, 1 = lowest risk)
- **Min Score:** Minimum predicted probability in decile
- **Max Score:** Maximum predicted probability in decile
- **Count:** Number of patients in decile
- **Readmit Rate:** Actual readmission rate in decile
- **Lift:** Ratio of decile readmission rate to overall rate
- **Cum Recall:** Cumulative percentage of all readmissions captured through this decile (from top down)

**Implementation Notes:**
- Sort predictions by probability descending
- Use `pd.qcut()` with `q=10` to create deciles
- Calculate cumulative recall by tracking cumulative sum of readmissions from highest to lowest decile

**5. Feature Importance:**
- Extract feature importances from each trained model using `est.named_steps['model'].feature_importances_`
- Average feature importances across all 5 folds
- **Normalize** importances by dividing by the maximum value (top feature = 1.0, others as percentage of top)
- Display:
  - Top 20 features in table format
  - All features sorted by importance (descending) with original column names

---

### 5. Excluded Features

The following columns must be **excluded** from the feature set:
* `readmitted` (target variable)
* `target` (derived binary target)
* `date` (temporal identifier used for splitting)
* `encounter_id` (unique identifier, not a predictive feature)

---

### 6. Implementation Summary

**File Structure:**
1. Import statements
2. Hardcoded feature lists (CATEGORICAL_FEATURES, NUMERICAL_FEATURES)
3. `DateBasedTimeSeriesSplitter` class definition
4. Main execution logic:
   - Load and prepare data
   - Initialize date-based splitter with n_splits=5
   - Create preprocessing and model pipeline
   - Generate cross-validation splits using list(splitter.split())
   - Enumerate splits to display date ranges
   - Execute `cross_validate` with full pipeline
   - Generate lift analysis for each fold (with predictions and probabilities)
   - Consolidate all predictions across folds
   - Output performance metrics:
     - AUC scores by fold
     - Classification report
     - Confusion matrix
     - Feature importance (normalized)

**Key Libraries:**
- pandas, numpy
- sklearn.pipeline, sklearn.preprocessing, sklearn.model_selection, sklearn.metrics
- xgboost
