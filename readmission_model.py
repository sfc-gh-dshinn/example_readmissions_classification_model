import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, make_scorer, classification_report, confusion_matrix
from xgboost import XGBClassifier

DATE_COLUMN = 'date'
DATA_FILE = 'diabetic_data_with_dates.csv'
TARGET_COLUMN = 'readmitted'
WINDOW_LENGTH = 547  # 18 months (18 * 30.42 ≈ 547 days)
FORECAST_HORIZON = 30
TEST_WINDOW_LENGTH = 60
STEP_LENGTH = 90
N_SPLITS = 5
NEGATIVE_CLASS_NAME = 'No Readmission'
POSITIVE_CLASS_NAME = 'Readmission'


def load_data(file_path, date_column, target_column):
    """
    Load and preprocess the dataset.

    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing the data
    date_column : str
        Name of the date column to parse
    target_column : str
        Name of the target column to convert to binary

    Returns:
    --------
    df : pd.DataFrame
        Loaded and preprocessed dataframe with:
        - Parsed date column
        - Binary target variable
        - Sorted by date in ascending order
    """
    print("Loading data...")
    df = pd.read_csv(file_path)

    print("Parsing date column...")
    df[date_column] = pd.to_datetime(df[date_column])

    print("Converting target variable to binary...")
    df['target'] = (df[target_column] != 'NO').astype(int)

    print("Sorting by date...")
    df = df.sort_values(date_column).reset_index(drop=True)

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df[date_column].min()} to {df[date_column].max()}")
    print(f"Target distribution: {df['target'].value_counts().to_dict()}")

    return df


CATEGORICAL_FEATURES = [
    'A1Cresult',
    'acarbose',
    'acetohexamide',
    'age',
    'change',
    'chlorpropamide',
    'citoglipton',
    'diabetesMed',
    'diag_1',
    'diag_2',
    'diag_3',
    'examide',
    'gender',
    'glimepiride',
    'glimepiride-pioglitazone',
    'glipizide',
    'glipizide-metformin',
    'glyburide',
    'glyburide-metformin',
    'insulin',
    'max_glu_serum',
    'medical_specialty',
    'metformin',
    'metformin-pioglitazone',
    'metformin-rosiglitazone',
    'miglitol',
    'nateglinide',
    'payer_code',
    'pioglitazone',
    'race',
    'repaglinide',
    'rosiglitazone',
    'tolazamide',
    'tolbutamide',
    'troglitazone',
    'weight',
]

NUMERICAL_FEATURES = [
    'admission_source_id',
    'admission_type_id',
    'discharge_disposition_id',
    'num_lab_procedures',
    'num_medications',
    'num_procedures',
    'number_diagnoses',
    'number_emergency',
    'number_inpatient',
    'number_outpatient',
    'time_in_hospital',
]

feature_names = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Below this line is unnecessary to edit to make it compatible with
# another dataset

class DateBasedTimeSeriesSplitter:
    """
    A time-series cross-validator that splits data based on actual date ranges.

    This splitter works backwards from the last date in the dataset to ensure
    the final test partition ends exactly on the last date. It then generates
    the specified number of splits with proper temporal ordering.

    This approach handles:
    - Multiple rows per date
    - Missing dates in the data
    - Proper time-based train/test splits
    - Ensures full data coverage by anchoring to the end date
    """

    def __init__(self, window_length, fh, test_window_length, step_length, n_splits=5):
        """
        Parameters:
        -----------
        window_length : int
            Training window length in days (e.g., 365 for 1 year)
        fh : int
            Forecast horizon in days (gap between train end and test start)
        test_window_length : int
            Test window length in days
        step_length : int
            Step size between consecutive training windows in days
        n_splits : int, default=5
            Number of splits to generate
        """
        self.window_length = window_length
        self.fh = fh
        self.test_window_length = test_window_length
        self.step_length = step_length
        self.n_splits = n_splits

    def split(self, df, date_column='date'):
        """
        Generate train/test indices based on date ranges.

        Works backwards from the last date to ensure the final test partition
        ends exactly on the last date of the data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing the data with a date column
        date_column : str
            Name of the date column

        Yields:
        -------
        train_indices : np.array
            Indices for training data
        test_indices : np.array
            Indices for testing data

        Raises:
        -------
        ValueError:
            If all partitions do not fit within the date range of the data
        """
        dates = pd.to_datetime(df[date_column])
        min_date = dates.min().normalize()
        max_date = dates.max().normalize()

        # Calculate minimum required date range for all splits
        # Final fold needs: window_length + fh + test_window_length
        # Each previous fold adds: step_length
        required_days = self.window_length + self.fh + self.test_window_length + self.step_length * (self.n_splits - 1)
        available_days = (max_date - min_date).days + 1

        if available_days < required_days:
            raise ValueError(
                f"Cannot fit {self.n_splits} splits within the available data. "
                f"Required: {required_days} days, Available: {available_days} days. "
                f"Reduce n_splits, window_length, test_window_length, or step_length."
            )

        # Calculate where the final test window should end (on max_date)
        final_test_end = max_date
        final_test_start = final_test_end - pd.Timedelta(days=self.test_window_length - 1)

        # Calculate where the final training window should end (before forecast horizon)
        final_train_end = final_test_start - pd.Timedelta(days=self.fh)
        final_train_start = final_train_end - pd.Timedelta(days=self.window_length - 1)

        # Calculate starting position for first split (working backwards)
        first_train_start = final_train_start - pd.Timedelta(days=self.step_length * (self.n_splits - 1))

        # Verify first split starts within or after the data range
        if first_train_start < min_date:
            raise ValueError(
                f"First training window starts at {first_train_start.date()}, "
                f"which is before the data starts at {min_date.date()}. "
                f"Cannot fit {self.n_splits} splits. Reduce n_splits or step_length."
            )

        # Generate splits going forward from calculated starting position
        for i in range(self.n_splits):
            train_start = first_train_start + pd.Timedelta(days=self.step_length * i)
            train_end = train_start + pd.Timedelta(days=self.window_length - 1)

            test_start = train_end + pd.Timedelta(days=self.fh)
            test_end = test_start + pd.Timedelta(days=self.test_window_length - 1)

            # Filter data by date ranges
            train_mask = (dates >= train_start) & (dates <= train_end)
            test_mask = (dates >= test_start) & (dates <= test_end)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


def generate_classification_report(y_true, y_pred, negative_class_name='Negative', positive_class_name='Positive'):
    """
    Generate and print a classification report for binary classification model evaluation.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
    negative_class_name : str, default='Negative'
        Name of the negative class (class 0) for display purposes
    positive_class_name : str, default='Positive'
        Name of the positive class (class 1) for display purposes

    Returns:
    --------
    None (prints the classification report)
    """
    print(f"{'='*60}")
    print(f"CLASSIFICATION REPORT")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=[negative_class_name, positive_class_name]))


def generate_confusion_matrix(y_true, y_pred, negative_class_name='Negative', positive_class_name='Positive'):
    """
    Generate and print a confusion matrix for binary classification model evaluation.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred : array-like
        Predicted binary labels (0 or 1)
    negative_class_name : str, default='Negative'
        Name of the negative class (class 0) for display purposes
    positive_class_name : str, default='Positive'
        Name of the positive class (class 1) for display purposes

    Returns:
    --------
    None (prints the confusion matrix and interpretation)
    """
    print(f"{'='*60}")
    print(f"CONFUSION MATRIX")
    print(f"{'='*60}")
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted {negative_class_name:<15} Predicted {positive_class_name}")
    print(f"Actual {negative_class_name:<13} {cm[0, 0]:8d}        {cm[0, 1]:8d}")
    print(f"Actual {positive_class_name:<13} {cm[1, 0]:8d}        {cm[1, 1]:8d}")
    print(f"\nInterpretation:")
    print(f"True Negatives (TN):  {cm[0, 0]:,} - Correctly predicted {negative_class_name.lower()}")
    print(f"False Positives (FP): {cm[0, 1]:,} - Incorrectly predicted {positive_class_name.lower()}")
    print(f"False Negatives (FN): {cm[1, 0]:,} - Incorrectly predicted {negative_class_name.lower()}")
    print(f"True Positives (TP):  {cm[1, 1]:,} - Correctly predicted {positive_class_name.lower()}")


def generate_lift_table(y_true, y_pred_proba, fold_num=None, positive_class_name='Positive'):
    """
    Generate and print a lift table for binary classification model evaluation.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    fold_num : int, optional
        Fold number for display purposes (e.g., in cross-validation)
    positive_class_name : str, default='Positive'
        Name of the positive class for display purposes

    Returns:
    --------
    None (prints the lift table)
    """
    # Create dataframe for lift analysis
    lift_df = pd.DataFrame({
        'actual': y_true,
        'predicted_proba': y_pred_proba
    })

    # Sort by predicted probability descending
    lift_df = lift_df.sort_values('predicted_proba', ascending=False).reset_index(drop=True)

    # Create deciles
    lift_df['decile'] = pd.qcut(lift_df['predicted_proba'], q=10, labels=False, duplicates='drop') + 1

    # Calculate lift statistics by decile
    fold_text = f"Fold {fold_num} " if fold_num is not None else ""
    print(f"\n{fold_text}Lift Table:")
    print(f"{'Decile':<8} {'Min Score':<12} {'Max Score':<12} {'Count':<8} {positive_class_name + ' Rate':<15} {'Lift':<8} {'Cum Recall':<12}")
    print("-" * 87)

    overall_rate = lift_df['actual'].mean()
    total_positive = lift_df['actual'].sum()
    cumulative_positive = 0

    for decile in sorted(lift_df['decile'].unique(), reverse=True):
        decile_data = lift_df[lift_df['decile'] == decile]
        count = len(decile_data)
        decile_positive = decile_data['actual'].sum()
        cumulative_positive += decile_positive
        positive_rate = decile_data['actual'].mean()
        lift = positive_rate / overall_rate if overall_rate > 0 else 0
        min_score = decile_data['predicted_proba'].min()
        max_score = decile_data['predicted_proba'].max()
        cum_recall = cumulative_positive / total_positive if total_positive > 0 else 0

        print(f"{int(decile):<8} {min_score:<12.4f} {max_score:<12.4f} {count:<8} {positive_rate:<15.2%} {lift:<8.2f} {cum_recall:<12.2%}")

    print(f"\nOverall {positive_class_name.lower()} rate: {overall_rate:.2%}")


df = load_data(DATA_FILE, DATE_COLUMN, TARGET_COLUMN)

X = df[feature_names].copy()
y = df['target'].copy()

print("\nSetting up time-series cross-validation...")
splitter = DateBasedTimeSeriesSplitter(
    window_length=WINDOW_LENGTH,
    fh=FORECAST_HORIZON,
    test_window_length=TEST_WINDOW_LENGTH,
    step_length=STEP_LENGTH,
    n_splits=N_SPLITS
)

print("Setting up preprocessing and model pipeline...")
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

print("\nGenerating cross-validation splits...")
splits = list(splitter.split(df, date_column=DATE_COLUMN))

print(f"Generated {len(splits)} splits")

print("\nEnumerating split date ranges...")
for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
    train_dates = df[DATE_COLUMN].iloc[train_idx]
    test_dates = df[DATE_COLUMN].iloc[test_idx]

    print(f"\n{'='*60}")
    print(f"Fold {fold_num}")
    print(f"{'='*60}")
    print(f"Train period: {train_dates.min()} to {train_dates.max()}")
    print(f"Test period: {test_dates.min()} to {test_dates.max()}")
    print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")

print("\n" + "="*60)
print("Running cross_validate...")
print("="*60)

cv_results = cross_validate(
    estimator=full_pipeline,
    X=X,
    y=y,
    cv=splits,
    scoring={'roc_auc': make_scorer(roc_auc_score, response_method='predict_proba')},
    return_estimator=True,
    return_train_score=False,
    verbose=1
)

print("\nCross-validation complete!")
print(f"Number of estimators trained: {len(cv_results['estimator'])}")

auc_scores = cv_results['test_roc_auc']
feature_importances = [est.named_steps['model'].feature_importances_ for est in cv_results['estimator']]

print("\n" + "="*80)
print("LIFT ANALYSIS BY FOLD")
print("="*80)

all_y_true = []
all_y_pred = []

for fold_num, ((train_idx, test_idx), estimator) in enumerate(zip(splits, cv_results['estimator']), 1):
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    y_pred_proba_fold = estimator.predict_proba(X_test_fold)[:, 1]
    y_pred_fold = estimator.predict(X_test_fold)

    # Generate lift table for this fold
    generate_lift_table(y_test_fold.values, y_pred_proba_fold, fold_num=fold_num, positive_class_name=POSITIVE_CLASS_NAME)

    all_y_true.extend(y_test_fold)
    all_y_pred.extend(y_pred_fold)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

print(f"\n{'='*80}")
print(f"Total predictions collected across all folds: {len(all_y_pred)}")
print(f"{'='*80}")

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
for i, auc in enumerate(auc_scores, 1):
    print(f"Fold {i} AUC: {auc:.4f}")
print(f"\nAverage AUC: {np.mean(auc_scores):.4f}")
print(f"Standard Deviation: {np.std(auc_scores):.4f}")

print()
generate_classification_report(all_y_true, all_y_pred, negative_class_name=NEGATIVE_CLASS_NAME, positive_class_name=POSITIVE_CLASS_NAME)

print()
generate_confusion_matrix(all_y_true, all_y_pred, negative_class_name=NEGATIVE_CLASS_NAME, positive_class_name=POSITIVE_CLASS_NAME)

print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (Averaged Across Folds, Normalized)")
print(f"{'='*60}")
avg_importance = np.mean(feature_importances, axis=0)
max_importance = avg_importance.max()
normalized_importance = avg_importance / max_importance

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': normalized_importance
}).sort_values('Importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
print(feature_importance_df.head(20).to_string(index=False))
print(f"\nAll Feature Importances:")
for idx, row in feature_importance_df.iterrows():
    print(f"{row['Feature']:30s} {row['Importance']:.6f}")
