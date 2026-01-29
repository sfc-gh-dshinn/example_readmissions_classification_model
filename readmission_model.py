import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, make_scorer, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Import shared utilities from model_utils
from model_utils import (
    DateBasedTimeSeriesSplitter,
    create_preprocessing_pipeline,
    calculate_tuning_cutoff,
    generate_classification_report,
    generate_confusion_matrix,
    generate_lift_table,
    generate_subgroup_auc_report,
    tune_hyperparameters,
)

DATE_COLUMN = 'date'
DATA_FILE = 'diabetic_data_with_dates.csv'
TARGET_COLUMN = 'readmitted'
TARGET_MAPPING_LOGIC = lambda x: 0 if x == 'NO' else 1
CATEGORICAL_MISSING_VALUE = -1
NUMERICAL_MISSING_VALUE = -9999
WINDOW_LENGTH = 547  # 18 months (18 * 30.42 ≈ 547 days)
FORECAST_HORIZON = 30
TEST_WINDOW_LENGTH = 60
STEP_LENGTH = 90
N_SPLITS = 5
NEGATIVE_CLASS_NAME = 'No Readmission'
POSITIVE_CLASS_NAME = 'Readmission'
SUBGROUP_ANALYSIS_FEATURE = 'race'  # Feature for AUC subgroup analysis

# Hyperparameter tuning constants
TUNING_N_SPLITS = 3  # Fewer folds for tuning (speed)
TUNING_N_TRIALS = 30  # Number of FLAML trials
TUNING_TIMEOUT = 1800  # 30 minute timeout
EARLY_STOPPING_ROUNDS = 50
MAX_N_ESTIMATORS = 1000  # High value, early stopping finds optimal


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
    df['target'] = df[target_column].apply(TARGET_MAPPING_LOGIC)

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

print("Setting up preprocessing pipeline...")
preprocessing = create_preprocessing_pipeline(
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_MISSING_VALUE,
    NUMERICAL_MISSING_VALUE
)

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

# ============================================================================
# HYPERPARAMETER TUNING PHASE
# ============================================================================
# Use the first fold's training data for hyperparameter tuning
# This avoids data leakage while still leveraging pre-transformation for speed

print("\n" + "="*60)
print("HYPERPARAMETER TUNING PHASE")
print("="*60)

# Use first fold's training indices for tuning
tuning_train_idx = splits[0][0]
X_tuning = X.iloc[tuning_train_idx]
y_tuning = y.iloc[tuning_train_idx].values

print(f"Tuning on {len(tuning_train_idx)} samples from first fold training data")

# Pre-transform data ONCE for tuning efficiency
print("Pre-transforming data for tuning...")
preprocessing_for_tuning = create_preprocessing_pipeline(
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_MISSING_VALUE,
    NUMERICAL_MISSING_VALUE
)
X_tuning_transformed = preprocessing_for_tuning.fit_transform(X_tuning)

# Run hyperparameter tuning
best_params, best_tuning_score, study = tune_hyperparameters(
    X_tuning_transformed, 
    y_tuning,
    n_splits=TUNING_N_SPLITS,
    n_trials=TUNING_N_TRIALS,
    timeout=TUNING_TIMEOUT
)

# ============================================================================
# FINAL EVALUATION PHASE
# ============================================================================
# Use tuned parameters with full pipeline for proper cross-validation

print("\n" + "="*60)
print("FINAL CROSS-VALIDATION WITH TUNED PARAMETERS")
print("="*60)

# Create model with tuned parameters
tuned_model = XGBClassifier(
    tree_method='hist',
    n_estimators=best_params['n_estimators'],
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_child_weight=best_params['min_child_weight'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    reg_lambda=best_params['reg_lambda'],
)

# Create full pipeline with tuned model
full_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', tuned_model)
])

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
    return_train_score=True,
    verbose=1
)

print("\nCross-validation complete!")
print(f"Number of estimators trained: {len(cv_results['estimator'])}")

auc_scores = cv_results['test_roc_auc']
train_auc_scores = cv_results['train_roc_auc']
feature_importances = [est.named_steps['model'].feature_importances_ for est in cv_results['estimator']]

print("\n" + "="*80)
print("LIFT ANALYSIS BY FOLD")
print("="*80)

all_y_true = []
all_y_pred = []
all_y_proba = []
all_subgroup_values = []

for fold_num, ((train_idx, test_idx), estimator) in enumerate(zip(splits, cv_results['estimator']), 1):
    X_test_fold = X.iloc[test_idx]
    y_test_fold = y.iloc[test_idx]

    y_pred_proba_fold = estimator.predict_proba(X_test_fold)[:, 1]
    y_pred_fold = estimator.predict(X_test_fold)

    # Generate lift table for this fold
    generate_lift_table(y_test_fold.values, y_pred_proba_fold, fold_num=fold_num, positive_class_name=POSITIVE_CLASS_NAME)

    all_y_true.extend(y_test_fold)
    all_y_pred.extend(y_pred_fold)
    all_y_proba.extend(y_pred_proba_fold)
    all_subgroup_values.extend(df.iloc[test_idx][SUBGROUP_ANALYSIS_FEATURE].values)

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
all_y_proba = np.array(all_y_proba)
all_subgroup_values = np.array(all_subgroup_values)

print(f"\n{'='*80}")
print(f"Total predictions collected across all folds: {len(all_y_pred)}")
print(f"{'='*80}")

print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
print(f"\n{'Fold':<6} {'Train AUC':<12} {'Test AUC':<12}")
print("-" * 30)
for i, (train_auc, test_auc) in enumerate(zip(train_auc_scores, auc_scores), 1):
    print(f"{i:<6} {train_auc:<12.4f} {test_auc:<12.4f}")
print("-" * 30)
print(f"{'Avg':<6} {np.mean(train_auc_scores):<12.4f} {np.mean(auc_scores):<12.4f}")
print(f"{'Std':<6} {np.std(train_auc_scores):<12.4f} {np.std(auc_scores):<12.4f}")

print()
generate_classification_report(all_y_true, all_y_pred, negative_class_name=NEGATIVE_CLASS_NAME, positive_class_name=POSITIVE_CLASS_NAME)

print()
generate_confusion_matrix(all_y_true, all_y_pred, negative_class_name=NEGATIVE_CLASS_NAME, positive_class_name=POSITIVE_CLASS_NAME)

# Subgroup AUC analysis
generate_subgroup_auc_report(all_y_true, all_y_proba, all_subgroup_values, subgroup_name=SUBGROUP_ANALYSIS_FEATURE)

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
