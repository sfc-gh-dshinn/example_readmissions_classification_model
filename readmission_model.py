import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import roc_auc_score, make_scorer, classification_report, confusion_matrix
from sktime.split.slidingwindow import SlidingWindowSplitter
from xgboost import XGBClassifier


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


class DateBasedTimeSeriesSplitter:
    """
    A time-series cross-validator that splits data based on actual date ranges.
    
    This splitter creates a complete date range from the data and uses 
    SlidingWindowSplitter on unique dates (1 row = 1 date), then filters 
    the actual data to return indices corresponding to those date ranges.
    
    This approach handles:
    - Multiple rows per date
    - Missing dates in the data
    - Proper time-based train/test splits
    """
    
    def __init__(self, window_length, fh, test_window_length, step_length):
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
        """
        self.window_length = window_length
        self.fh = fh
        self.test_window_length = test_window_length
        self.step_length = step_length
    
    def split(self, df, date_column='date'):
        """
        Generate train/test indices based on date ranges.
        
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
        """
        dates = pd.to_datetime(df[date_column])
        
        min_date = dates.min().normalize()
        max_date = dates.max().normalize()
        
        date_range = pd.date_range(start=min_date, end=max_date, freq='D')
        
        splitter = SlidingWindowSplitter(
            window_length=self.window_length,
            fh=self.fh,
            step_length=self.step_length,
            start_with_window=True
        )
        
        date_series = pd.Series(date_range)
        
        for train_date_idx, test_date_idx in splitter.split(date_series):
            train_start_date = date_range[train_date_idx[0]]
            train_end_date = date_range[train_date_idx[-1]]
            
            test_start_date = date_range[test_date_idx[0]]
            test_end_date = test_start_date + pd.Timedelta(days=self.test_window_length - 1)
            
            train_mask = (dates >= train_start_date) & (dates <= train_end_date)
            test_mask = (dates >= test_start_date) & (dates <= test_end_date)
            
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


print("Loading data...")
df = pd.read_csv('diabetic_data_with_dates.csv')

print("Parsing date column...")
df['date'] = pd.to_datetime(df['date'])

print("Converting target variable to binary...")
df['target'] = (df['readmitted'] != 'NO').astype(int)

print("Sorting by date...")
df = df.sort_values('date').reset_index(drop=True)

print(f"Dataset shape: {df.shape}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Target distribution: {df['target'].value_counts().to_dict()}")

X = df.drop(columns=['readmitted', 'target', 'date', 'encounter_id'])
y = df['target']

feature_names = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

print("\nSetting up time-series cross-validation...")
splitter = DateBasedTimeSeriesSplitter(
    window_length=547,  # 18 months (18 * 30.42 ≈ 547 days)
    fh=30,
    test_window_length=60,
    step_length=90
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
splits = []
for train_idx, test_idx in splitter.split(df, date_column='date'):
    splits.append((train_idx, test_idx))
    if len(splits) >= 5:
        break

print(f"Generated {len(splits)} splits")

print("\nEnumerating split date ranges...")
for fold_num, (train_idx, test_idx) in enumerate(splits, 1):
    train_dates = df['date'].iloc[train_idx]
    test_dates = df['date'].iloc[test_idx]
    
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
    
    # Create dataframe for lift analysis
    lift_df = pd.DataFrame({
        'actual': y_test_fold.values,
        'predicted_proba': y_pred_proba_fold
    })
    
    # Sort by predicted probability descending
    lift_df = lift_df.sort_values('predicted_proba', ascending=False).reset_index(drop=True)
    
    # Create deciles
    lift_df['decile'] = pd.qcut(lift_df['predicted_proba'], q=10, labels=False, duplicates='drop') + 1
    
    # Calculate lift statistics by decile
    print(f"\nFold {fold_num} Lift Table:")
    print(f"{'Decile':<8} {'Min Score':<12} {'Max Score':<12} {'Count':<8} {'Readmit Rate':<15} {'Lift':<8} {'Cum Recall':<12}")
    print("-" * 87)
    
    overall_rate = lift_df['actual'].mean()
    total_readmissions = lift_df['actual'].sum()
    cumulative_readmissions = 0
    
    for decile in sorted(lift_df['decile'].unique(), reverse=True):
        decile_data = lift_df[lift_df['decile'] == decile]
        count = len(decile_data)
        decile_readmissions = decile_data['actual'].sum()
        cumulative_readmissions += decile_readmissions
        readmit_rate = decile_data['actual'].mean()
        lift = readmit_rate / overall_rate if overall_rate > 0 else 0
        min_score = decile_data['predicted_proba'].min()
        max_score = decile_data['predicted_proba'].max()
        cum_recall = cumulative_readmissions / total_readmissions if total_readmissions > 0 else 0
        
        print(f"{int(decile):<8} {min_score:<12.4f} {max_score:<12.4f} {count:<8} {readmit_rate:<15.2%} {lift:<8.2f} {cum_recall:<12.2%}")
    
    print(f"\nOverall readmission rate: {overall_rate:.2%}")
    
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

print(f"\n{'='*60}")
print("CLASSIFICATION REPORT (Consolidated Across All Folds)")
print(f"{'='*60}")
print(classification_report(all_y_true, all_y_pred, target_names=['No Readmission', 'Readmission']))

print(f"{'='*60}")
print("CONFUSION MATRIX (Consolidated Across All Folds)")
print(f"{'='*60}")
cm = confusion_matrix(all_y_true, all_y_pred)
print("\nConfusion Matrix:")
print(f"                    Predicted No    Predicted Yes")
print(f"Actual No (0)       {cm[0, 0]:8d}        {cm[0, 1]:8d}")
print(f"Actual Yes (1)      {cm[1, 0]:8d}        {cm[1, 1]:8d}")
print(f"\nInterpretation:")
print(f"True Negatives (TN):  {cm[0, 0]:,} - Correctly predicted no readmission")
print(f"False Positives (FP): {cm[0, 1]:,} - Incorrectly predicted readmission")
print(f"False Negatives (FN): {cm[1, 0]:,} - Incorrectly predicted no readmission")
print(f"True Positives (TP):  {cm[1, 1]:,} - Correctly predicted readmission")

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
