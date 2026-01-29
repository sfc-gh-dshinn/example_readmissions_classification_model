"""
Shared utility functions for readmission prediction model.

This module contains common functions used by both the main Python script
(readmission_model.py) and the Jupyter notebook (readmission_analysis.ipynb).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from flaml import tune
from flaml.tune.searcher import CFO
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix


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


def create_preprocessing_pipeline(categorical_features, numerical_features,
                                   categorical_missing_value=-1,
                                   numerical_missing_value=-9999):
    """
    Create a preprocessing pipeline for categorical and numerical features.

    Parameters:
    -----------
    categorical_features : list
        List of categorical feature column names
    numerical_features : list
        List of numerical feature column names
    categorical_missing_value : int, default=-1
        Value to use for unknown/missing categorical values
    numerical_missing_value : int, default=-9999
        Value to use for missing numerical values

    Returns:
    --------
    preprocessing : sklearn.pipeline.FeatureUnion
        Preprocessing pipeline that handles both feature types
    """
    preprocessing = FeatureUnion([
        ('categorical', Pipeline([
            ('selector', FunctionTransformer(lambda X: X[categorical_features].values, validate=False)),
            ('encoder', OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=categorical_missing_value,
                encoded_missing_value=categorical_missing_value
            ))
        ])),
        ('numerical', Pipeline([
            ('selector', FunctionTransformer(lambda X: X[numerical_features].values, validate=False)),
            ('imputer', SimpleImputer(strategy='constant', fill_value=numerical_missing_value))
        ]))
    ])
    return preprocessing


def calculate_tuning_cutoff(df, date_column, cv_config):
    """
    Calculate the end date for tuning data, ensuring temporal separation
    from the main CV evaluation period.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    date_column : str
        Name of the date column
    cv_config : dict
        Cross-validation configuration with keys:
        'window', 'fh', 'test_window', 'step', 'n_splits'

    Returns:
    --------
    tuning_cutoff : pd.Timestamp
        The cutoff date for tuning data (30 days before main CV starts)
    """
    max_date = df[date_column].max()

    # Calculate where main CV fold 1 training starts
    final_test_end = max_date
    final_train_start = (final_test_end
                         - pd.Timedelta(days=cv_config['test_window'] - 1)
                         - pd.Timedelta(days=cv_config['fh'])
                         - pd.Timedelta(days=cv_config['window'] - 1))
    first_train_start = final_train_start - pd.Timedelta(
        days=cv_config['step'] * (cv_config['n_splits'] - 1)
    )

    # Tuning ends 30 days before main CV starts
    tuning_cutoff = first_train_start - pd.Timedelta(days=30)
    return tuning_cutoff


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
    print(f"{'Decile':<8} {'Min Score':<12} {'Max Score':<12} {'Count':<8} {positive_class_name + ' Rate':<15} {'Lift':<8} {'Cum Recall':<12} {'Cum Precision':<13}")
    print("-" * 100)

    overall_rate = lift_df['actual'].mean()
    total_positive = lift_df['actual'].sum()
    cumulative_positive = 0
    cumulative_total = 0

    for decile in sorted(lift_df['decile'].unique(), reverse=True):
        decile_data = lift_df[lift_df['decile'] == decile]
        count = len(decile_data)
        decile_positive = decile_data['actual'].sum()
        cumulative_positive += decile_positive
        cumulative_total += count
        positive_rate = decile_data['actual'].mean()
        lift = positive_rate / overall_rate if overall_rate > 0 else 0
        min_score = decile_data['predicted_proba'].min()
        max_score = decile_data['predicted_proba'].max()
        cum_recall = cumulative_positive / total_positive if total_positive > 0 else 0
        cum_precision = cumulative_positive / cumulative_total if cumulative_total > 0 else 0

        print(f"{int(decile):<8} {min_score:<12.4f} {max_score:<12.4f} {count:<8} {positive_rate:<15.2%} {lift:<8.2f} {cum_recall:<12.2%} {cum_precision:<13.2%}")

    print(f"\nOverall {positive_class_name.lower()} rate: {overall_rate:.2%}")


def generate_subgroup_auc_report(y_true, y_pred_proba, subgroup_values, subgroup_name='Subgroup'):
    """
    Generate and print AUC scores by subgroup to assess model fairness and performance
    across different levels of a categorical feature.

    Parameters:
    -----------
    y_true : array-like
        True binary labels (0 or 1)
    y_pred_proba : array-like
        Predicted probabilities for the positive class
    subgroup_values : array-like
        Values of the subgroup feature for each observation
    subgroup_name : str, default='Subgroup'
        Name of the subgroup feature for display purposes

    Returns:
    --------
    pd.DataFrame
        DataFrame with subgroup-level AUC statistics
    """
    print(f"\n{'='*60}")
    print(f"AUC BY {subgroup_name.upper()}")
    print(f"{'='*60}")

    # Create dataframe for analysis
    subgroup_df = pd.DataFrame({
        'actual': y_true,
        'predicted_proba': y_pred_proba,
        'subgroup': subgroup_values
    })

    results = []
    print(f"\n{subgroup_name:<25} {'Count':>8} {'Prevalence':>12} {'AUC':>8}")
    print("-" * 55)

    for subgroup in sorted(subgroup_df['subgroup'].unique()):
        mask = subgroup_df['subgroup'] == subgroup
        subgroup_data = subgroup_df[mask]
        count = len(subgroup_data)
        prevalence = subgroup_data['actual'].mean()

        # AUC requires at least one positive and one negative case
        if subgroup_data['actual'].nunique() < 2:
            auc = np.nan
            auc_str = "N/A"
        else:
            auc = roc_auc_score(subgroup_data['actual'], subgroup_data['predicted_proba'])
            auc_str = f"{auc:.4f}"

        results.append({
            subgroup_name: subgroup,
            'Count': count,
            'Prevalence': prevalence,
            'AUC': auc
        })

        print(f"{str(subgroup):<25} {count:>8,} {prevalence:>11.2%} {auc_str:>8}")

    # Overall
    overall_auc = roc_auc_score(subgroup_df['actual'], subgroup_df['predicted_proba'])
    overall_prevalence = subgroup_df['actual'].mean()
    print("-" * 55)
    print(f"{'Overall':<25} {len(subgroup_df):>8,} {overall_prevalence:>11.2%} {overall_auc:>8.4f}")

    return pd.DataFrame(results)


def tune_hyperparameters(X_transformed, y, n_splits=3, n_trials=30, timeout=1800,
                         early_stopping_rounds=50, max_n_estimators=1000, random_state=42):
    """
    Tune XGBoost hyperparameters using FLAML with xgb.cv and early stopping.

    Pre-transforms data once to avoid repeated preprocessing overhead.
    Uses cost-effective search algorithms to efficiently explore the hyperparameter space.

    Parameters:
    -----------
    X_transformed : np.ndarray
        Pre-transformed feature matrix (already encoded/imputed)
    y : array-like
        Target variable
    n_splits : int, default=3
        Number of cross-validation folds for tuning
    n_trials : int, default=30
        Maximum number of optimization trials (may complete fewer if timeout reached)
    timeout : int, default=1800
        Maximum time in seconds for optimization (30 min default)
    early_stopping_rounds : int, default=50
        Number of rounds without improvement before stopping
    max_n_estimators : int, default=1000
        Maximum number of boosting rounds
    random_state : int, default=42
        Random seed for xgb.cv reproducibility

    Returns:
    --------
    best_params : dict
        Dictionary of best hyperparameters found
    best_score : float
        Best AUC score achieved during tuning
    analysis : tune.ExperimentAnalysis
        The FLAML analysis object for further inspection
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING WITH FLAML")
    print("="*60)
    print(f"Using {n_splits}-fold CV, up to {n_trials} trials, {timeout}s timeout")

    # Create DMatrix once for efficiency
    dtrain = xgb.DMatrix(X_transformed, label=y)

    # Track trial count and best results for reporting and interrupt handling
    trial_count = [0]
    best_trial_results = [None]  # Store {'config': ..., 'result': ..., 'score': ...}

    def objective(config):
        """Objective function for FLAML tune."""
        trial_count[0] += 1
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'verbosity': 0,
            'learning_rate': config['learning_rate'],
            'max_depth': config['max_depth'],
            'min_child_weight': config['min_child_weight'],
            'subsample': config['subsample'],
            'colsample_bytree': config['colsample_bytree'],
            'reg_lambda': config['reg_lambda'],
        }

        cv_result = xgb.cv(
            params,
            dtrain,
            num_boost_round=max_n_estimators,
            nfold=n_splits,
            early_stopping_rounds=early_stopping_rounds,
            metrics='auc',
            as_pandas=True,
            seed=random_state,
            verbose_eval=False
        )

        best_auc = cv_result['test-auc-mean'].max()
        best_round = int(cv_result['test-auc-mean'].idxmax() + 1)

        result = {'auc': best_auc, 'n_estimators': best_round}
        
        # Track best result for interrupt handling
        if best_trial_results[0] is None or best_auc > best_trial_results[0]['score']:
            best_trial_results[0] = {
                'config': dict(config),
                'result': result,
                'score': best_auc
            }

        # Return metrics - FLAML will track these
        return result

    # Define search space
    search_space = {
        'learning_rate': tune.loguniform(0.01, 0.3),
        'max_depth': tune.randint(3, 10),  # upper is exclusive, so 3-9
        'min_child_weight': tune.randint(1, 11),  # 1-10
        'subsample': tune.uniform(0.5, 1.0),
        'colsample_bytree': tune.uniform(0.5, 1.0),
        'reg_lambda': tune.loguniform(0.01, 10),
    }

    # Start from XGBoost defaults (low cost starting point)
    low_cost_partial_config = {
        'learning_rate': 0.3,
        'max_depth': 6,
        'min_child_weight': 1,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_lambda': 1.0,
    }

    # Use CFO (Cost-Frugal Optimization) - FLAML's native algorithm that doesn't require Optuna
    search_alg = CFO(
        space=search_space,
        low_cost_partial_config=low_cost_partial_config,
        metric='auc',
        mode='max',
        seed=random_state,
    )

    # Run optimization with graceful cancellation handling
    interrupted = False
    analysis = None
    try:
        analysis = tune.run(
            objective,
            config=search_space,
            metric='auc',
            mode='max',
            time_budget_s=timeout,
            num_samples=n_trials,
            search_alg=search_alg,
            verbose=1,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n" + "="*60)
        print("OPTIMIZATION INTERRUPTED BY USER")
        print("="*60)
        # Get partial results from the search algorithm
        analysis = search_alg

    # Extract best results - works for both ExperimentAnalysis and CFO searcher
    if hasattr(analysis, 'best_config'):
        best_config = analysis.best_config
    else:
        best_config = None
    
    if hasattr(analysis, 'best_result'):
        best_result = analysis.best_result
    else:
        best_result = None
    
    # Check if we have valid results
    if best_config is None or not best_config:
        if trial_count[0] == 0:
            raise ValueError("No trials completed. Cannot return best parameters.")
        # Try to get from best_trial_results if available
        if best_trial_results[0] is not None:
            best_config = best_trial_results[0]['config']
            best_result = best_trial_results[0]['result']
        else:
            raise ValueError("No valid trial results found.")

    # Build best_params dict matching expected format
    best_params = {
        'learning_rate': best_config['learning_rate'],
        'max_depth': best_config['max_depth'],
        'min_child_weight': best_config['min_child_weight'],
        'subsample': best_config['subsample'],
        'colsample_bytree': best_config['colsample_bytree'],
        'reg_lambda': best_config['reg_lambda'],
        'n_estimators': best_result['n_estimators'],
    }
    best_score = best_result['auc']

    n_completed = trial_count[0]
    print(f"\nTrials completed: {n_completed}" + (f"/{n_trials}" if n_completed < n_trials else "") + (" (interrupted)" if interrupted else ""))
    print(f"Best AUC: {best_score:.4f}")
    print(f"Best parameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    return best_params, best_score, analysis
