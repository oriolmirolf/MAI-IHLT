# experiments.py

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
# from .evaluation import calculate_pearson_per_dataset
import pandas as pd

import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model.

    Args:
        y_true (array): True labels.
        y_pred (array): Predicted labels.

    Returns:
        results (dict): Dictionary containing evaluation results.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    results = {
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson_corr
    }

    return results


def calculate_pearson_per_dataset(df, y_true_col='score', y_pred_col='predicted_score', dataset_col='dataset'):
    """
    Calculate Pearson correlation per dataset.

    Args:
        df (DataFrame): DataFrame containing predictions and dataset labels.
        y_true_col (str): Column name for true scores.
        y_pred_col (str): Column name for predicted scores.
        dataset_col (str): Column name for dataset labels.

    Returns:
        pearson_per_dataset (dict): Dictionary with dataset names as keys and Pearson correlations as values.
    """
    datasets = df[dataset_col].unique()
    pearson_per_dataset = {}
    for dataset in datasets:
        df_subset = df[df[dataset_col] == dataset]
        pearson_corr, _ = pearsonr(df_subset[y_true_col], df_subset[y_pred_col])
        pearson_per_dataset[dataset] = pearson_corr
    return pearson_per_dataset


def run_experiment(train_df, test_df, feature_columns, feature_set_name):
    """
    Run experiment for the given feature set.
    """
    # Extract features and labels from train data
    X_train = train_df[feature_columns]
    y_train = train_df['score']

    # Perform cross-validation on training data
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pearson_overall_list = []
    pearson_per_dataset_list = []
    for train_index, val_index in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)

        # Predict on validation set
        y_val_pred = model.predict(X_val)

        # Compute overall Pearson
        pearson_corr, _ = pearsonr(y_val, y_val_pred)
        pearson_overall_list.append(pearson_corr)

        # Compute Pearson correlation per dataset
        val_df = train_df.iloc[val_index].copy()
        val_df['predicted_score'] = y_val_pred
        pearson_per_dataset = calculate_pearson_per_dataset(
            val_df,
            y_true_col='score',
            y_pred_col='predicted_score',
            dataset_col='dataset'
        )
        pearson_per_dataset_list.append(pearson_per_dataset)

    # Calculate average Pearson correlations across all folds
    avg_pearson_overall = np.mean(pearson_overall_list)

    # Aggregate per-dataset Pearson correlations
    datasets = set()
    for per_dataset in pearson_per_dataset_list:
        datasets.update(per_dataset.keys())
    avg_pearson_per_dataset = {}
    for dataset in datasets:
        values = [per_dataset.get(dataset, np.nan) for per_dataset in pearson_per_dataset_list]
        # Remove NaN values in case any folds don't have data for that dataset
        values = [v for v in values if not np.isnan(v)]
        avg_pearson_per_dataset[dataset] = np.mean(values)

    # Report cross-validation results
    print(f"Cross-validation results for feature set '{feature_set_name}':")
    print(f"Average Pearson overall: {avg_pearson_overall:.4f}")
    for dataset, pearson in avg_pearson_per_dataset.items():
        print(f"    Dataset {dataset}: Average Pearson = {pearson:.4f}")
    print("")

    # Train model on full training data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test data
    X_test = test_df[feature_columns]
    y_test = test_df['score']
    y_test_pred = model.predict(X_test)
    test_df['predicted_score'] = y_test_pred

    # Compute Pearson correlation per dataset
    pearson_per_dataset = calculate_pearson_per_dataset(
        test_df,
        y_true_col='score',
        y_pred_col='predicted_score',
        dataset_col='dataset'
    )

    # Compute overall Pearson
    pearson_corr, _ = pearsonr(y_test, y_test_pred)
    print(f"Test results for feature set '{feature_set_name}':")
    print(f"Overall Pearson correlation: {pearson_corr:.4f}")
    for dataset, pearson in pearson_per_dataset.items():
        print(f"    Dataset {dataset}: Pearson = {pearson:.4f}")
    print("")

    # Get feature importances
    feature_importances = pd.Series(model.feature_importances_, index=feature_columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    return model, test_df, feature_importances
