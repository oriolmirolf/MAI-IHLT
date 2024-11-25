# evaluation.py

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