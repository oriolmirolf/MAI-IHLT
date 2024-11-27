# experiments.py

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
import joblib  # For saving and loading models

def pearson_correlation_scorer(y_true, y_pred):
    """
    Custom scorer function to compute Pearson correlation.
    """
    corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0

def calculate_pearson_correlations_per_dataset(df, y_true_col='score', y_pred_col='predicted_score', dataset_col='dataset'):
    """
    Calculate Pearson correlations per dataset.
    """
    datasets = df[dataset_col].unique()
    correlations_per_dataset = {}
    for dataset in datasets:
        df_subset = df[df[dataset_col] == dataset]
        pearson_corr, _ = pearsonr(df_subset[y_true_col], df_subset[y_pred_col])
        correlations_per_dataset[dataset] = pearson_corr if not np.isnan(pearson_corr) else 0.0
    return correlations_per_dataset

def run_experiment(train_df, test_df, feature_columns, feature_set_name, model_save_path):
    """
    Run experiment for the given feature set using GradientBoostingRegressor.
    Saves the best model to disk for later analysis.
    """
    # Extract features and labels from train data
    X_train = train_df[feature_columns]
    y_train = train_df['score']

    # Define the model and extended parameter grid
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }

    # Set up GridSearchCV for hyperparameter tuning
    pearson_scorer = make_scorer(pearson_correlation_scorer, greater_is_better=True)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=pearson_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    # Fit GridSearchCV
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Report best parameters
    print(f"Best parameters for feature set '{feature_set_name}':")
    print(best_params)
    print(f"Best cross-validation Pearson correlation: {best_cv_score:.4f}")
    print("")

    # Save the best model to disk
    model_filename = f"{model_save_path}/best_model_{feature_set_name}.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Best model for '{feature_set_name}' saved to: {model_filename}")

    # Predict on test data
    X_test = test_df[feature_columns]
    y_test = test_df['score']
    y_test_pred = best_model.predict(X_test)
    test_df['predicted_score'] = y_test_pred

    # Compute correlations per dataset
    correlations_per_dataset = calculate_pearson_correlations_per_dataset(
        test_df,
        y_true_col='score',
        y_pred_col='predicted_score',
        dataset_col='dataset'
    )

    # Compute overall metrics
    pearson_corr, _ = pearsonr(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    print(f"Test results for feature set '{feature_set_name}':")
    print(f"Overall Pearson correlation: {pearson_corr:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    for dataset, correlation in correlations_per_dataset.items():
        print(f"    Dataset {dataset}: Pearson = {correlation:.4f}")
    print("")

    # Prepare metrics dictionary
    metrics = {
        'overall_pearson': pearson_corr,
        'overall_rmse': rmse,
        'correlations_per_dataset': correlations_per_dataset,
        'best_cv_score': best_cv_score,
        'best_params': best_params
    }

    return metrics
