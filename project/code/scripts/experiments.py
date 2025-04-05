import warnings
import os
import numpy as np
import pandas as pd
import joblib

from scipy.stats import pearsonr

from sklearn.model_selection import GridSearchCV, RepeatedKFold
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV

def pearson_correlation_scorer(y_true, y_pred):
    """
    Custom scorer function to compute Pearson correlation.
    Returns 0.0 if correlation is NaN.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        corr, _ = pearsonr(y_true, y_pred)
    return corr if not np.isnan(corr) else 0.0


def calculate_pearson_correlations_per_dataset(
    df, y_true_col='score', y_pred_col='predicted_score', dataset_col='dataset'
):
    """
    Calculate Pearson correlations per dataset.
    """
    datasets = df[dataset_col].unique()
    correlations_per_dataset = {}
    for dataset in datasets:
        df_subset = df[df[dataset_col] == dataset]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pearson_corr, _ = pearsonr(df_subset[y_true_col], df_subset[y_pred_col])
        correlations_per_dataset[dataset] = pearson_corr if not np.isnan(pearson_corr) else 0.0
    return correlations_per_dataset


def run_experiment(train_df, test_df, feature_columns, feature_set_name, model_save_path):
    """
    Run an experiment for the given feature set using multiple regression models and hyperparameter tuning.
    This version exclusively uses Recursive Feature Elimination (RFE) for feature selection, exploring
    selecting features from 20 to the nearest ceiling multiple of 10 in 10-feature increments,
    without exceeding 100 or the total number of features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data including 'score' column.
    test_df : pd.DataFrame
        Test data including 'score' column.
    feature_columns : list of str
        Feature names to be used.
    feature_set_name : str
        Name of the feature set (used for saving the model).
    model_save_path : str
        Directory path to save the best model.

    Returns
    -------
    dict
        Dictionary containing best model name, best model, best parameters, metrics for the best model,
        selected features for the best model, and a dictionary of all model results.
    """
    # features and labels
    X_train = train_df[feature_columns]
    y_train = train_df['score']
    X_test = test_df[feature_columns]
    y_test = test_df['score']

    # NaNs?
    if X_train.isna().sum().sum() > 0 or X_test.isna().sum().sum() > 0:
        print("Warning: NaNs detected in input features. Consider imputing or removing them.")
        X_train = X_train.dropna()
        y_train = y_train.loc[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test.loc[X_test.index]

    # align the test set features with train set features
    X_test = X_test[X_train.columns]

    models = {
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'SVR': SVR(),
    }

    # pipeline; placeholder for feature_selector and model
    base_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('feature_selector', 'passthrough'),
        ('model', None)
    ])

    # RFE as feature selector, RandomForestRegressor as the estimator
    rfe_estimator = RandomForestRegressor(random_state=42)

    n_features = len(feature_columns)
    min_k = 20
    max_k_limit = 100

    if n_features < min_k:
        print(f"Warning: Number of features ({n_features}) is less than the minimum required for RFE (k={min_k}). Adjusting k to {n_features}.")
        k_values = [n_features]
    else:
        # ceiling multiple of 10
        ceiling_k = ((n_features + 9) // 10) * 10
        ceiling_k = min(ceiling_k, max_k_limit)

        k_values = list(range(min_k, ceiling_k + 1, 10))

        if ceiling_k not in k_values and ceiling_k > k_values[-1]:
            k_values.append(ceiling_k)

        k_values = [k if k <= n_features else n_features for k in k_values]

    print(f"Number of features: {n_features}")
    print(f"RFE will try k values: {k_values}\n")

    rfe_params = {
        'feature_selector': [RFE(estimator=rfe_estimator)],
        'feature_selector__n_features_to_select': k_values
    }

    gradient_boosting_common = {
        'model__n_estimators': [150, 200, 250],
        'model__max_depth': [5, 7, 9, 11],
        'model__learning_rate': [0.025, 0.05],
        'model__min_samples_split': [2, 5],
        'model__subsample': [0.4, 0.5]
    }

    random_forest_common = {
        'model__n_estimators': [200],
        'model__max_depth': [7],
        'model__min_samples_split': [5],
        'model__max_features': ['sqrt'],
    }

    svr_common = {
        'model__kernel': ['rbf',],
        'model__C': [10],
        'model__epsilon': [0.2],
        'model__gamma': ['auto'],
    }

    gradient_boosting_params = []
    random_forest_params = []
    svr_params = []

    for model_name, model in models.items():
        if model_name == 'GradientBoosting':
            params = rfe_params.copy()
            params.update(gradient_boosting_common)
            params['model'] = [model]
            gradient_boosting_params.append(params)
        elif model_name == 'RandomForest':
            params = rfe_params.copy()
            params.update(random_forest_common)
            params['model'] = [model]
            random_forest_params.append(params)
        elif model_name == 'SVR':
            params = rfe_params.copy()
            params.update(svr_common)
            params['model'] = [model]
            svr_params.append(params)

    param_grids = {
        'GradientBoosting': gradient_boosting_params,
        'RandomForest': random_forest_params,
        'SVR': svr_params
    }

    # CV strategy
    cv_strategy = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)

    # scorer
    pearson_scorer = make_scorer(pearson_correlation_scorer, greater_is_better=True)

    best_model = None
    best_score = -np.inf
    best_model_name = ''
    best_params = None

    all_models_results = {}
    os.makedirs(model_save_path, exist_ok=True)

    for model_name in models.keys():
        print(f"Training model: {model_name}")

        grid_search = GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grids[model_name],
            scoring=pearson_scorer,
            cv=cv_strategy,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            grid_search.fit(X_train, y_train)

        # evalue on test set (just to save, we do not use this anywhere to choose the model)
        model_best = grid_search.best_estimator_
        y_test_pred = model_best.predict(X_test)
        test_df = test_df.copy()
        test_df['predicted_score'] = y_test_pred

        correlations_per_dataset = calculate_pearson_correlations_per_dataset(
            test_df,
            y_true_col='score',
            y_pred_col='predicted_score',
            dataset_col='dataset'
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            test_pearson_corr, _ = pearsonr(y_test, y_test_pred)
        test_pearson_corr = test_pearson_corr if not np.isnan(test_pearson_corr) else 0.0
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # selected features
        feature_selector = model_best.named_steps['feature_selector']
        if hasattr(feature_selector, 'get_support'):
            selected_feature_indices = feature_selector.get_support(indices=True)
            selected_features = X_train.columns[selected_feature_indices]
        elif isinstance(feature_selector, RFE):
            selected_feature_indices = feature_selector.get_support(indices=True)
            selected_features = X_train.columns[selected_feature_indices]
        else:
            selected_features = X_train.columns

        # store this model's results
        all_models_results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_pearson': test_pearson_corr,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'correlations_per_dataset': correlations_per_dataset,
            'selected_features': list(selected_features),
        }

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = model_best
            best_model_name = model_name
            best_params = grid_search.best_params_

        print(f"Best parameters for {model_name}: {grid_search.best_params_}")
        print(f"Best cross-validation Pearson correlation: {grid_search.best_score_:.4f}")
        print(f"Test Pearson correlation: {test_pearson_corr:.4f}")
        print("")

    # best model overall
    print(f"Best model overall: {best_model_name}")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation Pearson correlation: {best_score:.4f}\n")

    # save best model
    model_filename = os.path.join(model_save_path, f"best_model_{feature_set_name}.joblib")
    joblib.dump(best_model, model_filename)
    print(f"Best model saved to: {model_filename}")

    # compute metrics for the best model on test set  (redundant? TODO: check)
    y_test_pred = best_model.predict(X_test)
    test_df['predicted_score'] = y_test_pred
    correlations_per_dataset = calculate_pearson_correlations_per_dataset(
        test_df,
        y_true_col='score',
        y_pred_col='predicted_score',
        dataset_col='dataset'
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        pearson_corr, _ = pearsonr(y_test, y_test_pred)
    pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)
    r2 = r2_score(y_test, y_test_pred)

    print(f"Test results for feature set '{feature_set_name}':")
    print(f"Overall Pearson correlation: {pearson_corr:.4f}")
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    print(f"Overall R-squared: {r2:.4f}")
    for dataset, correlation in correlations_per_dataset.items():
        print(f"    Dataset {dataset}: Pearson = {correlation:.4f}")
    print("")

    best_feature_selector = best_model.named_steps['feature_selector']
    if hasattr(best_feature_selector, 'get_support'):
        best_selected_indices = best_feature_selector.get_support(indices=True)
        best_selected_features = X_train.columns[best_selected_indices]
    elif isinstance(best_feature_selector, RFE):
        best_selected_indices = best_feature_selector.get_support(indices=True)
        best_selected_features = X_train.columns[best_selected_indices]
    else:
        best_selected_features = X_train.columns

    print("Selected features for best model:")
    print(best_selected_features)
    print("")

    best_metrics = {
        'best_model_name': best_model_name,
        'best_params': best_params,
        'best_cv_score': best_score,
        'overall_pearson': pearson_corr,
        'overall_rmse': rmse,
        'overall_mae': mae,
        'overall_r2': r2,
        'correlations_per_dataset': correlations_per_dataset,
        'selected_features': list(best_selected_features),
    }

    return {
        'best_model_name': best_model_name,
        'best_model': best_model,
        'best_params': best_params,
        'metrics': best_metrics,
        'features_used': list(best_selected_features),
        'all_models_results': all_models_results
    }
