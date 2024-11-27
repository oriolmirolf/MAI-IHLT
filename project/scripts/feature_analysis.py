# feature_analysis.py

import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib  # For loading models
import seaborn as sns

def load_best_model(feature_set_name, model_save_path):
    """
    Load the best model from disk.
    """
    model_filename = f"{model_save_path}/best_model_{feature_set_name}.joblib"
    best_model = joblib.load(model_filename)
    print(f"Loaded best model for '{feature_set_name}' from: {model_filename}")
    return best_model

def get_feature_importances(model, feature_columns):
    """
    Get feature importances from the model.
    """
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    return feature_importances

def analyze_feature_importance_per_dataset(model, test_df, feature_columns):
    """
    Analyze feature importance for each test dataset using permutation importance.
    """
    datasets = test_df['dataset'].unique()
    feature_importance_per_dataset = {}

    for dataset in datasets:
        df_subset = test_df[test_df['dataset'] == dataset]
        X_subset = df_subset[feature_columns]
        y_subset = df_subset['score']

        # Compute permutation feature importances
        result = permutation_importance(
            model, X_subset, y_subset, n_repeats=10, random_state=42, scoring='neg_mean_squared_error'
        )
        perm_importances = pd.Series(result.importances_mean, index=feature_columns)
        feature_importance_per_dataset[dataset] = perm_importances.sort_values(ascending=False)

    return feature_importance_per_dataset

def get_top_features(feature_importances, top_n=10):
    """
    Get the top N features based on importance scores.
    """
    top_features = feature_importances.head(top_n).reset_index()
    top_features.columns = ['Feature', 'Importance']
    return top_features

def plot_feature_importances_grid(feature_importances_dict, top_n=20):
    """
    Plot top N feature importances for all feature sets in a 2x2 grid.
    """
    feature_sets = list(feature_importances_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, feature_set_name in enumerate(feature_sets):
        feature_importances = feature_importances_dict[feature_set_name]
        top_features = feature_importances.head(top_n)
        ax = axes[idx]
        top_features.plot(kind='bar', ax=ax)
        ax.set_title(f"Top {top_n} Feature Importances for {feature_set_name}")
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.tick_params(axis='x', labelrotation=45)
    plt.tight_layout()
    plt.show()

def plot_dataset_permutation_importances_grid(feature_importance_per_dataset_dict, feature_set_names, top_n=10):
    """
    Plot top N permutation feature importances per dataset for all feature sets in a 2x2 grid.
    """
    datasets = list(next(iter(feature_importance_per_dataset_dict.values())).keys())
    for dataset in datasets:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        for idx, feature_set_name in enumerate(feature_set_names):
            importances = feature_importance_per_dataset_dict[feature_set_name][dataset]
            top_features = importances.head(top_n)
            ax = axes[idx]
            top_features.plot(kind='bar', ax=ax)
            ax.set_title(f"Top {top_n} Permutation Importances for {feature_set_name} in {dataset}")
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            ax.tick_params(axis='x', labelrotation=45)
        plt.tight_layout()
        plt.show()

def plot_error_distribution_grid(error_dict, feature_set_names):
    """
    Plot error distributions for all feature sets in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, feature_set_name in enumerate(feature_set_names):
        errors = error_dict[feature_set_name]
        ax = axes[idx]
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title(f"Error Distribution for {feature_set_name}")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def plot_true_vs_predicted_density_grid(y_true_dict, y_pred_dict, feature_set_names):
    """
    Plot density scatter plots of true vs. predicted scores for all feature sets in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, feature_set_name in enumerate(feature_set_names):
        y_true = y_true_dict[feature_set_name]
        y_pred = y_pred_dict[feature_set_name]
        ax = axes[idx]
        sns.kdeplot(x=y_true, y=y_pred, fill=True, cmap='Blues', thresh=0, levels=100, ax=ax)
        ax.plot([0, 5], [0, 5], 'r--')
        ax.set_title(f"True vs. Predicted Scores Density for {feature_set_name}")
        ax.set_xlabel("True Score")
        ax.set_ylabel("Predicted Score")
        ax.set_xlim(0, 5)
        ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_matrix_grid(data_dict, feature_importances_dict, feature_set_names, top_n=10):
    """
    Plot correlation matrices of the top N features for all feature sets in a 2x2 grid.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, feature_set_name in enumerate(feature_set_names):
        data = data_dict[feature_set_name]
        feature_importances = feature_importances_dict[feature_set_name]
        top_features = feature_importances.head(top_n).index.tolist()

        corr = data[top_features].corr()
        ax = axes[idx]
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title(f"Feature Correlation Matrix for {feature_set_name}")
    plt.tight_layout()
    plt.show()

def get_hardest_failures(test_df, original_data, y_true_col='score', y_pred_col='predicted_score', top_n=5):
    """
    Identify the examples with the largest prediction errors and include sentences.
    """
    # Create a DataFrame from original_data
    original_df = pd.DataFrame(original_data, columns=["sentence1", "sentence2", "score", "dataset"])


    # Merge test_df and original_df on index
    test_df.reset_index(inplace=True, drop=True)
    original_df.reset_index(inplace=True, drop=True)
    merged_df = pd.concat([test_df, original_df[['sentence1', 'sentence2']]], axis=1)

    datasets = merged_df['dataset'].unique()
    failures = {}
    for dataset in datasets:
        df_subset = merged_df[merged_df['dataset'] == dataset].copy()
        df_subset['error'] = (df_subset[y_pred_col] - df_subset[y_true_col]).abs()
        top_failures = df_subset.nlargest(top_n, 'error')
        failures[dataset] = top_failures[['sentence1', 'sentence2', y_true_col, y_pred_col, 'error']]
    return failures
