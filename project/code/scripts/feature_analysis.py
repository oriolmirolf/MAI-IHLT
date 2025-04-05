# feature_analysis.py

import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

sns.set_theme(style='whitegrid')

import warnings
warnings.filterwarnings("ignore", message="X has feature names, but GradientBoostingRegressor was fitted without feature names")


def load_best_model(feature_set_name, model_save_path):
    """
    Load the best model from disk.
    """
    model_filename = f"{model_save_path}/best_model_{feature_set_name}.joblib"
    best_model = joblib.load(model_filename)['model']
    # print(f"Loaded best model for '{feature_set_name}' from: {model_filename}")
    return best_model

def get_selected_features(results_df, feature_set_name, model_name):
    """
    Extract the selected features from the results DataFrame.
    """
    row = results_df[
        (results_df['Feature_Set'] == feature_set_name) &
        (results_df['Model_Name'] == model_name)
    ].iloc[0]
    selected_features = row['Selected_Features'].split(', ')
    return selected_features

def get_feature_importances(model, feature_columns):
    """
    Get feature importances from the model.
    """
    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=feature_columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    return feature_importances


def plot_top_features_grid(feature_importances_dict, top_n=20):
    """
    Plots the top N features from each feature set in a 2x2 grid.

    Parameters:
    - feature_importances_dict (dict): Dictionary with feature set names as keys and
      pandas Series of feature importances as values (for lexical, syntactic, semantic, stylistic).
    - top_n (int): The number of top features to plot from each feature set.
    """
    feature_set_names = list(feature_importances_dict.keys())
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (ax, feature_set_name) in enumerate(zip(axes, feature_set_names)):
        feature_importances = feature_importances_dict[feature_set_name]

        feature_importances = feature_importances.sort_values(ascending=False)
        sns.barplot(
            x=feature_importances.values[:top_n],
            y=feature_importances.index[:top_n],
            ax=ax
        )
        ax.set_title(f"Top {min(top_n, len(feature_importances))} Features for {feature_set_name}", fontsize=14)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_ylabel("Feature", fontsize=12)
        ax.tick_params(axis='y', labelsize=10)
        


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    output_path = "results/figures/Top_Features_Per_Feature_Set_Model.png"
    plt.savefig(output_path, dpi=300)
    
    plt.show()

def plot_top_features(feature_importances, feature_set_name, top_n=20):
    """
    Plot the top N features based on importance for a single feature set.

    Parameters:
    - feature_importances (pd.Series): Series of feature importances, indexed by feature names.
    - feature_set_name (str): Name of the feature set.
    - top_n (int): The number of top features to plot.
    """

    feature_importances = feature_importances.sort_values(ascending=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=feature_importances.values[:top_n],
        y=feature_importances.index[:top_n]
    )
    plt.title(f"Top {top_n} Feature Importances for {feature_set_name}", fontsize=16)
    plt.xlabel("Importance", fontsize=14)
    plt.ylabel("Feature", fontsize=14)
    plt.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    
    output_path = "results/figures/Top_Features_Model.png"
    plt.savefig(output_path, dpi=300)
    
    plt.show()


def print_selected_features(results_df, train_features_df):
    """
    Analyze and print selected and not selected features for different feature sets.

    Parameters:
        results_csv_path (str): Path to the results CSV file.
        train_features_csv_path (str): Path to the train features CSV file.
    """


    lexical_features_columns = [col for col in train_features_df.columns if col.startswith('lex_')]
    syntactic_features_columns = [col for col in train_features_df.columns if col.startswith('syn_')]
    semantic_features_columns = [col for col in train_features_df.columns if col.startswith('sem_')]
    stylistic_features_columns = [col for col in train_features_df.columns if col.startswith('sty_')]

    feature_sets = {
        'lexical': lexical_features_columns,
        'syntactic': syntactic_features_columns,
        'semantic': semantic_features_columns,
        'stylistic': stylistic_features_columns,
        'combined': lexical_features_columns + syntactic_features_columns + semantic_features_columns
    }

    best_models = results_df.loc[results_df.groupby('Feature_Set')['Test_Pearson'].idxmax()]

    for feature_set, features in feature_sets.items():

        selected_features_row = best_models.loc[best_models['Feature_Set'] == feature_set, 'Selected_Features']
        
        if selected_features_row.empty:
            print(f"\nFeature Set: {feature_set}")
            print("No best model found.")
            continue
        
        selected_features = selected_features_row.iloc[0].split(', ')

        not_selected_features = [feature for feature in features if feature not in selected_features]

        print("-"*50)
        print(f"Feature Set: {feature_set}")
        print(f"Selected Features ({len(selected_features)}):")
        print(", ".join(selected_features))
        print(f"Not Selected Features ({len(not_selected_features)}):")
        print(", ".join(not_selected_features))
        

def plot_full_analysis_feature_importance_per_dataset(results_df, train_features_df, test_features_df, model_save_path):
    feature_sets = ['lexical', 'syntactic', 'semantic', 'stylistic', 'combined']

    models = {}
    selected_features_dict = {}

    for feature_set_name in feature_sets:
        try:

            best_row = results_df[results_df['Feature_Set'] == feature_set_name] \
                .sort_values('Test_Pearson', ascending=False).iloc[0]
            best_model_name = best_row['Model_Name']


            model = load_best_model(feature_set_name, model_save_path)
            models[feature_set_name] = model


            selected_features = get_selected_features(results_df, feature_set_name, best_model_name)
            selected_features_dict[feature_set_name] = selected_features

        except IndexError:
            print(f"No model found for feature set {feature_set_name}")
        except FileNotFoundError:
            print(f"No saved model file found for feature set {feature_set_name}")

    for feature_set_name in feature_sets:
        if feature_set_name in models:
            print(f"Analyzing feature set: {feature_set_name}")
            model = models[feature_set_name]
            selected_features = selected_features_dict[feature_set_name]

            missing_features = [feat for feat in selected_features if feat not in test_features_df.columns]
            if missing_features:
                print(f"Warning: The following selected features are missing in the test dataset for {feature_set_name}: {missing_features}")

                selected_features = [feat for feat in selected_features if feat in test_features_df.columns]

            plot_feature_importance_per_dataset(
                model, test_features_df, selected_features, feature_set_name, top_n=20
            )
        else:
            print(f"Skipping feature set {feature_set_name} as no model was loaded.")

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

        result = permutation_importance(
            model, X_subset, y_subset, n_repeats=10, random_state=42, scoring='neg_mean_squared_error'
        )
        perm_importances = pd.Series(result.importances_mean, index=feature_columns)
        feature_importance_per_dataset[dataset] = perm_importances.sort_values(ascending=False)

    return feature_importance_per_dataset

def plot_feature_importance_per_dataset(
    model, test_df, feature_columns, feature_set_name, top_n=10
):
    """
    Analyze permutation feature importance per dataset for a given feature set,
    and plot the top N features in a grid.

    Parameters:
    - model: Trained model for the feature set.
    - test_df (pd.DataFrame): The test dataset containing features and dataset labels.
    - feature_columns: List of feature names for the feature set.
    - feature_set_name (str): Name of the feature set.
    - top_n (int): The number of top features to display in the plots.
    """

    datasets = test_df['dataset'].unique()
    num_datasets = len(datasets)

    ncols = 3
    nrows = (num_datasets + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 5, nrows * 4),
        constrained_layout=True
    )
    
    axes = axes.flatten()

    feature_importance_per_dataset = analyze_feature_importance_per_dataset(
        model, test_df, feature_columns
    )
    

    for idx, dataset in enumerate(datasets):
        perm_importances = feature_importance_per_dataset[dataset]
        top_features = perm_importances.head(top_n)

        ax = axes[idx]
        sns.barplot(
            x=top_features.values,
            y=top_features.index,
            ax=ax
        )
        ax.set_title(f"{feature_set_name.title()} - {dataset}", fontsize=12)
        ax.set_xlabel("Importance", fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)
        ax.tick_params(axis='y', labelsize=8)

    # Hide any unused subplots
    for idx in range(num_datasets, len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(f"Top {top_n} Feature Importances per Dataset\nFeature Set: {feature_set_name.title()}", fontsize=16)
    
    output_path = f"results/figures/FI_dataset_{feature_set_name.title()}.png"
    plt.savefig(output_path, dpi=300)
    
    plt.show()
    
def plot_error_distribution_grid(error_dict, feature_set_names):
    """
    Plot error distributions for all feature sets in a grid.
    """
    n_feature_sets = len(feature_set_names)
    n_cols = 2
    n_rows = (n_feature_sets + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    axes = axes.flatten()
    
    for idx, feature_set_name in enumerate(feature_set_names):
        errors = error_dict[feature_set_name]
        ax = axes[idx]
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title(f"Error Distribution for {feature_set_name}")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Frequency")

    for idx in range(len(feature_set_names), len(axes)):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.show()


def plot_true_vs_predicted_density(y_true, y_pred):
    """
    Plot a density scatter plot of true vs. predicted scores.

    Parameters:
    - y_true (array-like): True scores.
    - y_pred (array-like): Predicted scores.
    """
    plt.figure(figsize=(8, 8))
    
    sns.kdeplot(
        x=y_true, 
        y=y_pred, 
        fill=True, 
        cmap='Blues', 
        thresh=0, 
        levels=100,
        alpha=0.7
    )
    
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Fit (y = x)')
    
    plt.title("True vs. Predicted Scores Density for the Combined Model")
    plt.xlabel("True Score")
    plt.ylabel("Predicted Score")
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()

    plt.tight_layout()
    
    output_path = "results/figures/density_true_vs_combined.png"
    plt.savefig(output_path, dpi=300)
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
    
def collect_predictions_and_errors(results_df, test_df, model_save_path):
    """
    xxx
    """
    feature_sets = ['lexical', 'syntactic', 'semantic', 'stylistic', 'combined']
    error_dict = {}
    y_true_dict = {}
    y_pred_dict = {}
    data_dict = {}
    feature_importances_dict = {}

    for feature_set_name in feature_sets:
        # print(f"\nProcessing feature set: {feature_set_name}")
        
        best_row = results_df[results_df['Feature_Set'] == feature_set_name].sort_values('Test_Pearson', ascending=False).iloc[0]
        best_model_name = best_row['Model_Name']
        best_model = load_best_model(feature_set_name, model_save_path)
        selected_features = get_selected_features(results_df, feature_set_name, best_model_name)
        
        feature_importances = get_feature_importances(best_model, selected_features)
        if feature_importances is not None:
            feature_importances_dict[feature_set_name] = feature_importances
        else:
            print(f"Cannot compute feature importances for {feature_set_name}.")
        
        test_df_copy = test_df.copy()
        X_test = test_df_copy[selected_features]
        y_test = test_df_copy['score']
        
        y_pred = best_model.predict(X_test)
        
        errors = y_test - y_pred
        
        error_dict[feature_set_name] = errors
        y_true_dict[feature_set_name] = y_test
        y_pred_dict[feature_set_name] = y_pred
        data_dict[feature_set_name] = X_test
        
    return error_dict, y_true_dict, y_pred_dict, data_dict


def get_hardest_failures(test_df, original_data, y_true_col='score', y_pred_col='predicted_score', top_n=5):
    """
    Identify the examples with the largest prediction errors and include sentences.
    """
    original_df = pd.DataFrame(original_data, columns=["sentence1", "sentence2", "score", "dataset"])

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

def plot_combined_distribution_grid(y_true, y_pred_combined):
    """
    Plot distributions of true scores and the predicted scores for the "combined" model in a single figure.

    The plot will have two rows: one for the true distribution and one for the combined predicted distribution.
    """
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

    sns.histplot(y_true, kde=True, bins=20, ax=axes[0])
    axes[0].set_title("True Distribution")
    axes[0].set_xlabel("Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 5)
    axes[0].legend([], frameon=False) 

    sns.histplot(y_pred_combined, kde=True, bins=20, ax=axes[1])
    axes[1].set_title("Predicted Distribution")
    axes[1].set_xlabel("Predicted Score")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, 5)

    plt.tight_layout()
    
    output_path = "results/figures/true_vs_pred_dist.png"
    plt.savefig(output_path, dpi=300)
    
    plt.show()