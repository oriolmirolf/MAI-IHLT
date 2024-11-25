# cross_validation.py

from sklearn.model_selection import KFold
from .evaluation import evaluate_model
import numpy as np

def cross_validate_model(model, X, y, cv=5):
    """
    Perform cross-validation and return the average Pearson correlation.

    Args:
        model: Machine learning model with fit and predict methods.
        X (DataFrame): Feature matrix.
        y (Series): Target vector.
        cv (int): Number of cross-validation folds.

    Returns:
        float: Average Pearson correlation across folds.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    correlations = []

    for train_index, val_index in kf.split(X):
        X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
        y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_val_cv)
        correlation = evaluate_model(y_val_cv, y_pred_cv)
        correlations.append(correlation)

    avg_correlation = np.mean(correlations)
    return avg_correlation
