# evaluation.py

from scipy.stats import pearsonr
import xgboost as xgb
from sklearn.svm import SVR

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model using Pearson correlation.

    Args:
        y_true (list): True scores.
        y_pred (list): Predicted scores.

    Returns:
        float: Pearson correlation coefficient.
    """
    correlation = pearsonr(y_true, y_pred)[0]
    return correlation


def train_xgboost(X_train, y_train, params=None):
    """
    Train an XGBoost regressor.

    Args:
        X_train: Training feature matrix.
        y_train: Training target values.
        params: Dictionary of XGBoost parameters.

    Returns:
        model: Trained XGBoost model.
    """
    if params is None:
        params = {'objective': 'reg:squarederror', 'random_state': 42}
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model