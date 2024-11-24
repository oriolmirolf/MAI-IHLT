# models.py

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor


def train_model(X, y, model_type='linear'):
    """
    Train a regression model.

    Args:
        X (DataFrame): Feature matrix.
        y (ndarray): Target values.
        model_type (str): Type of model ('linear', 'svr', 'random_forest', 'gradient_boosting', 'decision_tree').

    Returns:
        model: Trained model.
    """
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'svr':
        model = SVR(kernel='rbf')
    elif model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=42)
    else:
        raise ValueError('Invalid model type specified.')

    model.fit(X, y)
    return model
