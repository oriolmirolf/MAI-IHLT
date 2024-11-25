# models.py

import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def cross_validate_model(X, y, cv=5):
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]
        model = train_model(X_train_cv, y_train_cv)
        y_pred_cv = model.predict(X_val_cv)
        score = evaluate_predictions(y_val_cv, y_pred_cv)
        scores.append(score)
    return np.mean(scores)

def predict(model, X):
    return model.predict(X)
