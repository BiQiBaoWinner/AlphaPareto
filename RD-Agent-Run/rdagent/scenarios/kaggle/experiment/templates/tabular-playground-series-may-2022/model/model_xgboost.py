"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame) -> xgb.Booster:
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "learning_rate": 0.5,
        "max_depth": 10,
        "device": "cuda",
        "tree_method": "hist",
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }
    num_boost_round = 10

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dvalid, "validation")], verbose_eval=100)
    return model


def predict(model: xgb.Booster, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest).reshape(-1, 1)
    return y_pred
