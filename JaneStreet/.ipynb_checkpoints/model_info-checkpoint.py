import polars as pl
import pandas as pd
import numpy as np
import warnings; warnings.filterwarnings(action='ignore')

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping, log_evaluation

MODEL = 'lgb' # Change this to the trained model in train.ipynb

# Custom R2 metric for XGBoost
def r2_xgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    
    return -r2 # Must be negative for early stopping to work


# Custom R2 metric for LightGBM
def r2_lgb(y_true, y_pred, sample_weight):
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (np.average((y_true) ** 2, weights=sample_weight) + 1e-38)
    
    return 'r2', r2, True

xgb_params = {'n_estimators': 500,
              'learning_rate': 0.03, 
              'max_depth': 6, 
              'tree_method': 'hist', 
              'objective': 'reg:squarederror',
              'early_stopping_rounds': 30,
              'eval_metric': r2_xgb,
              'disable_default_eval_metric': True,
              'device': 'cuda'}

lgb_params = {'n_estimators': 500, 
              'learning_rate': 0.03, 
              'max_depth': 6, 
              'objective': 'l2'}

def get_model(model_name):
    """Factory function to retrive model currently in use."""
    
    if model_name == 'xgb':    
        return XGBRegressor(**xgb_params)
        
    elif model_name == 'lgb': 
        return LGBMRegressor(**lgb_params)
        
    return None 


def fit_model(model_name, model, X_train, X_test, y_train, y_test, w_train, w_test):
    fit_params = None 
    
    # Fits XGB fold 
    if model_name == 'xgb': 
        fit_params = {
            'X': X_train, 
            'y': y_train, 
            'eval_set': [(X_test, y_test)],
            'sample_weight': w_train,
            'sample_weight_eval_set': [w_test],
            'verbose': 50
        }
        
        model.fit(**fit_params)
    
    # Fits LGB fold
    elif model_name == 'lgb': 
        fit_params = {
            'X': X_train, 
            'y': y_train, 
            'eval_metric': [r2_lgb],
            'eval_set': [(X_test, y_test, w_test)],
            'callbacks': [
                early_stopping(30),
                log_evaluation(50)
            ]
        }
        
        model.fit(**fit_params)
        
    return 