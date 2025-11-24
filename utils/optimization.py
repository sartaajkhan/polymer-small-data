import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

# models (linear regression, krr, xgb, tabPFN)
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge #krr
import xgboost as xgb #xgb

# feature selection + hyp opt
from sklearn.feature_selection import RFECV
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Integer
from skopt.space import Categorical
from sklearn.inspection import permutation_importance
from loguru import logger
import os
import json

import warnings
warnings.simplefilter('ignore')

def feature_selection(X_train, y_train, label, verbose = 10, min_features = 30, 
                      random_state = 42, scorer = "neg_mean_absolute_error", model_name = "xgboost"):
    label_to_k = {"Tg" : 10, #not enough datapoints + noisy label
                  "Tc" : 5,
                  "FFV" : 3,
                  "Rg" : 10, #same reason as Tg
                  "Density" : 5}
    
    if model_name == "xgboost":
        reg = xgb.XGBRegressor(objective = "reg:absoluteerror", random_state = random_state)
        cv = KFold(label_to_k[label], random_state = random_state, shuffle = True)
        rfecv = RFECV(
            estimator=reg,
            step=1,
            cv=cv,
            scoring=scorer,
            min_features_to_select=min_features,
            n_jobs=-1,
            verbose=verbose,
        )

        rfecv.fit(X_train, y_train)
        features_used = []
        for name, rank in sorted(zip(rfecv.feature_names_in_, rfecv.ranking_), key=lambda x: x[1]):
            if rank == 1:
                features_used.append(name)
        
        return features_used

    elif model_name == "krr":
        reg = KernelRidge()
        reg.fit(X_train, y_train)

        r = permutation_importance(reg, X_train, y_train, n_repeats = 10, random_state = 42)
        importances = r.importances_mean
        idx = np.argsort(importances)[::-1][:min_features]

        return X_train.columns[idx]


def hyperparameter_tuning(X_train_selected, y_train, label, verbose = 10, random_state = 42, scorer = "neg_mean_absolute_error", model_name = "xgboost"):
    label_to_k = {"Tg" : 10, #not enough datapoints + noisy label
                  "Tc" : 5,
                  "FFV" : 3,
                  "Rg" : 10, #same reason as Tg
                  "Density" : 5}
    
    cv = KFold(label_to_k[label], random_state = random_state, shuffle = True)
    
    if model_name == "xgboost":
        reg = xgb.XGBRegressor(objective = 'reg:absoluteerror', random_state = random_state)
        search_spaces = {
                'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                'max_depth': Integer(3, 10),
                'subsample': Real(0.1, 1.0, 'uniform'),
                'colsample_bytree': Real(0.1, 1.0, 'uniform'),  # subsample ratio of columns by tree
                'reg_lambda': Real(1e-6, 1000, 'log-uniform'),
                'reg_alpha': Real(1e-6, 1.0, 'log-uniform'),
                'n_estimators' : Categorical([100, 200, 300, 400, 500])
            }
    
    elif model_name == "krr":
        reg = KernelRidge()
        search_spaces = {
            "kernel": Categorical(["rbf", "laplacian", "linear"]),
            "alpha": Real(1e-6, 1e3, prior="log-uniform"),
            "gamma": Real(1e-6, 1e2, prior="log-uniform")
        }

    opt = BayesSearchCV(
        estimator=reg,
        search_spaces=search_spaces,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_iter=50,
        n_points=1,
        n_jobs=-1,
        verbose=verbose,
        return_train_score=False,
        refit=False,
        optimizer_kwargs={'base_estimator': 'GP'},
        random_state=random_state
    )

    opt.fit(X_train_selected, y_train)

    df_opt = pd.DataFrame(opt.cv_results_)
    best_score = opt.best_score_
    best_score_std = df_opt.iloc[opt.best_index_].std_test_score
    best_params = opt.best_params_

    bscv_result = {
        "best_parameters": best_params,
        "best_score": best_score,
        "best_score_std": best_score_std
    }

    d = {'bscv_result' : bscv_result}
    return d