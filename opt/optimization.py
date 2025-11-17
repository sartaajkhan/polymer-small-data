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
    
    elif model_name == "krr":
        reg = KernelRidge()
    
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

if __name__ == "__main__":
    data_feat = pd.read_csv("RDKit_topological.csv")
    data_feat = data_feat.dropna(axis = 1)
    data_train = pd.read_csv("train.csv")

    descriptor_names = data_feat.columns[2::].tolist()
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    data_feat = data_feat[['SMILES'] + descriptor_names]
    data_train = data_train[['SMILES'] + labels]

    data_concat = data_feat.merge(data_train, on = 'SMILES', how = 'inner')

    label_to_k = {"Tg" : 10, #not enough datapoints + noisy label
                "Tc" : 5, #already easy label, not enough datapoints, so 5 is healthy choice
                "FFV" : 3, #got a lot of datapoints
                "Rg" : 10, #same as Tg
                "Density" : 5} #same as Tc
    
    model_name = "xgboost"
    os.makedirs(f"saved_params/{model_name}", exist_ok = True)

    for label in list(label_to_k.keys()):
        data_concat_prime = data_concat[descriptor_names + [label]].dropna()
        X = data_concat_prime[descriptor_names]
        y = data_concat_prime[label]
        logger.info(f"For label {label} and with total number of datapoints: {len(X)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        #RFECV
        features_used = feature_selection(X_train, y_train, label) #other params i left as optional.. feel free to set them (random_state might be important?)
        X_train_selected = X_train[features_used]
        
        #impact of removing features alone :)
        if model_name == 'xgboost':
            model = xgb.XGBRegressor()
            X_test_1 = X_test[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection for XGBoost, SRCC: {srcc_}, MAE: {mae_}")
        
        elif model_name == "krr":
            model = KernelRidge()
            X_test_1 = X_test[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection for KRR, SRCC: {srcc_}, MAE: {mae_}")

        #tuning + evaluation
        d = hyperparameter_tuning(X_train_selected, y_train, label)
        d['features_used'] = features_used

        hp = d['bscv_result']['best_parameters']
        if model_name == 'xgboost':
            model = xgb.XGBRegressor(**hp)
            X_test_1 = X_test[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection + hyperparameter optimization for XGBoost, SRCC: {srcc_}, MAE: {mae_}")
        
        elif model_name == "krr":
            model = KernelRidge(**hp)
            X_test_1 = X_test[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection + hyperparameter optimization for KRR, SRCC: {srcc_}, MAE: {mae_}")

        path_to_json = os.path.join(f"saved_params/{model_name}", f"{label}.json")
        with open(path_to_json, 'w') as fp:
            json.dump(d, fp)
        
        logger.info("")

