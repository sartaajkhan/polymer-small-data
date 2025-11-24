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
from tqdm import tqdm

from utils.optimization import feature_selection, hyperparameter_tuning
from utils.featurizer import Featurizer, canonicalize_smiles

import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    ### if you are truly interested in featurizing, set featurize = True.. it takes a while to run though
    featurize = False
    if featurize:
        data_train = pd.read_csv("data/train.csv")
        data_train['canonical_smiles'] = data_train['SMILES'].apply(canonicalize_smiles)

        results = []
        for i, smi in enumerate(tqdm(data_train['canonical_smiles'], desc = "calculating descriptors...")):
            summary = Featurizer(smi = smi).summary_of_results()
            summary["SMILES"] = data_train["SMILES"].tolist()[i]
            summary["canonical_smiles"] = data_train["canonical_smiles"].tolist()[i]
            results.append(summary)
        
        df_descriptors = pd.DataFrame(results)
        all_descriptors = df_descriptors.columns[:-2].tolist()
        df_descriptors = df_descriptors[["SMILES", "canonical_smiles"] + all_descriptors]
        df_descriptors.to_csv("data/RDKit_topological.csv", index = False)

    ### run from here!
    data_feat = pd.read_csv("data/RDKit_topological.csv")
    data_feat = data_feat.dropna(axis = 1)

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
    os.makedirs(f"ML_results/tuned_results/{model_name}", exist_ok = True)

    for label in list(label_to_k.keys()):
        data_concat_prime = data_concat[descriptor_names + [label]].dropna()
        Q1 = data_concat_prime[label].quantile(0.25)
        Q3 = data_concat_prime[label].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        data_concat_prime = data_concat_prime[(data_concat_prime[label] >= lower) & (data_concat_prime[label] <= upper)]
        data_concat_prime = data_concat_prime.clip(upper = 1e6)

        X = data_concat_prime[descriptor_names]
        y = data_concat_prime[label]
        logger.info(f"For label {label} and with total number of datapoints: {len(X)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled  = scaler.transform(X_test)

        #RFECV
        features_used = feature_selection(X_train_scaled, y_train, label, model_name = model_name) #other params i left as optional.. feel free to set them (random_state might be important?)
        X_train_selected = X_train_scaled[features_used]
        
        #impact of removing features alone :)
        if model_name == 'xgboost':
            model = xgb.XGBRegressor()
            X_test_1 = X_test_scaled[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection for XGBoost, SRCC: {srcc_}, MAE: {mae_}")
        
        elif model_name == "krr":
            model = KernelRidge()
            X_test_1 = X_test_scaled[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection for KRR, SRCC: {srcc_}, MAE: {mae_}")

        #tuning + evaluation
        d = hyperparameter_tuning(X_train_selected, y_train, label, model_name = model_name)
        d['features_used'] = features_used

        hp = d['bscv_result']['best_parameters']
        if model_name == 'xgboost':
            model = xgb.XGBRegressor(**hp)
            X_test_1 = X_test_scaled[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection + hyperparameter optimization for XGBoost, SRCC: {srcc_}, MAE: {mae_}")
        
        elif model_name == "krr":
            model = KernelRidge(**hp)
            X_test_1 = X_test_scaled[features_used]

            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_1)
            srcc_ = spearmanr(y_test, y_pred)[0]
            mae_ = mean_absolute_error(y_test, y_pred)
            logger.info(f"After feature selection + hyperparameter optimization for KRR, SRCC: {srcc_}, MAE: {mae_}")

        path_to_json = os.path.join(f"ML_results/tuned_results/{model_name}", f"{label}.json")
        d['scores'] = {"SRCC" : srcc_, "MAE" : mae_}
        with open(path_to_json, 'w') as fp:
            json.dump(d, fp)
        
        logger.info("")

