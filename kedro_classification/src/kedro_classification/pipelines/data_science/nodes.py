# coding: utf-8
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List

from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import KFold, train_test_split

import lightgbm as lgb


def split_data(data: pd.DataFrame, parameters: Dict) -> List:
    y = data['y'].values
    X = data.drop(['ID','y'], axis=1)
    X = X.values

    X_train, y_train, X_valid, y_valid = train_test_split(
        X, y, test_size=parameters['test_size'], random_state=parameters['random_state']
    )
    return(X_train, y_train, X_valid, y_valid)

def Linear_Regression_model(
    X_train: np.ndarray,
    y_train: np.ndarray
    ) -> LinearRegression:

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    return regressor

def LightGBM_model(
    data: pd.DataFrame,
    #y: np.ndarray,
    parameters: Dict
    ) -> lgb.LGBMRegressor:
    
    ### define classes
    regressor = lgb.LGBMRegressor()
    y = data['y']
    X = data.drop(['y', 'ID'], axis=1)
    ### hyperparameters from parameters.yml
    lgb_params = {
            'n_estimators'         : parameters['n_estimators'],
            'boosting_type'        : parameters['boosting_type'],
            'objective'            : parameters['objective'],
            'metric'               : parameters['metric'],
            'subsample'            : parameters['subsample'],
            'subsample_freq'       : parameters['subsample_freq'],
            'learning_rate'        : parameters['learning_rate'],
            'feature_fraction'     : parameters['feature_fraction'],
            'max_depth'            : parameters['max_depth'],
            'lambda_l1'            : parameters['lambda_l1'],  
            'lambda_l2'            : parameters['lambda_l2'],
            'verbose'              : parameters['verbose'],
            'early_stopping_rounds': parameters['early_stopping_rounds'],
            'eval_metric'          : parameters['eval_metric'],
            'seed'                 : parameters['seed']
            }


    ### fold?
    ### I want to define this function out of LightGBM_model.
    ### Keep thinking...
    fold = KFold(n_splits=parameters['folds'], random_state=parameters['random_state'])

    oof_pred = np.zeros(len(X))
    ### run model with kfold
    for k,   (train_index, valid_index) in enumerate(fold.split(X, y)):
        print(train_index)

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)

        result = lgb.train(lgb_params, lgb_train, valid_sets=lgb_valid)

    
    ### todo: evaluation, predict oof...
    
    return result
