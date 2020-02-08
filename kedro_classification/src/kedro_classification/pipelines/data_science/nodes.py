# coding: utf-8
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List

import warnings
warnings.simplefilter('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score
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
            'num_iterations'         : parameters['n_estimators'],
            'boosting_type'        : parameters['boosting_type'],
            'objective'            : parameters['objective'],
            'metric'               : parameters['metric'],
            'num_leaves'           : parameters['num_leaves'],
            #'subsample'            : parameters['subsample'],
            #'subsample_freq'       : parameters['subsample_freq'],
            'learning_rate'        : parameters['learning_rate'],
            #'feature_fraction'     : parameters['feature_fraction'],
            'max_depth'            : parameters['max_depth'],
            #'lambda_l1'            : parameters['lambda_l1'],  
            #'lambda_l2'            : parameters['lambda_l2'],
            'verbosity'              : parameters['verbose'],
            'early_stopping_round': parameters['early_stopping_rounds'],
            #'eval_metric'          : parameters['eval_metric'],
            'seed'                 : parameters['seed']
            }


    ### fold?
    ### I want to define this function out of LightGBM_model.
    ### Keep thinking...
    fold = KFold(n_splits=parameters['folds'], random_state=parameters['random_state'])

    oof_pred = np.zeros(len(X))
    ### run model with kfold
    for k,   (train_index, valid_index) in enumerate(fold.split(X, y)):
        #print(train_index)
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
    
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_valid = lgb.Dataset(X_valid, y_valid)

        regressor = lgb.train(lgb_params, lgb_train, valid_sets=lgb_valid, verbose_eval=False)
        
        y_train_pred = regressor.predict(X_train, num_iteration=regressor.best_iteration)
        y_valid_pred = regressor.predict(X_valid, num_iteration=regressor.best_iteration)

        auc_train = roc_auc_score(y_train, y_train_pred)
        auc_valid = roc_auc_score(y_valid, y_valid_pred)
        print('Early stopping round is: {iter}'.format(iter=regressor.current_iteration()))
        print('Fold {n_folds}: train AUC is {train: .3f} valid AUC is {valid: .3f}'.format(n_folds=k+1, train=auc_train, valid=auc_valid))

        #train_roc = roc_auc_score()
    #print(type(regressor))
    
    ### todo: evaluation, predict oof...
    
    return regressor


def evaluate_LightGBM_model(regressor: lgb.basic.Booster, X_test: np.ndarray, y_test: np.ndarray): 
    y_pred = regressor.predict(X_test, num_iteration=regressor.best_iteration)
    print("y predicted!")
    print(type(y_pred)) 
    #y_pred = np.argmax(y_pred, axis=1)
    #roc_curve = r
    score  = roc_auc_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("AUC is %.3f.", score)
