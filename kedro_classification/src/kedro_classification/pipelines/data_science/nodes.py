# coding: utf-8
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List
import gc

import warnings
warnings.simplefilter('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

import matplotlib.pyplot as plt

import lightgbm as lgb
import xgboost as xgb

from kedro.contrib.io.matplotlib import MatplotlibLocalWriter


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
    lgb_params = parameters['LightGBM_model'] 


    ### fold?
    ### I want to define this function out of LightGBM_model.
    ### Keep thinking...
    fold = KFold(n_splits=parameters['folds'], random_state=parameters['random_state'])
    #print(lgb_params)
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

def XGBoost_model(
    data:pd.DataFrame,
    parameters:Dict
) -> xgb.core.Booster:
    #oof_preds_xgb = np.zeros(train_df.shape[0])
    #sub_preds_xgb = np.zeros(test_df.shape[0])
    

    fold_xgb = KFold(n_splits=parameters['folds'], random_state=parameters['random_state'])

    train_df = data.drop('ID', axis=1)
    #train_df = train_df.values()
    

    xgb_params = parameters['XGBoost_model']

    num_rounds = xgb_params['num_rounds']

    y = train_df[parameters['target']]
    train_df = train_df.drop(parameters['target'], axis=1)

    feature = train_df.columns.values

    for fold_, (train_idx, valid_idx) in enumerate(fold_xgb.split(train_df.values)):
        train_x, train_y = train_df.iloc[train_idx], y.iloc[train_idx]
        valid_x, valid_y = train_df.iloc[valid_idx], y.iloc[valid_idx]
        
        print("fold n °{}".format(fold_+1))
        trn_Data = xgb.DMatrix(train_x, label = train_y, feature_names=feature)
        val_Data = xgb.DMatrix(valid_x, label = valid_y, feature_names=feature)
        watchlist = [(trn_Data, "Train"), (val_Data, "Valid")]
        print("xgb trainng folds " + str(fold_) + "-" * 50)

        #regressor = xgb.XGBRegressor(**xgb_params)
        #xgb_model = regressor.fit(X=train_x,
        #                          y=train_y,
        #                          eval_set=([train_x,train_y],[valid_x,valid_y]), 
        #                          early_stopping_rounds=50
                                  #verbose_eval=1000,
        #                          )
        xgb_model = xgb.train(xgb_params, trn_Data,num_rounds,watchlist,early_stopping_rounds=100, verbose_eval= 1000)
        #oof_preds_xgb[valid_idx] = xgb_model.predict(xgb.DMatrix(train_df.iloc[valid_idx][feats]), ntree_limit = xgb_model.best_ntree_limit + 50)
        #sub_preds_xgb = xgb_model.predict(xgb.DMatrix(test_df[feats]),ntree_limit= xgb_model.best_ntree_limit)/fold_xgb.n_splits
        
        del train_idx,valid_idx
        gc.collect()
    #xgb.plot_importance(xgb_model)
    #plt.figure(figsize = (16,10))
    #plt.savefig("importance.png")
    #xgb.to_graphviz(xgb_model)
    return xgb_model


def evaluate_LightGBM_model(regressor: lgb.basic.Booster, X_test: pd.DataFrame, y_test: np.ndarray)->pd.DataFrame: 
    y_pred = regressor.predict(X_test, num_iteration=regressor.best_iteration)
    print("y predicted on LightGBM!")
    print(type(y_pred)) 
    #y_pred = np.argmax(y_pred, axis=1)
    #roc_curve = r
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='LGBM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    score  = roc_auc_score(y_test, y_pred)

    single_plot_writer = MatplotlibLocalWriter(filepath="data/07_model_output/ROC_plot_LGBM.png")
    single_plot_writer.save(plt)
    plt.clf()

    logger = logging.getLogger(__name__)
    logger.info("LightGBM AUC is %.3f.", score)

    output = pd.DataFrame({'y_pred': y_pred, 'y_act':y_test})
    return output


def evaluate_XGBoost_model(regressor: xgb.core.Booster, X_test: pd.DataFrame, y_test: np.ndarray)->pd.DataFrame: 
    #X_test = X_test.values
    #print(regressor.feature_names)
    xgb_test = xgb.DMatrix(X_test, feature_names=regressor.feature_names)
    y_pred = regressor.predict(xgb_test, ntree_limit=regressor.best_ntree_limit)
    print("y predicted on XGBoost!")
    print(type(y_pred)) 
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label='XGBM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

    score  = roc_auc_score(y_test, y_pred)

    single_plot_writer = MatplotlibLocalWriter(filepath="data/07_model_output/ROC_plot_XGB.png")
    single_plot_writer.save(plt)
    plt.clf()

    #y_pred = np.argmax(y_pred, axis=1)
    #roc_curve = r
    score  = roc_auc_score(y_test, y_pred)
    logger = logging.getLogger(__name__)
    logger.info("XGBoost AUC is %.3f.", score)

    output = pd.DataFrame({'y_pred': y_pred, 'y_act':y_test})
    return output