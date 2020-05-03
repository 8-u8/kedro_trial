# coding: utf-8

from kedro.pipeline import node, Pipeline
from typing import Dict, Any, List
from kedro_new_project.pipelines.data_science.nodes import (
    LightGBM_model,
    XGBoost_model,
    evaluate_LightGBM_model,
    evaluate_XGBoost_model
)

def create_pipeline(**kwargs):
    return Pipeline(
        [         
            node(
                func=LightGBM_model,
                inputs=['usedata', 'parameters'],
                outputs='regressor_LGB',
                name='regressor_LGB',
            ),
            node(
                func=evaluate_LightGBM_model,
                inputs=['regressor_LGB', 'preprocessed_Data','parameters'],
                outputs='pred_LGBM',
            ),
            node(
                func=XGBoost_model,
                inputs=['usedata', 'parameters'],
                outputs='regressor_XGB',
                name='regressor_XGB',
            ),
            node(
                func=evaluate_XGBoost_model,
                inputs=['regressor_XGB', 'preprocessed_Data', 'parameters'],
                outputs='pred_XGBM',
            ),
        ]
    )
