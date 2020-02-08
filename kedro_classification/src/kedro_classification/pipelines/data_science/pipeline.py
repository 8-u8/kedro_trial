# coding: utf-8

from kedro.pipeline import node, Pipeline
from typing import Dict, Any, List
from kedro_classification.pipelines.data_science.nodes import (
    split_data,
    Linear_Regression_model,
    LightGBM_model,
    evaluate_LightGBM_model
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=split_data,
                inputs=["preprocessed_Data", "parameters"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
            ),            
            node(
                func=LightGBM_model,
                inputs=['preprocessed_Data', 'parameters'],
                outputs='regressor',
                name='regressor',
            ),
            node(
                func=evaluate_LightGBM_model,
                inputs=["regressor", "X_test", "y_test"],
                outputs=None,
            ),
        ]
    )