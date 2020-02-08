# coding: utf-8

from kedro.pipeline import node, Pipeline
from typing import Dict, Any, List
from kedro_classification.pipelines.data_science.nodes import (
    split_data,
    Linear_Regression_model,
    LightGBM_model
)

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=LightGBM_model,
                inputs=['preprocessed_Data', 'parameters'],
                outputs='regressor',
                name='regressor',
            )
        ]
    )
