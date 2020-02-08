# coding: utf-8

from kedro.pipeline import node, Pipeline
from kedro_classification.pipelines.data_engineering.nodes import preprocessing

def create_pipeline(**kwargs):
    print('loading create_pipeline in pipeline.py....')
    return Pipeline(
        [
            node(
                func=preprocessing,
                inputs='usedata',
                outputs='preprocessed_Data',
                name='preprocessed_Data',
            ),
        ]
    )