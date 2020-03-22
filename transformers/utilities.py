"""
A set of utility functions associated with feature engineering.
"""

from functools import reduce
import pandas as pd


def get_name(pipeline_name):
    """Get variable names from an sklearn Pipeline
    """
    last_step = pipeline_name.steps[-1][0]
    return pipeline_name.named_steps[last_step].get_feature_names()


def get_feature_names(stages):
    """Get all variables names from a list of Pipelines
    """
    stages_names = [pipeline for name, pipeline in stages]

    #Reverse order since using reduce functio
    stages_names.reverse()

    feature_names = reduce(lambda acc, x: get_name(x) + acc, stages_names, [])

    return feature_names


def dataframe_from_pipeline(array, stages, dtypes):
    """Create a DataFrame from a transformed Pipeline
    """

    feature_names = get_feature_names(stages)

    df = pd.DataFrame(array, columns=feature_names)
    df = df.astype(dtypes)
    return df
