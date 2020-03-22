"""
A set of utility functions related to active learning
"""

import pandas as pd
import numpy as np


def get_labeled_samples(df, matches_index, non_matches_index):
    """Given a list of indexes associated with matching and non-matching
    record pairs, create a labeled dataset.
    """

    matches_df = df.loc[matches_index]
    matches_df['label'] = 1

    non_matches_df = df.loc[non_matches_index]
    non_matches_df['label'] = 0

    labeled_features = pd.concat([matches_df, non_matches_df], axis=0, ignore_index=False)
    labeled_features = labeled_features.sample(frac=1, replace=False)

    return labeled_features
