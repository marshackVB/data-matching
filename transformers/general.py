"""
Classes associated with feature engineering for general problems.
The Classes are designed to be used with sci-kit learn Pipelines.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import re

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_keep):
        self.columns = columns_to_keep

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]

    def get_feature_names(self):
        return self.columns


class NumpyConverter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.columns = X.columns.tolist()

        y = len(X.columns)
        x = len(X)
        return np.reshape(X.values, (x, y))

    def get_feature_names(self):
        return self.columns

class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, drop_first=True):
        self.drop_first = drop_first

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame()
        for column in X.columns:
            df[column] = X[column].astype(str)

        dummies = pd.get_dummies(df, drop_first=self.drop_first)
        self.columns = dummies.columns.tolist()
        return dummies

    # Make sure feture names can be accessed
    def get_feature_names(self):
        return self.columns


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Replace one or more categorical variables with a numerical representations
       for each distinct category"""

    def __create_bin_categories(self, distinct_vals):
        """Create numerical mapping given list of distinct values"""
        bins = {}

        for index, value in enumerate(distinct_vals):
            bins[value] = index

        return bins


    def __apply_bin_categories(self, X):
        """Create numerical representations of a columns distinct values"""

        if isinstance(X, pd.core.series.Series):
            self.columns = [X.name]
            self.distinct_vals = {}
            self.distinct_vals[self.columns[0]] = self.__create_bin_categories(X.unique().tolist())

        else:
            self.columns = X.columns.tolist()
            self.distinct_vals = {column: 0 for column in self.columns}
            for column in self.distinct_vals.keys():
                self.distinct_vals[column] = self.__create_bin_categories(X[column].unique().tolist())


    def fit(self, X, y=None):
        """Generate the category to numeric value mapping"""
        self.__apply_bin_categories(X)
        return self


    def transform(self, X, y=None):
        """Apply the category to numeric value mapping"""


        if isinstance(X, pd.core.series.Series):
            return X.apply(lambda x: self.distinct_vals[self.columns][x])


        else:
            df = pd.DataFrame()
            for column in self.columns:
                df[column] = X[column].apply(lambda x: self.distinct_vals[column][x])

        return df


    def get_feature_names(self):
        return self.columns

    def get_category_mapping(self):
        return self.distinct_vals
