"""
Classes associated with feature engineering and used for string
cleaning/regex/transformations.The Classes are designed to be used
with sci-kit learn Pipelines.
"""

import re
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

import re
from sklearn.base import BaseEstimator, TransformerMixin

class TokenRemover(BaseEstimator, TransformerMixin):
    """Given a list of tokens, remove all their instances from a string"""

    def __init__(self, col_to_clean, tokens_to_remove):
        self.col_to_clean = col_to_clean
        self.tokens_to_remove = tokens_to_remove


    def remove_tokens(self, name):
        for token in self.tokens_to_remove:

            expression = "^{0}(?=\s)|(?<=\s){0}(?=\s)|(?<=\s){0}$".format(token)
            p = re.compile(expression)
            name = p.sub('', name)
            name = re.sub('[-\s]+', ' ', name)

        return name.strip()


    def fit(self, X, y=None):
        return self


    def transform(self, X):

        col_a = self.col_to_clean + "_a"
        col_b = self.col_to_clean + "_b"

        new_col_name = self.col_to_clean + 'uncommon'

        X[new_col_name + "_a"] = X[col_a].apply(self.remove_tokens)
        X[new_col_name + "_b"] = X[col_b].apply(self.remove_tokens)

        return X


    def get_feature_names(self):
        return self.columns


class StringCleaner(BaseEstimator, TransformerMixin):
    """Clean up strings using different logic for names and address"""
    def __init__(self, dataType):
        self.dataType = dataType

    def fit(self, X, y=None):
        return self

    @staticmethod
    def uppercase(var):
        """Uppercase characters"""
        var = str(var)
        return var.upper()

    @staticmethod
    def singlespace(var):
        """Replace multiple spaces with a single space"""
        var = str(var)
        return re.sub('[-\s]+', ' ', var)

    @staticmethod
    def specialchars(var):
        """Relace special characters with no spaces"""
        var = str(var)
        return re.sub("[',.#]+", '', var)

    def transform(self, X):
        """Apply different string replacement functions
        depending on the type of character
        """

        self.columns = X.columns.tolist()

        if self.dataType.upper() == "NAME":
            return X.applymap(self.__uppercase) \
                    .applymap(self.__singlespace) \
                    .applymap(self.__specialchars)

        if self.dataType.upper() == "ADDRESS":
            return X.applymap(self.__uppercase) \
                    .applymap(self.__singlespace) \
                    .applymap(self.__specialchars)

        if self.dataType.upper() == "CITY":
            return X.applymap(self.__uppercase)

        if self.dataType.upper() == "ZIP":
            return X.applymap(lambda x: str(x)[:5])

    def get_feature_names(self):
        return self.columns
