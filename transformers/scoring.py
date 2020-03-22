"""
Classes associated with feature engineering for data matching problems.
The Classes are designed to be used with sci-kit learn Pipelines.
"""
import re
from statistics import mean
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import jellyfish as jf
from fuzzywuzzy import fuzz


class StringSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, sim_funcs):
        self.sim_funcs = sim_funcs
        self.column_name = None
        self.columns = []
        self.missing_vals = [None, np.nan]

        self.variants = {"tk": StringSimilarity.top_k_similar,
                         "ms": StringSimilarity.most_similar}

        self.func_mapping = {"jaro_winkler":           StringSimilarity.jaro_winkler,
                             "levenshtein":            StringSimilarity.demerau_levenshtein,
                             "match_ratio_comparison": StringSimilarity.match_rating_comparison,
                             "token_sort":             StringSimilarity.token_sort,
                             "token_set":              StringSimilarity.token_set,
                             "partial_ratio":          StringSimilarity.partial_ratio,
                             "WRatio":                 StringSimilarity.WRatio,
                             "num_perc_diff":          StringSimilarity.num_perc_diff
                             }

        self.functions_to_apply = {}

        # Create a list of functions to apply, adding a decorator
        # variant if specified (tk, ms)
        for func in self.sim_funcs:
            variant_subscript = func[-2:]
            get_variant = self.variants.get(variant_subscript, None)
            if get_variant == None:
                self.functions_to_apply[func] = self.func_mapping[func]
            else:
                base_func = func[:-3]
                get_base_func = self.func_mapping[base_func]
                self.functions_to_apply[func] = get_variant(get_base_func)


    @staticmethod
    def sort_tokens(x):
        """Split a string and sort"""
        x = x.split()
        x.sort()
        return " ".join(x)

    @staticmethod
    def most_similar(func):
        """Return the score of the most similar token"""
        def sim_apply(x, y):
            max_sim = 0
            x = x.split()
            y = y.split()
            for token_x in x:
                for token_y in y:
                    max_sim = max(func(token_x,token_y), max_sim)
            return max_sim
        return sim_apply

    @staticmethod
    def top_k_similar(func):
        """Return the most similar n number of tokens, where
        n equals the number of tokens that make up the shorter
        of the two strings
        """

        def sim_apply(x, y):
            all_sims = []

            x = x.split()
            y = y.split()

            min_length = min(len(x), len(y))

            for token_x in x:
                for token_y in y:
                    all_sims.append(func(token_x,token_y))

            all_sims.sort(reverse=True)

            return int(mean(all_sims[:min_length]))

        return sim_apply

    @staticmethod
    def jaro_winkler(x, y):
        x = x.upper()
        y = y.upper()
        """Jaro-winkler similarity score"""
        return round(jf.jaro_winkler(x, y) * 100)

    @staticmethod
    def demerau_levenshtein(x, y):
        """Demerau Levenshtein similarity applied to
        sorted tokens
        """
        x = x.upper()
        y = y.upper()
        x = StringSimilarity.sort_tokens(x)
        y = StringSimilarity.sort_tokens(y)

        dl = jf.damerau_levenshtein_distance(x, y)
        return round((1 - (dl / (max(len(x), len(y))))) * 100)

    @staticmethod
    def match_rating_comparison(x, y):
        """The Match Rating comparison score of the
        Jellyfish package
        """
        return 100 if jf.match_rating_comparison(x, y) else 0

    @staticmethod
    def token_sort(x, y):
        """Token sort ratio from the FuzzyWuzzy package"""
        return fuzz.token_sort_ratio(x, y)

    @staticmethod
    def token_set(x, y):
        """Token set ratio from the FuzzyWuzzy package"""
        return fuzz.token_set_ratio(x, y)

    @staticmethod
    def partial_ratio(x, y):
        """Partial ratio from the FuzzyWuzzy package"""
        return fuzz.partial_ratio(x, y)

    @staticmethod
    def WRatio(x, y):
        """WRatio from the FuzzyWuzzy package"""
        return fuzz.WRatio(x, y)

    @staticmethod
    def num_perc_diff(x, y):
        x = int(x)
        y = int(y)
        return int(abs(x - y) / min(x, y) * 100)

    @staticmethod
    def apply_func(func, X):
        """Try to apply a function, if it fails return np.nan.
        This would be due to missing values
        """
        x, y = X[0], X[1]
        try:
            return func(x, y)
        except:
            return np.nan

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Iteratively applies a list of functions to a Pandas DataFrame,
        creating a new column for each function. The column name is a
        combination of the input variables names, which should be named the
        same except for a differeing suffix, such as _a, _b.
        """

        df = X.copy(deep=True)

        # Use the portion of column name before the "_"
        self.column_name = re.search('([A-Za-z]+_)', df.columns[0]).group(1)

        all_new_column_names = []

        for func_name, func in self.functions_to_apply.items():

            new_column_name = self.column_name + func_name
            all_new_column_names.append(new_column_name)

            df[new_column_name] = df.apply(lambda row: self.apply_func(func, row), axis=1)


        return df[all_new_column_names]

    def get_feature_names(self):
        """Return the names of the features generated by the transform method"""
        self.columns = []
        for func in self.sim_funcs:
            self.columns.append(self.column_name + func)
        return self.columns
