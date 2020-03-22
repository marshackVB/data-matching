"""
A collection of test for the string similarity metrics. If this were part of
an actual work product, all functions and class methods would be tested using
this format.
"""
import os
import sys
import pytest

# Adding the parent directory folder to the system path
# so that project modules can be imported
current_file_directory = os.path.realpath((os.getcwd()))
project_directory = os.path.dirname(current_file_directory)
sys.path.append(project_directory)

from transformers.matching import *


@pytest.mark.parametrize("x, y, expected_score", [("SOME BUSINESS NAME", "ANOTHER BUSINESS NAME", 38),
                                                 ("some business name", "ANOTHER BUSINESS NAME", 38),
                                                 ("some business name", "Name SOME business", 100),
                                                 ("", "another business", 0)])
def test_demerau_levenshtein(x, y, expected_score):
    actual_score = StringSimilarity.demerau_levenshtein(x, y)
    assert actual_score == expected_score


@pytest.mark.parametrize("func, X, expected_score", [(StringSimilarity.demerau_levenshtein, (None, None), np.nan),
                                                     (StringSimilarity.demerau_levenshtein, (None, "some business"), np.nan),
                                                     (StringSimilarity.token_set, (None, None), np.nan),
                                                     (StringSimilarity.token_set, (None, "some business"), np.nan),
                                                     (StringSimilarity.demerau_levenshtein, ("some business", "some business"), 100),
                                                     (StringSimilarity.token_set, ("some business", "some business"), 100)])
def test_apply_func(func, X, expected_scorei):
    actual_score = StringSimilarity.apply_func(func, X)
    assert actual_score is expected_score
