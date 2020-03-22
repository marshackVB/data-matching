from sklearn.base import BaseEstimator, TransformerMixin

from address_parser.parser import Parser

class AddressParser(BaseEstimator, TransformerMixin, Parser):

    def __init__(self, addr_a, addr_b):
        self.addr_a = addr_a
        self.addr_b = addr_b

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        addr_a_parsed = 'addr_a_parse'
        addr_b_parsed = 'addr_b_parse'

        X[addr_a_parsed] = X[self.addr_a].apply(AddressParser.get_addr_number_streetname)
        X[addr_b_parsed] = X[self.addr_b].apply(AddressParser.get_addr_number_streetname)

        get_number = lambda x: x[0]
        get_street = lambda x: x[1]

        X['addrnumber_a'] = X[addr_a_parsed].apply(get_number)
        X['addrstreetname_a'] = X[addr_a_parsed].apply(get_street)

        X['addrnumber_b'] = X[addr_b_parsed].apply(get_number)
        X['addrstreetname_b'] = X[addr_b_parsed].apply(get_street)

        X.drop(addr_a_parsed, axis=1, inplace=True)
        X.drop(addr_b_parsed, axis=1, inplace=True)

        return X

    def get_feature_names(self):
        return self.columns
