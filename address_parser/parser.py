"""
This module applies conditional random field model to parse addresses
into components. Given an address, it generates featurse for the model
and classifies address components into the categories defined in
labels.py
"""

import re
import pickle
import os
from address_parser.labels import STREET_NAMES, DIRECTIONS

# This file doesnt exist currentl
#model_path = os.path.dirname(__file__)
crf = pickle.load(open(model_path + './address_parser/crf_model.p', 'rb'))

class Parser():

    @staticmethod
    def contains_punctuation(x):
        x = str(x)
        return bool(re.findall('[^A-Za-z0-9]+', x))

    @staticmethod
    def is_numeric(x):
        return x.isnumeric()

    @staticmethod
    def is_character(x):
        return x.isalpha()

    @staticmethod
    def is_street_type(x):
        x = str(x)
        x = re.sub('[^A-Za-z0-9]+', '', x).lower()
        return x in STREET_NAMES

    @staticmethod
    def is_directional(x):
        x = str(x)
        x = re.sub('[^A-Za-z0-9]+', '', x).lower()
        return x in DIRECTIONS

    @staticmethod
    def get_length(x):
        return len(x)

    @staticmethod
    def is_pobox(x):
        return True if x.upper() in ["P", "O", "PO", "BOX"] else False

    @staticmethod
    def has_vowels(x):
        return bool(set(x.upper()) & set('AEIOU'))

    @staticmethod
    def test(i, val):
        is_first = True if i == 0 else False
        return (i, is_first)

    @staticmethod
    def get_current_and_neighbor_features(index, address_list):
        """Generates features for each address tokens in a windowing
        fashion. If the token is the first in a string, features for
        the token and the next token will be generated. If the token is
        the last token, features for the token and the prior token will
        be generated. Otherwise, features for the token, the prior token
        and the next token will be generated.
        """

        # Get index position of last token in list
        address_length = len(address_list) - 1

        def get_features(index, pf = "0_"):

            address = address_list[index]

            features_dict = {pf + "first_element": True if index == 0 else False,
                             pf + "last_element": True if index == address_length else False,
                             pf + "contains_punctuation": Parser.contains_punctuation(address),
                             pf + "is_numeric": Parser.is_numeric(address),
                             pf + "is_character": Parser.is_character(address),
                             pf + "word_length": Parser.get_length(address),
                             pf + "is_street_type": Parser.is_street_type(address),
                             pf + "is_directional": Parser.is_directional(address),
                             pf + "is_pobox": Parser.is_pobox(address),
                             pf + "has_vowels": Parser.has_vowels(address)}

            return features_dict

        features = get_features(index)

        if index < address_length:
            features.update(get_features(index + 1, "+1_"))

        if index != 0:
            features.update(get_features(index - 1, "-1_"))

        return features

    @staticmethod
    def addr2features(address):
        """Accepts a list of addrss tokens"""
        return [Parser.get_current_and_neighbor_features(i, address) for i in range(len(address))]

    @staticmethod
    def tag_address(address):
        """Categorize address elements"""

        address = [address.split()]

        features =  [Parser.addr2features(token) for token in address]

        predictions = crf.predict(features)

        address_and_preds = list(zip(*predictions, *address))

        #Save results in a dictionary
        final_parsed_results = {}

        # Combining tokens of the same label by concatenation
        for label, token in address_and_preds:
            if label in final_parsed_results:
                final_parsed_results[label] += " {0}".format(token)
            else:
                final_parsed_results[label] = token

        return final_parsed_results

    @staticmethod
    def get_addr_number_streetname(address):

        try:

            parsed_address = Parser.tag_address(address)

            number = parsed_address.get('AddressNumber', None)
            po_box_number = parsed_address.get('USPSBoxID', None)

            street = parsed_address.get('StreetName', None)
            po_box = parsed_address.get('USPSBoxType', None)

            addr_number = number if number else po_box_number
            addr_street = street if street else po_box

            return (addr_number, addr_street)
        except:
            return (np.nan, np.nan)
