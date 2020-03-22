import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
import scipy.stats
import pandas as pd
import numpy as np
import os
import re
import pickle


from address_parser.labels import labels, label_types, DIRECTIONS, STREET_NAMES
from address_parser.parser import *

# Get the set of label index locations
labels_idx = [i for i, v in enumerate(labels)]

# All labels and features
all_labels = [[label for token, label in address] for address in labels]
all_tokens = [[token for token, label in address] for address in labels]

all_tokens_features = [addr2features(address) for address in all_tokens]

# Randomly choose index location as the test set
testing_idx = np.random.choice(labels_idx, 30, replace=False)

# Split out training and testing sets
training_labels = [v for i, v in enumerate(labels) if i not in testing_idx]
testing_labels = [v for i, v in enumerate(labels) if i in testing_idx]


# Generate features for training and testing sets
x_train = [[addr[0] for addr in address] for address in training_labels]
y_train = [[addr[1] for addr in address] for address in training_labels]

x_test = [[addr[0] for addr in address] for address in testing_labels]
y_test = [[addr[1] for addr in address] for address in testing_labels]

# Calculate the features
x_train_features = [addr2features(address) for address in x_train]
x_test_features = [addr2features(address) for address in x_test]


# Train the model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True)

crf.fit(x_train_features, y_train)
y_pred = crf.predict(x_test_features)

metrics.flat_f1_score(y_pred, y_test, average='weighted', labels=label_types)

# group B and I results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0]))

print(metrics.flat_classification_report(
    y_test, y_pred, labels=label_types, digits=3))

# Model fit statistics

                           precision    recall  f1-score   support

            AddressNumber      1.000     1.000     1.000        27
        StreetNamePreType      1.000     1.000     1.000        10
               StreetName      1.000     1.000     1.000        30
       StreetNamePostType      1.000     0.941     0.970        17
 StreetNamePreDirectional      1.000     1.000     1.000         7
            OccupancyType      0.800     0.571     0.667         7
      OccupancyIdentifier      0.833     1.000     0.909        10
              USPSBoxType      1.000     1.000     1.000         1
                USPSBoxID      1.000     1.000     1.000         1
StreetNamePostDirectional      0.000     0.000     0.000         0
           SubaddressType      1.000     1.000     1.000         1
     SubaddressIdentifier      1.000     1.000     1.000         1

              avg / total      0.973     0.964     0.966       112


# Fit model on all data (Final model)
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True)

crf.fit(all_tokens_features, all_labels)
y_pred = crf.predict(all_tokens_features)

print(metrics.flat_classification_report(
    all_labels, y_pred, labels=label_types, digits=2))


pickle.dump(crf, open("/parser/cfr_model.p", 'wb'))


def get_pred_proba(address):
    """Get probability of each class for each address token"""

    address = [address.split()]

    features =  [addr2features(token) for token in address]

    predictions = crf.predict_marginals(features)

    return predictions



# View mis-classified cases
pred_with_labels = list(zip(y_pred, y_test, x_test))

mis_classified = []
for pred, label, tokens in pred_with_labels:
    if pred != label:
        mis_classified.append([pred, label, tokens])

len(mis_classified)
mis_classified[0]

test = y_pred[:3]


# Active learning portion
# View addresses with lowest mean probability
def get_tag_and_prob(pred_proba):
    from statistics import median

    best_for_address = []

    for address in pred_proba:
        tag_and_prob =[]
        for dict in address:
            max_prob = 0
            for tag, prob in dict.items():
                if prob > max_prob:
                    highest_prob_tag = (tag, prob)
                    max_prob = prob
            tag_and_prob.append(highest_prob_tag)
        mean_prob_for_address = median([i[1] for i in tag_and_prob])
        best_for_address.append([tag_and_prob, mean_prob_for_address])
    return best_for_address


best_for_address = get_tag_and_prob(test)
best_for_address


test_sort = sorted(best_for_address, key = lambda x: x[1])[0]




addresses = pd.concat([sample_business_100k[sample_business_100k.address_a.notnull()]['address_a'],
                        sample_business_100k[sample_business_100k.address_b.notnull()]['address_b']], ignore_index=True)
address_list = []
for index, addr in addresses.iteritems():
    address_list.append(addr)


address_proba = list(map(get_pred_proba, address_list))
address_proba_mean = list(map(get_tag_and_prob, address_proba))

address_proba_mean_raw = list(zip(address_proba_mean, address_list))
address_proba_mean_raw_sort = sorted(address_proba_mean_raw, key = lambda x: x[0][0][1])

address_proba_mean_sort = sorted(address_proba_mean, key = lambda x: x[0][1])

for address in address_proba_mean_raw_sort[:100]:
    print(address)
