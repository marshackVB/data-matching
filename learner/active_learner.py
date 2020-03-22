"""
This module contains the active learning Class. Given a set of labeled
sample records, the ActiveLeaner trains an initial model and makes
predictions on unlabeled pairs. It then asks the user to label samples of
pairs for which the model is least certain of the correct classification.
It also asks the user to label pairs for which the model is more confident
are matches and non-matches. This is done to create a training dataset that
is more representative of typical patterns in the data, but is still skewed
toward trickier, less certain outlier pairs.

To do: Add ability to choose different models, such as XGBoost  
"""

import pandas as pd
import numpy as np
from random import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature


class ActiveLearner():
    def __init__(self, labeled_features, labeled_labels, all_raw_data, all_features, sample_size, cv, param_grid):
        self.labeled_features  = labeled_features.copy(deep=True)
        self.labeled_labels = labeled_labels.copy(deep=True)
        self.all_raw_data = all_raw_data.copy(deep=True)
        self.all_features = all_features.copy(deep=True)
        self.n_uncertain, self.n_match, self.n_notmatch  = sample_size
        self.cv = cv
        self.statistics = []
        self.clf = RandomForestClassifier(n_estimators = 100,
                                          n_jobs =- 1)
        self.scoring = 'f1'
        self.param_grid = param_grid

    @staticmethod
    def __get_statistics(actual, predicted):
        """Return precision and recall"""

        precision = precision_score(actual, predicted, average='binary')
        recall = recall_score(actual, predicted, average='binary')
        return (round(precision, 2), round(recall, 2))


    def get_model(self):
        return self.clf


    def __fit_model(self):
        """Get model statistics and fit the model"""

        labels = self.labeled_labels
        features = self.labeled_features

        pred = np.array(cross_val_predict(self.clf,
                                          features,
                                          labels,
                                          cv=self.cv))

        stats = self.__get_statistics(labels, pred)
        self.statistics.append(stats)

        self.clf.fit(features, labels)

        return self


    def __query_pairs(self):
        """Return uncertain pairs from pool"""

        probs = self.clf.predict_proba(self.all_features)[:,1] # unlabeled_features

        probs_df = pd.DataFrame(probs, index=self.all_features.index.values, columns=['proba'])
        probs_df['certainty'] = abs(0.5 - probs_df.proba)
        probs_df.sort_values(by='certainty', axis=0, inplace=True)

        uncertain_pairs = probs_df[:self.n_uncertain]
        match_pairs = probs_df[probs_df.proba > 0.5].sample(self.n_match)
        notmatch_pairs = probs_df[probs_df.proba < 0.5].sample(self.n_notmatch)

        pairs_to_label = pd.concat([uncertain_pairs,
                                    match_pairs,
                                    notmatch_pairs], axis=0, ignore_index=False)

        return pairs_to_label.index.values


    def __user_input(self, str_message):
        """Get and validate user input"""

        label = ""
        while label not in ["0", "1"]:
            label = input(str_message + "\n ")
        return int(label)


    def __get_labels(self):
        """Label uncertain pairs and add to labeled training data"""

        uncertain_pairs_index = self.__query_pairs()

        to_label_raw = self.all_raw_data.loc[uncertain_pairs_index]
        to_label_features = self.all_features.loc[uncertain_pairs_index]

        # Remove uncertain pairs from the candidate pool
        self.all_features.drop(uncertain_pairs_index, axis=0, inplace=True)

        labels_list = []
        for index, row in to_label_raw.iterrows():

            print("\n{0:30}\t{1}\n{2:30}\t{3}\n{4:30}\t{5}\n{6:30}\t{7}\n".format(row.name_a, row.name_b,
                                                                                row.address_a, row.address_b,
                                                                                row.zip_a, row.zip_b,
                                                                                row.city_a, row.city_b))


            label = self.__user_input("Is this a match? (0/1)")
            labels_list.append((index, label))

        labels_index = [index for index, label in labels_list]
        labels_values = [label for index, label in labels_list]

        # Create dataframe with index and labels
        add_labels = pd.Series(labels_values, index=labels_index, name='label')

        # Union the new training set to the full training set
        self.labeled_features = pd.concat([self.labeled_features, to_label_features], axis = 0, ignore_index=False)
        self.labeled_labels = pd.concat([self.labeled_labels, add_labels], axis = 0, ignore_index=False)

        return self


    def begin_labeling(self):
        """Begin the active learning process"""

        self.__fit_model()

        while True:

            self.__get_labels()
            self.__fit_model()

            print ("\n")
            for stat in self.statistics:
                print ("precision: {0} recall: {1}".format(stat[0], stat[1]))

            another_round = input("\nContinue active labeling? (y/n)\n ")

            if another_round.upper() != "Y":

                break

    def get_final_model(self):
        """After labeling is complete, fit the model model using
        GridSearchCV
        """

        clf = RandomForestClassifier()

        clf_gscv = GridSearchCV(clf,
                                param_grid=self.param_grid,
                                cv=self.cv,
                                scoring=self.scoring)

        clf_gscv.fit(self.labeled_features, self.labeled_labels)
        return clf_gscv
