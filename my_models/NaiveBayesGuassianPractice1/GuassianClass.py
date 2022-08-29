import math
import numpy as np

class GaussianNB:

    def __init__(self):
        """
            Attributes:
                likelihoods: Likelihood of each feature per class
                class_priors: Prior probabilities of classes
                features: All features of dataset
            """
        self.features = list
        self.class_priors = {}
        self.likelihoods = {}

        self.num_feats = int
        self.train_size = int
        self.X_train = np.array
        self.y_train = np.array

    def fit(self, X, y):

        self.features = list(X.columns)
        self.X_train = X
        self.y_train = y
        self.train_size = X.shape[0]
        self.num_feats = X.shape[1]

        for feature in self.features:
            self.likelihoods[feature] = {}
            for outcome in np.unique(self.y_train):
                self.likelihoods[feature].update({outcome: {}})
                self.class_priors.update({outcome: 0})

        self._calc_class_prior()
        self._calc_likelihoods()

    def _calc_class_prior(self):

        for outcome in np.unique(self.y_train):
            outcome_count = sum(self.y_train == outcome)
            self.class_priors[outcome] = outcome_count / self.train_size

    def _calc_likelihoods(self):

        for feature in self.features:
            for outcome in np.unique(self.y_train):
                self.likelihoods[feature][outcome]['mean'] = self.X_train[feature][
                    self.y_train[self.y_train == outcome].index.values.tolist()].mean()
                self.likelihoods[feature][outcome]['variance'] = self.X_train[feature][
                    self.y_train[self.y_train == outcome].index.values.tolist()].var()

    def predict(self, X):

        results = []
        X = np.array(X)

        for query in X:
            probs_outcome = {}

            for outcome in np.unique(self.y_train):
                prior = self.class_priors[outcome]
                likelihood = 1
                evidence_temp = 1
                # print("Features: " + str(self.features))
                # print("query:" + str(query))

                for feat, feat_val in zip(self.features, query):
                    mean = self.likelihoods[feat][outcome]['mean']
                    var = self.likelihoods[feat][outcome]['variance']
                    # print("Var: " + str(var) + " feat_val: " + str(feat_val) + " mean: " + str(mean))
                    # print("\tMean: " + str(self.likelihoods[feat][outcome]['mean']))
                    # print("\tVarience:" + str(self.likelihoods[feat][outcome]['variance']))

                    likelihood *= (1 / math.sqrt(2 * math.pi * var)) * np.exp(-(feat_val - mean) ** 2 / (2 * var))

                posterior_numerator = (likelihood * prior)
                probs_outcome[outcome] = posterior_numerator

            result = max(probs_outcome, key=lambda x: probs_outcome[x])
            results.append(result)

        return np.array(results)
