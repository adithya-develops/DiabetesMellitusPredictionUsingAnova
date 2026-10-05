import math
import numpy as np

class MixedBayesianPredictor:
    """Naive-Bayes-style estimator for mixed numerical and categorical features."""
    NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    CATEGORICAL = ["gender", "hypertension", "heart_disease", "smoking_history"]

    def __init__(self, data, smoothing=1.0):
        self.data = data.copy()
        self.smoothing = float(smoothing)
        self.classes = [0, 1]
        self.groups = {c: self.data[self.data["diabetes"] == c] for c in self.classes}
        self.priors = {c: len(self.groups[c]) / len(self.data) for c in self.classes}
        self.stats = {}
        for c in self.classes:
            g = self.groups[c]
            self.stats[c] = {}
            for f in self.NUMERIC:
                mu = float(g[f].mean())
                sd = float(g[f].std(ddof=0))
                self.stats[c][f] = (mu, max(sd, 1e-6))
        self.cat_counts = {}
        for c in self.classes:
            g = self.groups[c]
            self.cat_counts[c] = {}
            for f in self.CATEGORICAL:
                counts = g[f].value_counts().to_dict()
                self.cat_counts[c][f] = counts

    @staticmethod
    def _gaussian(x, mu, sigma):
        z = (float(x) - mu) / sigma
        return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2 * math.pi))

    def _categorical(self, feature, value, cls):
        counts = self.cat_counts[cls][feature]
        categories = self.data[feature].nunique()
        return (counts.get(value, 0) + self.smoothing) / (len(self.groups[cls]) + self.smoothing * categories)

    def _log_score(self, sample, cls):
        score = math.log(self.priors[cls])
        for f in self.NUMERIC:
            mu, sd = self.stats[cls][f]
            score += math.log(max(self._gaussian(sample[f], mu, sd), 1e-300))
        for f in self.CATEGORICAL:
            score += math.log(max(self._categorical(f, sample[f], cls), 1e-300))
        return score

    def predict(self, sample):
        scores = {c: self._log_score(sample, c) for c in self.classes}
        m = max(scores.values())
        exp_scores = {c: math.exp(scores[c] - m) for c in self.classes}
        total = sum(exp_scores.values())
        return exp_scores[1] / total, exp_scores[0] / total

    def feature_contributions(self, sample):
        rows = []
        for f in self.NUMERIC + self.CATEGORICAL:
            if f in self.NUMERIC:
                a_mu, a_sd = self.stats[1][f]
                n_mu, n_sd = self.stats[0][f]
                a = math.log(max(self._gaussian(sample[f], a_mu, a_sd), 1e-300))
                n = math.log(max(self._gaussian(sample[f], n_mu, n_sd), 1e-300))
            else:
                a = math.log(max(self._categorical(f, sample[f], 1), 1e-300))
                n = math.log(max(self._categorical(f, sample[f], 0), 1e-300))
            rows.append({"feature": f, "log_likelihood_ratio": a - n})
        return rows
