# Example of a dynamic weighted ensemble classifier

# Imports
import ipdb
import time
from skmultiflow.data import MultilabelGenerator
from skmultiflow.meta import (MultiOutputLearner,
                              DynamicWeightedMajorityMultiLabel,
                              OzaBaggingMLClassifier)
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTreeClassifier
from sklearn.linear_model import Perceptron
from skmultiflow.metrics import hamming_score
# from sklearn.linear_model import SGDClassifier

# Setup a data stream
n_features = 10
n_labels = 5
pretrain_size = 150
max_samples = 10000 + pretrain_size

stream = MultilabelGenerator(
    n_samples=max_samples,
    random_state=1,
    n_features=n_features,
    n_targets=n_labels
)

# Binary Relevance Model
br = MultiOutputLearner(Perceptron())
X, y = stream.next_sample(pretrain_size)
br.partial_fit(X, y, classes=stream.target_values)

stream.restart()

# Dynamic weighted classifier model
X, y = stream.next_sample(pretrain_size)
dwm_ml_base_estimator = MultiOutputLearner(Perceptron())
dwm_ml_base_estimator.partial_fit(X, y, classes=stream.target_values)
dwm_ml = DynamicWeightedMajorityMultiLabel(
    labels=n_labels,
    base_estimator=dwm_ml_base_estimator
)

stream.restart()

# OzaBaggingMLClassifier - NaiveBayes
X, y = stream.next_sample(pretrain_size)
oza_ml_base_estimator = MultiOutputLearner(NaiveBayes())
oza_ml_base_estimator.partial_fit(X, y, classes=stream.target_values)
oza_ml = OzaBaggingMLClassifier(
    base_estimator=oza_ml_base_estimator
)
oza_ml.partial_fit(X, y, classes=stream.target_values)

stream.restart()

# OzaBaggingMLClassifier - Hoeffding Tree
X, y = stream.next_sample(pretrain_size)
oza_ml_ht_base_estimator = MultiOutputLearner(HoeffdingTreeClassifier())
oza_ml_ht_base_estimator.partial_fit(X, y, classes=stream.target_values)
oza_ml_ht = OzaBaggingMLClassifier(
    base_estimator=oza_ml_base_estimator
)
oza_ml_ht.partial_fit(X, y, classes=stream.target_values)


def train(stream, model, ensemble=False):

    # Keeping track of sample count, true labels and predictions to later
    # compute the classifier's hamming score
    count = pretrain_size
    true_labels = []
    predicts = []
    star_time = time.time()

    while stream.has_more_samples() and count < max_samples:
        X, y = stream.next_sample()
        p = model.predict(X)
        model.partial_fit(X, y)
        predicts.extend(p)
        true_labels.extend(y)
        count += 1

    end_time = time.time()
    perf = hamming_score(true_labels, predicts)

    print('{} samples analyzed.'.format(count))
    # Display results
    print('{} ({}): HS: {} - Time: {}s'.format(
        model.get_info().split("(")[0],
        model.base_estimator,
        perf,
        (end_time-star_time)
    ))

    stream.restart()


# train(stream, dwm_ml)
train(stream, br)
train(stream, oza_ml)
train(stream, oza_ml_ht)
train(stream, oza_ml)
