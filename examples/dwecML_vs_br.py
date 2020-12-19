# Example of a dynamic weighted ensemble classifier

# Imports
import time
from skmultiflow.data import MultilabelGenerator
from skmultiflow.meta import (
    MultiOutputLearner, DynamicWeightedMajorityMultiLabel)
from skmultiflow.bayes import NaiveBayes
from sklearn.linear_model import Perceptron
from skmultiflow.metrics import hamming_score
# from sklearn.linear_model import SGDClassifier

# Setup a data stream
n_features = 10
n_labels = 2
max_samples = 10000
pretrain_size = 150

stream = MultilabelGenerator(
    n_samples=max_samples,
    random_state=1,
    n_features=n_features,
    n_targets=2
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
    print('{}: HS: {} - Time: {}s'.format(
        model.get_info().split("(")[0],
        perf,
        (end_time-star_time)
    ))

    stream.restart()


train(stream, dwm_ml)
train(stream, br)
