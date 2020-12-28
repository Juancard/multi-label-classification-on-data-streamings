# Example of a dynamic weighted ensemble classifier

# Imports
import time
import numpy as np
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import (
    AccuracyWeightedEnsembleClassifier,
    DynamicWeightedMajorityClassifier,
    DynamicWeightedMajorityMultiLabel,
    MultiOutputLearner,
    OzaBaggingMLClassifier
)
from sklearn.linear_model import Perceptron
from skmultiflow.metrics import hamming_score

# Setup a data stream
max_samples = 12000
pretrain_size = 150

stream = SEAGenerator(random_state=1)

# Setup Dynamic Weighted Majority Ensemble Classifier
dwm_base_model = Perceptron()
dwm = DynamicWeightedMajorityClassifier()

# Setup Accuracy Weighted Ensemble Classifier
awm_base_model = Perceptron()
awm = AccuracyWeightedEnsembleClassifier()

# Pretrain
X, y = stream.next_sample(pretrain_size)
y = np.array([y]).T

# Setup OzaBagging
oza_base_estimator = MultiOutputLearner(Perceptron())
oza_base_estimator.partial_fit(
    X, y, classes=[[0, 1]])
oza = OzaBaggingMLClassifier(base_estimator=oza_base_estimator)

# Setup DWME_ML
dwm_ml_base_estimator = MultiOutputLearner(Perceptron())
dwm_ml_base_estimator.partial_fit(
    X, y, classes=[[0, 1]])
dwm_ml = DynamicWeightedMajorityMultiLabel(
    labels=1,
    base_estimator=dwm_ml_base_estimator
)


def train(stream, model):
    # Setup variables to control loop and track performance
    n_samples = 0
    correct_cnt = 0
    predicts = []
    true_labels = []
    star_time = time.time()

    # Train the classifier with the samples provided by the data stream
    while n_samples < max_samples and stream.has_more_samples():
        X, y = stream.next_sample()
        y_pred = model.predict(X)
        predicts.extend(np.array([y_pred]))
        true_labels.extend(np.array([y]))
        if y[0] == y_pred[0]:
            correct_cnt += 1
        model.partial_fit(X, y)
        n_samples += 1
    end_time = time.time()
    perf = hamming_score(true_labels, predicts)

    # Display results
    print('{} samples analyzed.'.format(n_samples))
    print('{}: HS: {} Acc: {} - Time: {}s'.format(
        model.get_info().split("(")[0],
        perf,
        correct_cnt / n_samples,
        (end_time-star_time)
    ))


def train_ml(stream, model):

    # Keeping track of sample count, true labels and predictions to later
    # compute the classifier's hamming score
    count = pretrain_size
    true_labels = []
    predicts = []
    star_time = time.time()

    while stream.has_more_samples() and count < max_samples:
        X, y = stream.next_sample()
        y = np.array([y]).T
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


# train_ml(stream, dwm_ml)
# stream.restart()
# train(stream, dwm)
# stream.restart()
# train(stream, awm)
stream.restart()
train(stream, oza)
