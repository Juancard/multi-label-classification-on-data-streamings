"""
Incremental Structured Output Prediction Tree (iSOUP-Tree) for multi-target
regression.  This is an implementation of the iSOUP-Tree proposed by A. Osojnik,
P. Panov, and S. Džeroski [1].

Aljaž Osojnik, Panče Panov, and Sašo Džeroski. “Tree-based methods for online
multi-target regression.” Journal of Intelligent Information Systems 50.2
(2018): 315-339.
"""

# Imports
import numpy as np
import ipdb
from skmultiflow.data import RegressionGenerator, MultilabelGenerator
from skmultiflow.trees import iSOUPTreeRegressor

# Setup a data stream
n_targets = 20
regression_stream = RegressionGenerator(
    n_targets=n_targets, random_state=1, n_samples=200)

ml_stream = MultilabelGenerator(random_state=1, n_samples=200,
                                n_targets=n_targets, n_features=10)

# Setup iSOUP Tree Regressor
isoup_tree = iSOUPTreeRegressor()

# Auxiliary variables to control loop and track performance
n_samples = 0
max_samples = 200
y_pred = np.zeros((max_samples, n_targets))
y_true = np.zeros((max_samples, n_targets))

# Run test-then-train loop for max_samples and while there is data
while n_samples < max_samples and ml_stream.has_more_samples():
    X, y = ml_stream.next_sample()
    y_true[n_samples] = y[0]
    y_pred[n_samples] = isoup_tree.predict(X)[0]
    isoup_tree.partial_fit(X, y)
    n_samples += 1

# Display results
print('iSOUP Tree regressor example')
print('{} samples analyzed.'.format(n_samples))
print('Mean absolute error: {}'.format(np.mean(np.abs(y_true - y_pred))))
