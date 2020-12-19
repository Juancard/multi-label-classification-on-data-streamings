# Imports
from skmultiflow.data import MultilabelGenerator
from skmultiflow.meta import MultiOutputLearner
from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier, HoeffdingTreeClassifier
from skmultiflow.metrics import hamming_score

# Setting up a data stream
stream = MultilabelGenerator(random_state=1, n_samples=200,
                             n_targets=5, n_features=10)

# Setup Label Combination Hoeffding Tree classifier
lc_ht = LabelCombinationHoeffdingTreeClassifier(n_labels=stream.n_targets)

# Setup variables to control loop and track performance
n_samples = 0
max_samples = 200
true_labels = []
predicts = []

# Train the estimator with the samples provided by the data stream
while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = lc_ht.predict(X)
    lc_ht.partial_fit(X, y, classes=stream.target_values)
    predicts.extend(y_pred)
    true_labels.extend(y)
    n_samples += 1

# Display results
perf = hamming_score(true_labels, predicts)
print('{} samples analyzed.'.format(n_samples))
print('Label Combination Hoeffding Tree Hamming score: ' + str(perf))
