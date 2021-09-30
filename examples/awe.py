# Imports
from skmultiflow.data import SEAGenerator
from skmultiflow.meta import AccuracyWeightedEnsembleClassifier

# Setting up a data stream
stream = SEAGenerator(random_state=1)

# Setup Accuracy Weighted Ensemble Classifier
awe = AccuracyWeightedEnsembleClassifier()

# Setup variables to control loop and track performance
n_samples = 0
correct_cnt = 0
max_samples = 200

# Train the classifier with the samples provided by the data stream
while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = awe.predict(X)
    if y[0] == y_pred[0]:
        correct_cnt += 1
    awe.partial_fit(X, y)
    n_samples += 1

# Display results
print('{} samples analyzed.'.format(n_samples))
print('Accuracy Weighted Ensemble accuracy: {}'.format(
    correct_cnt / n_samples
))
