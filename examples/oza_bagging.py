from skmultiflow.meta import OzaBaggingClassifier
from skmultiflow.lazy import KNNClassifier
from skmultiflow.data import SEAGenerator
# Setting up the stream
stream = SEAGenerator(1, noise_percentage=0.07)
# Setting up the OzaBagging classifier to work with KNN as base estimator
clf = OzaBaggingClassifier(base_estimator=KNNClassifier(
    n_neighbors=8, max_window_size=2000, leaf_size=30), n_estimators=2)
# Keeping track of sample count and correct prediction count
sample_count = 0
corrects = 0

# Pre training the classifier with 200 samples
X, y = stream.next_sample(200)
clf = clf.partial_fit(X, y, classes=stream.target_values)
for i in range(2000):
    X, y = stream.next_sample()
    pred = clf.predict(X)
    clf = clf.partial_fit(X, y)
    if pred is not None:
        if y[0] == pred[0]:
            corrects += 1
    sample_count += 1

# Displaying the results
print(str(sample_count) + ' samples analyzed.')
print('OzaBaggingClassifier performance: ' + str(corrects / sample_count))
