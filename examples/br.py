from skmultiflow.data import MultilabelGenerator
from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.data.file_stream import FileStream
from sklearn.linear_model import Perceptron
from skmultiflow.metrics import hamming_score
# Setup the file stream
stream = MultilabelGenerator(random_state=1, n_samples=200,
                             n_targets=5, n_features=10)
ht = HoeffdingTreeClassifier()
br = MultiOutputLearner(ht)
# Setup the pipeline
# Pre training the classifier with 150 samples
X, y = stream.next_sample(150)
br.partial_fit(X, y, classes=stream.target_values)
# Keeping track of sample count, true labels and predictions to later
# compute the classifier's hamming score
count = 0
true_labels = []
predicts = []
while stream.has_more_samples():
    X, y = stream.next_sample()
    p = br.predict(X)
    br.partial_fit(X, y)
    predicts.extend(p)
    true_labels.extend(y)
    count += 1

perf = hamming_score(true_labels, predicts)
print('Total samples analyzed: ' + str(count))
print("The classifier's static Hamming score    : " + str(perf))
