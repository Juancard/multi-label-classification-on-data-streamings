# Example of a dynamic weighted ensemble classifier

# Imports
import time
from skmultiflow.data import MultilabelGenerator
from skmultiflow.meta import (MultiOutputLearner,
                              ClassifierChain,
                              DynamicWeightedMajorityMultiLabel,
                              MajorityEnsembleMultilabel,
                              OzaBaggingMLClassifier)
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.bayes import NaiveBayes
from skmultiflow.metrics import hamming_score
from functools import partial
# from sklearn.linear_model import SGDClassifier

ESTIMATORS = ["BR", "CC"]


def data_stream(n_features, n_labels, max_samples):
    return MultilabelGenerator(
        n_samples=max_samples,
        random_state=1,
        n_features=n_features,
        n_targets=n_labels
    )


def build_model(stream, ml_model):
    if ml_model == "BR":
        return MultiOutputLearner(
            NaiveBayes(),
            n_targets=stream.n_targets
        )
    elif ml_model == "CC":
        return ClassifierChain(
            NaiveBayes(),
            n_targets=stream.n_targets
        )


def train(stream_generator, model):

    # Keeping track of sample count, true labels and predictions to later
    # compute the classifier's hamming score
    stream = stream_generator()
    count = 0
    true_labels = []
    predicts = []
    start_time = time.time()

    while stream.has_more_samples() and count < stream.n_samples:
        X, y = stream.next_sample()
        p = model.predict(X)
        model.partial_fit(X, y, classes=stream.target_values[0])
        predicts.extend(p)
        true_labels.extend(y)
        count += 1

    end_time = time.time()
    perf = hamming_score(true_labels, predicts)

    # Display results
    print('{} ({}): HS: {} - Time: {}s'.format(
        model.get_info().split("(")[0],
        model.base_estimator,
        perf,
        (end_time-start_time)
    ))


def models_for(stream, estimator):
    return [
        build_model(stream, estimator),
        DynamicWeightedMajorityMultiLabel(
            labels=stream.n_targets,
            base_estimator=build_model(stream, estimator),
            n_estimators=10,
            period=round(stream.n_remaining_samples() /
                         (20 * 2)),  # Twice every window
            beta=0.5,
        ),
        OzaBaggingMLClassifier(
            base_estimator=build_model(stream, estimator),
            n_estimators=10
        )
    ]


def train_models_on(stream_generator, estimator):
    list(map(
        partial(train, stream_generator),
        models_for(stream_generator(), estimator)
    ))


def my_model(stream, poisson=False):
    return MajorityEnsembleMultilabel(
        labels=stream.n_targets,
        base_estimators=[
            build_model(stream, "CC"),
            ClassifierChain(
                HoeffdingTreeClassifier(),
                n_targets=stream.n_targets
            ),
            build_model(stream, "BR"),
        ],
        period=round(stream.n_remaining_samples() / 20),
        beta=0.5,
        n_estimators=3,
        sampling="poisson" if poisson else None
    )


def main():
    n_features = 10
    n_labels = 5
    max_samples = 1000

    def stream_generator():
        return data_stream(n_features, n_labels, max_samples)

    list(map(
        partial(
            train_models_on,
            stream_generator
        ),
        ESTIMATORS
    ))

    train(
        stream_generator,
        my_model(stream_generator())
    )
    train(
        stream_generator,
        my_model(stream_generator(), poisson=True)
    )


if __name__ == "__main__":
    main()
