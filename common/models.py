from sklearn.linear_model import Perceptron, SGDClassifier
from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.meta import (
    ClassifierChain,
    AccuracyWeightedEnsemble,
    DynamicWeightedMajorityMultiLabel,
    OzaBaggingMLClassifier,
    MajorityEnsembleMultilabel,
    MonteCarloClassifierChain,
    ProbabilisticClassifierChain,
)
from skmultiflow.bayes import NaiveBayes
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.trees import (
    LabelCombinationHoeffdingTreeClassifier,
    iSOUPTreeRegressor,
    HoeffdingTreeClassifier,
)

SUPPORTED_MODELS = {
    "br": {
        "name": "Binary Relevance - Perceptron",
        "model": lambda data_stream: MultiOutputLearner(
            Perceptron(), n_targets=data_stream.n_targets
        ),
        "ensemble": False,
    },
    "br_ht": {
        "name": "Binary Relevance - Hoeffding Tree",
        "model": lambda data_stream: MultiOutputLearner(
            HoeffdingTreeClassifier(), n_targets=data_stream.n_targets
        ),
        "ensemble": False,
    },
    "br_nb": {
        "name": "Binary Relevance - Naive Bayes",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(), n_targets=data_stream.n_targets
        ),
        "ensemble": False,
    },
    "cc": {
        "name": "Classifier Chain - Perceptron",
        "model": lambda data_stream: ClassifierChain(
            Perceptron(), n_targets=data_stream.n_targets
        ),
        "ensemble": False,
    },
    "cc_ht": {
        "name": "Binary Relevance - HoeffdingTreeClassifier",
        "model": lambda data_stream: ClassifierChain(
            PerceptronMask(), n_targets=data_stream.n_targets
        ),
        "ensemble": False,
    },
    "cc_nb": {
        "name": "Classifier Chain - Naive Bayes",
        "model": lambda data_stream: ClassifierChain(
            NaiveBayes(), n_targets=data_stream.n_targets
        ),
        "ensemble": False,
    },
    "mcc": {
        "name": "Monte Carlo Sampling Classifier Chains",
        "model": lambda _: MonteCarloClassifierChain(),
        "ensemble": False,
    },
    "pcc": {
        "name": "Probabilistic Sampling Classifier Chains",
        "model": lambda _: ProbabilisticClassifierChain(
            SGDClassifier(max_iter=100, loss="log", random_state=1)
        ),
        "ensemble": False,
    },
    "lcht": {
        "name": "Label Combination Hoeffding Tree",
        "model": lambda data_stream: LabelCombinationHoeffdingTreeClassifier(
            n_labels=data_stream.n_targets,
            stop_mem_management=True,
            memory_estimate_period=100,
            remove_poor_atts=False,  # There is a bug when True
        ),
        "ensemble": False,
    },
    "awec": {
        "name": "Accuracy Weighted Ensemble Classifier",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(), n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, _: AccuracyWeightedEnsemble(
            base_estimator=model
        ),
    },
    "me": {
        "name": "Majority Ensemble Classifier",
        "model": lambda data_stream: [
            ClassifierChain(NaiveBayes(), n_targets=data_stream.n_targets),
            ClassifierChain(
                HoeffdingTreeClassifier(), n_targets=data_stream.n_targets
            ),
            MultiOutputLearner(NaiveBayes(), n_targets=data_stream.n_targets),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3,
        ),
    },
    "me_lcht": {
        "name": "Majority Ensemble Classifier",
        "model": lambda data_stream: [
            ClassifierChain(NaiveBayes(), n_targets=data_stream.n_targets),
            LabelCombinationHoeffdingTreeClassifier(
                n_labels=data_stream.n_targets,
                stop_mem_management=True,
                memory_estimate_period=100,
                remove_poor_atts=False,  # There is a bug when True
            ),
            MultiOutputLearner(NaiveBayes(), n_targets=data_stream.n_targets),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3,
        ),
    },
    "me2": {
        "name": "Majority Ensemble Classifier with poisson",
        "model": lambda data_stream: [
            ClassifierChain(NaiveBayes(), n_targets=data_stream.n_targets),
            ClassifierChain(
                HoeffdingTreeClassifier(), n_targets=data_stream.n_targets
            ),
            MultiOutputLearner(NaiveBayes(), n_targets=data_stream.n_targets),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3,
            sampling="poisson",
        ),
    },
    "me2_lcht": {
        "name": "Majority Ensemble Classifier with poisson and lcht",
        "model": lambda data_stream: [
            ClassifierChain(NaiveBayes(), n_targets=data_stream.n_targets),
            LabelCombinationHoeffdingTreeClassifier(
                n_labels=data_stream.n_targets,
                stop_mem_management=True,
                memory_estimate_period=100,
                remove_poor_atts=False,  # There is a bug when True
            ),
            MultiOutputLearner(NaiveBayes(), n_targets=data_stream.n_targets),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3,
            sampling="poisson",
        ),
    },
    "dwmc_br": {
        "name": "Dynamically Weighted Majority Classifier (br)",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(), n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: DynamicWeightedMajorityMultiLabel(
            labels=stream.n_targets,
            base_estimator=model,
            period=round(
                stream.n_remaining_samples() / (20 * 2)
            ),  # Twice every window
            beta=0.5,
            n_estimators=10,
        ),
    },
    "dwmc_cc": {
        "name": "Dynamically Weighted Majority Classifier (cc)",
        "model": lambda data_stream: ClassifierChain(
            NaiveBayes(), n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: DynamicWeightedMajorityMultiLabel(
            labels=stream.n_targets,
            base_estimator=model,
            period=round(
                stream.n_remaining_samples() / (20 * 2)
            ),  # Twice every window
            beta=0.5,
            n_estimators=10,
        ),
    },
    "oza_ml_br_nb": {
        "name": "OzaBagging (br) / ebr - nb",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(), n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model, n_estimators=10
        ),
    },
    "oza_ml_br_ht": {
        "name": "OzaBagging (br) / ebr - ht",
        "model": lambda data_stream: MultiOutputLearner(
            HoeffdingTreeClassifier(), n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model, n_estimators=10
        ),
    },
    "oza_ml_cc_nb": {
        "name": "OzaBagging (cc) / ecc - nb",
        "model": lambda data_stream: ClassifierChain(
            NaiveBayes(), n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model, n_estimators=10
        ),
    },
    "oza_ml_cc_ht": {
        "name": "OzaBagging (cc) / ecc - ht",
        "model": lambda data_stream: ClassifierChain(
            HoeffdingTreeClassifier(), n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model, n_estimators=10
        ),
    },
    "isoup": {
        "name": "iSoup-Tree",
        "model": lambda _: iSOUPTreeRegressor(),
        "ensemble": False,
    },
}
