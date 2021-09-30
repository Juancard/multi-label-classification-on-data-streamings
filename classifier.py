import sys
import os
import time
import json
import logging
import argparse
import numpy as np
from functools import reduce
from toolz import pipe, curry
import pandas as pd
from urllib.error import HTTPError
from skmultiflow.data.data_stream import DataStream

from sklearn.linear_model import Perceptron, SGDClassifier
from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.meta import ClassifierChain,\
    AccuracyWeightedEnsemble, \
    DynamicWeightedMajorityMultiLabel, \
    OzaBaggingMLClassifier, \
    MajorityEnsembleMultilabel, \
    MonteCarloClassifierChain, \
    ProbabilisticClassifierChain
from skmultiflow.bayes import NaiveBayes
from skmultiflow.neural_networks import PerceptronMask
from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier,\
    iSOUPTreeRegressor, \
    HoeffdingTreeClassifier

from common.helpers import (
    evaluar, repeatInstances
)
from common.my_datasets import (
    load_given_dataset, load_moa_stream, available_datasets,
)
from common.evaluation_metrics import evaluation_metrics


TIME_STR = "%Y%m%d_%H%M%S"

SUPPORTED_MODELS = {
    "br": {
        "name": "Binary Relevance - Perceptron",
        "model": lambda data_stream: MultiOutputLearner(
            Perceptron(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": False
    },
    "br_ht": {
        "name": "Binary Relevance - Hoeffding Tree",
        "model": lambda data_stream: MultiOutputLearner(
            HoeffdingTreeClassifier(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": False
    },
    "br_nb": {
        "name": "Binary Relevance - Naive Bayes",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": False
    },
    "cc": {
        "name": "Classifier Chain - Perceptron",
        "model": lambda data_stream: ClassifierChain(
            Perceptron(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": False
    },
    "cc_ht": {
        "name": "Binary Relevance - HoeffdingTreeClassifier",
        "model": lambda data_stream: ClassifierChain(
            PerceptronMask(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": False
    },
    "cc_nb": {
        "name": "Classifier Chain - Naive Bayes",
        "model": lambda data_stream: ClassifierChain(
            NaiveBayes(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": False
    },
    "mcc": {
        "name": "Monte Carlo Sampling Classifier Chains",
        "model": lambda _: MonteCarloClassifierChain(),
        "ensemble": False
    },
    "pcc": {
        "name": "Probabilistic Sampling Classifier Chains",
        "model": lambda _: ProbabilisticClassifierChain(
            SGDClassifier(
                max_iter=100,
                loss='log',
                random_state=1
            )
        ),
        "ensemble": False
    },
    "lcht": {
        "name": "Label Combination Hoeffding Tree",
        "model": lambda data_stream: LabelCombinationHoeffdingTreeClassifier(
            n_labels=data_stream.n_targets,
            stop_mem_management=True,
            memory_estimate_period=100,
            remove_poor_atts=False  # There is a bug when True
        ),
        "ensemble": False
    },
    "awec": {
        "name": "Accuracy Weighted Ensemble Classifier",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, _: AccuracyWeightedEnsemble(
            base_estimator=model
        )
    },
    "me": {
        "name": "Majority Ensemble Classifier",
        "model": lambda data_stream: [
            ClassifierChain(
                NaiveBayes(),
                n_targets=data_stream.n_targets
            ),
            ClassifierChain(
                HoeffdingTreeClassifier(),
                n_targets=data_stream.n_targets
            ),
            MultiOutputLearner(
                NaiveBayes(),
                n_targets=data_stream.n_targets
            ),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3
        )
    },
    "me_lcht": {
        "name": "Majority Ensemble Classifier",
        "model": lambda data_stream: [
            ClassifierChain(
                NaiveBayes(),
                n_targets=data_stream.n_targets
            ),
            LabelCombinationHoeffdingTreeClassifier(
                n_labels=data_stream.n_targets,
                stop_mem_management=True,
                memory_estimate_period=100,
                remove_poor_atts=False  # There is a bug when True
            ),
            MultiOutputLearner(
                NaiveBayes(),
                n_targets=data_stream.n_targets
            ),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3
        )
    },
    "me2": {
        "name": "Majority Ensemble Classifier with poisson",
        "model": lambda data_stream: [
            ClassifierChain(
                NaiveBayes(),
                n_targets=data_stream.n_targets
            ),
            ClassifierChain(
                HoeffdingTreeClassifier(),
                n_targets=data_stream.n_targets
            ),
            MultiOutputLearner(
                NaiveBayes(), n_targets=data_stream.n_targets
            ),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3,
            sampling="poisson"
        )
    },
    "me2_lcht": {
        "name": "Majority Ensemble Classifier with poisson and lcht",
        "model": lambda data_stream: [
            ClassifierChain(
                NaiveBayes(),
                n_targets=data_stream.n_targets
            ),
            LabelCombinationHoeffdingTreeClassifier(
                n_labels=data_stream.n_targets,
                stop_mem_management=True,
                memory_estimate_period=100,
                remove_poor_atts=False  # There is a bug when True
            ),
            MultiOutputLearner(
                NaiveBayes(),
                n_targets=data_stream.n_targets
            ),
        ],
        "ensemble": lambda model, stream: MajorityEnsembleMultilabel(
            labels=stream.n_targets,
            base_estimator=model,
            base_estimators=model if isinstance(model, list) else False,
            period=round(stream.n_remaining_samples() / 20),
            beta=0.5,
            n_estimators=3,
            sampling="poisson"
        )
    },
    "dwmc_br": {
        "name": "Dynamically Weighted Majority Classifier (br)",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: DynamicWeightedMajorityMultiLabel(
            labels=stream.n_targets,
            base_estimator=model,
            period=round(stream.n_remaining_samples() / \
                         (20 * 2)),  # Twice every window
            beta=0.5,
            n_estimators=10
        )
    },
    "dwmc_cc": {
        "name": "Dynamically Weighted Majority Classifier (cc)",
        "model": lambda data_stream: ClassifierChain(
            NaiveBayes(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: DynamicWeightedMajorityMultiLabel(
            labels=stream.n_targets,
            base_estimator=model,
            period=round(stream.n_remaining_samples() / \
                         (20 * 2)),  # Twice every window
            beta=0.5,
            n_estimators=10
        )
    },
    "oza_ml_br_nb": {
        "name": "OzaBagging (br) / ebr - nb",
        "model": lambda data_stream: MultiOutputLearner(
            NaiveBayes(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model,
            n_estimators=10
        )
    },
    "oza_ml_br_ht": {
        "name": "OzaBagging (br) / ebr - ht",
        "model": lambda data_stream: MultiOutputLearner(
            HoeffdingTreeClassifier(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model,
            n_estimators=10
        )
    },
    "oza_ml_cc_nb": {
        "name": "OzaBagging (cc) / ecc - nb",
        "model": lambda data_stream: ClassifierChain(
            NaiveBayes(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model,
            n_estimators=10
        )
    },
    "oza_ml_cc_ht": {
        "name": "OzaBagging (cc) / ecc - ht",
        "model": lambda data_stream: ClassifierChain(
            HoeffdingTreeClassifier(),
            n_targets=data_stream.n_targets
        ),
        "ensemble": lambda model, stream: OzaBaggingMLClassifier(
            base_estimator=model,
            n_estimators=10
        )
    },
    "isoup": {
        "name": "iSoup-Tree",
        "model": lambda _: iSOUPTreeRegressor(),
        "ensemble": False
    },
}
DEFAULT_DATASETS = ["enron", "mediamill", "20NG"]

parser = argparse.ArgumentParser("Script to classify streams")
parser.add_argument("-e", "--experiment", help="Description of the experiment")
parser.add_argument("-d", "--datasets",
                    help="List of skmultilearn datasets (including 20NG)",
                    nargs="*", default=DEFAULT_DATASETS)
parser.add_argument("-m", "--models", help="List of models to train data",
                    nargs='*', default=SUPPORTED_MODELS.keys())
parser.add_argument("-s", "--streams", help="Path to stream", nargs='*')
parser.add_argument("-S", "--streamsnames", help="Names of streams", nargs='*')
parser.add_argument("-l", "--labels", type=int, help="Number of labels")
parser.add_argument("-p", "--pretrainsize", type=float,
                    help="Pretrain size proportion (default=0)", default=0)
parser.add_argument("-c", "--copies", nargs="*",
                    help="Number of copies per instance for each dataset",
                    default=[])
parser.add_argument("-o", "--output", help="Directory to save output.",
                    default=False)
parser.add_argument("-v", "--verbose",
                    help="increase output verbosity", action="store_true")
parser.add_argument(
    "-f", "--catch", help="Catches errors during training",
    action="store_true")


def set_logger(verbosity):
    """
    To print logs
    """
    level = logging.DEBUG if verbosity else logging.INFO
    logging.basicConfig(level=level)
    logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    return logging


def to_absolute_path(dir_path, path):
    if os.path.isabs(path):
        return os.path.join(path)
    return os.path.join(dir_path, path)


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def filename_path(name, dataset_name, output_dir, ext="csv"):
    filename = "{}_{}.{}".format(dataset_name, name, ext)
    return os.path.join(output_dir, filename)


def valid_args(args):
    """ Validate arguments passed to this script. """
    try:
        all_datasets_available = available_datasets()
    except HTTPError:
        logging.warning(
            "Conection error: skmultilearn datasets are unavailable." +
            " Only local datasets will be available."
        )
    valid_dataset = True
    for i in args.datasets:
        if i.lower() not in all_datasets_available:
            logging.error("Dataset '%s' does not exists", i)
            valid_dataset = False
    if not valid_dataset:
        logging.info("Valid datasets are: %s", str(all_datasets_available))
        return False

    if args.copies == []:
        args.copies = ["1" for i in args.datasets]
    if len(args.copies) != len(args.datasets):
        logging.error(
            "Number of copies per instance of dataset (%s) " +
            "and number of datasets (%s) has to be equal.",
            len(args.copies),
            len(args.datasets)
        )
        return False
    copies_are_digits = reduce(
        lambda are_digits, copy: are_digits and copy.isdigit() and int(
            copy
        ) > 0,
        args.copies,
        True
    )
    if not copies_are_digits:
        logging.error("Copies have to be valid positive integers.")
        return False

    pretrain_size_is_prop = args.pretrainsize >= 0 and args.pretrainsize < 1
    if not pretrain_size_is_prop:
        logging.error(
            "Pretrain size has to be a value between 0 and 1, got {}".format(
                args.pretrainsize
            )
        )
        return False

    valid_models = True
    for i in args.models:
        if i.lower() not in SUPPORTED_MODELS.keys():
            logging.error("Model not supported '%s'", i)
            valid_models = False
    if not valid_models:
        logging.info("Valid models are: %s", SUPPORTED_MODELS.keys())
        return False

    return True


def main():
    args = parser.parse_args()
    logging = set_logger(args.verbose)
    if not valid_args(args):
        sys.exit(0)

    datasets = args.datasets
    models = [i.lower() for i in args.models]
    copies = [int(i) for i in args.copies]

    dir_path = os.path.dirname(os.path.realpath(__file__))
    to_absolute = curry(to_absolute_path)(dir_path)

    metadata = {
        "experimento": args.experiment or "",
        "command": " ".join(sys.argv),
        "date": time.strftime("%Y%m%d%H%M%S"),
        "models": models,
        "copies": copies,
        "datasets": []
    }
    logging.debug(metadata)

    # DATASET CLASSIFICATION ######
    all_train_data = []
    true_vs_pred = []
    logging.debug(datasets)
    for idx, dataset in enumerate(datasets):
        logging.info("Classifying dataset %s", dataset)
        logging.debug("Loading dataset: %s", dataset)
        x_stream, y_stream, _, label_names = load_given_dataset(dataset)
        logging.debug("Copies per instance: %s", copies[idx])
        x_stream, y_stream = repeatInstances(
            x_stream.todense(), y_stream.todense(), copies=copies[idx])

        data_stream = DataStream(data=x_stream,
                                 y=y_stream, name=dataset)
        cardinality = sum(np.sum(y_stream, axis=1)
                          ) / y_stream.shape[0]
        dataset_metadata = {
            "name": dataset,
            "instances": data_stream.n_remaining_samples(),
            "x_shape": x_stream.shape,
            "y_shape": y_stream.shape,
            "cardinality": cardinality,
            "label_names": [i[0] for i in label_names],
            "copies": copies[idx]
        }
        logging.debug(dataset_metadata)

        for model_id in models:
            model = SUPPORTED_MODELS[model_id]
            logging.info(model["name"])
            train_data = {"model": model["name"], "model_id": model_id,
                          "stream": data_stream.name, "copies": copies[idx]}
            train_stats, true_labels, predictions = evaluar(
                data_stream,
                model["model"](data_stream),
                pretrain_size=args.pretrainsize,
                ensemble=model["ensemble"],
                catch_errors=args.catch,
                logging=logging,
                train_logs_max=100000,
                window_size=20
            )
            eval_stats = {}
            if true_labels and predictions:
                logging.info("Evaluating...")
                eval_stats = evaluation_metrics(
                    true_labels,
                    predictions,
                    train_stats["start_time"],
                    train_stats["end_time"]
                )
                true_vs_pred.append({
                    "model": model_id,
                    "dataset": dataset,
                    "true": true_labels,
                    "pred": predictions
                })
            train_data.update(train_stats)
            train_data.update(eval_stats)
            all_train_data.append(train_data)
            data_stream.restart()

        metadata["datasets"].append(dataset_metadata)
        # Limpia memoria
        del x_stream, y_stream, data_stream

    # FIN DATASET CLASSIFICATION ######

    # STREAM ANALYSIS ######

    if args.streams:
        print("Stream classification. Not yet implemented.")
        sys.exit(0)
        stream_names = args.streamsnames or []
        if len(stream_names) != len(args.streams):
            logging.error(
                "La cantidad de streams y la cantidad de nombres" +
                " de streams no coinciden."
            )
            sys.exit(1)
            metadata["syn_streams"] = []
            for idx, i in enumerate(args.streams):
                stream_path = to_absolute(i)
                stream_name = stream_names[idx]

                logging.info("Classifying syn stream: %s", stream_name)

                logging.info("Loading syn stream to memory")
                _, y_syn, _, _ = load_moa_stream(stream_path, args.labels)

                cardinality = sum(
                    np.sum(y_syn.toarray(), axis=1)
                ) / y_syn.toarray().shape[0]

                metadata["syn_streams"].append({
                    "labels": args.labels,
                    "stream_path": stream_path,
                    "stream_name": stream_name,
                    "y_shape": y_syn.shape,
                    "cardinality": cardinality,
                })

                # FIN STREAM ANALYSIS ######

    default_output_path = "experiments/"
    dest_dir = "{}_classification".format(
        time.strftime(TIME_STR)
    )
    output_rel = os.path.join(
        args.output if args.output else default_output_path,
        dest_dir
    )
    output_dir = pipe(
        output_rel,
        to_absolute,
        create_path_if_not_exists
    )

    logging.info("Saving results in a csv...")
    pd.DataFrame.from_dict(all_train_data).to_csv(
        os.path.join(
            output_dir, "results.csv"
        )
    )

    logging.info("Saving true_vs_pred in a csv...")
    for i in true_vs_pred:
        true_file = '{}_{}_true.csv'.format(i["dataset"], i["model"])
        pred_file = '{}_{}_predicted.csv'.format(i["dataset"], i["model"])
        np.savetxt(os.path.join(output_dir, true_file),
                   i["true"], delimiter=',')
        np.savetxt(os.path.join(output_dir, pred_file),
                   i["pred"], delimiter=',')

    logging.info("Saving metadata")
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f_p:
        json.dump(metadata, f_p, indent=4)

    logging.info("Files saved in %s", output_dir)


if __name__ == "__main__":
    main()
