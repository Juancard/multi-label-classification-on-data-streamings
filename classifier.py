import sys
import os
import time
import json
import logging
import argparse
import numpy as np
from toolz import pipe, curry
import pandas as pd
from skmultilearn.dataset import load_dataset
from skmultiflow.data.data_stream import DataStream

from sklearn.linear_model import Perceptron
from skmultiflow.meta.multi_output_learner import MultiOutputLearner
from skmultiflow.meta import ClassifierChain
from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier

from common.helpers import (load_20ng_dataset, load_moa_stream,
                            evaluar, evaluation_metrics, repeatInstances)

parser = argparse.ArgumentParser("Script to classify streams")
parser.add_argument("-e", "--experiment", help="Description of the experiment")
parser.add_argument("-d", "--dataset", help="Name of skmultilearn dataset")
parser.add_argument("-s", "--streams", help="Path to stream", nargs='*')
parser.add_argument("-S", "--streamsnames", help="Names of streams", nargs='*')
parser.add_argument("-l", "--labels", type=int, help="Number of labels")
parser.add_argument("-r", "--repetitions", type=int,
                    help="Number of copies per instance", default=1)
parser.add_argument("-o", "--output", help="Directory to save output.",
                    default="experiments/{}_classification".format(time.strftime("%Y%m%d%H%M%S")))


def set_logger():
    """
    To print logs
    """
    logging.basicConfig(level=logging.INFO)
    logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    return logging


def to_absolute_path(dir_path, path):
    if os.path.isabs(path):
        return path
    return os.path.join(dir_path, path)


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path


def filename_path(name, dataset_name, output_dir, ext="csv"):
    filename = "{}_{}.{}".format(dataset_name, name, ext)
    return os.path.join(output_dir, filename)


def load_given_dataset(dataset):
    if dataset.lower() == "20ng":
        return load_20ng_dataset()
    return load_dataset(dataset, 'undivided')


def main():
    logging = set_logger()
    args = parser.parse_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    to_absolute = curry(to_absolute_path)(dir_path)
    output_dir = pipe(
        args.output,
        to_absolute,
        create_path_if_not_exists
    )
    metadata = {
        "experimento": args.experiment or "",
        "command": " ".join(sys.argv),
        "date": time.strftime("%Y%m%d%H%M%S"),
    }

    if args.dataset:

        #### DATASET CLASSIFICATION ######

        logging.info("Classifying dataset %s", args.dataset)
        logging.info("Loading dataset: %s", args.dataset)
        x_stream, y_stream, feature_names, label_names = load_given_dataset(
            args.dataset)
        logging.info("Copies per instance: %s", args.repetitions)
        x_stream, y_stream = repeatInstances(
            x_stream.todense(), y_stream.todense(), copies=args.repetitions)

        data_stream = DataStream(data=x_stream,
                                 y=y_stream, name=args.dataset)
        cardinality = sum(np.sum(y_stream, axis=1)
                          ) / y_stream.shape[0]
        metadata["dataset"] = {
            "name": args.dataset,
            "instances": data_stream.n_remaining_samples(),
            "X_shape": x_stream.shape,
            "y_shape": y_stream.shape,
            "cardinality": cardinality,
            "label_names": [i[0] for i in label_names],
            "copies": args.repetitions
        }

        train_stats = []
        eval_stats = []

        model = "Binary Relevance"
        logging.info(model)
        classifier_br = MultiOutputLearner(
            Perceptron()
        )
        stats_br, true_labels, predictions = evaluar(
            data_stream,
            classifier_br,
            0.1,
            logging=logging
        )
        stats_br.update({"model": model})
        train_stats.append(stats_br)
        eval_br = {}
        if true_labels is not None and predictions is not None:
            logging.info("Evaluating...")
            eval_br = evaluation_metrics(
                true_labels,
                predictions,
                stats_br["start_time"],
                stats_br["end_time"]
            )

        eval_br.update({"model": model})
        eval_stats.append(eval_br)
        data_stream.restart()

        model = "Classifier Chain"
        logging.info(model)
        classifier_cc = ClassifierChain(
            Perceptron()
        )
        stats_cc, true_labels, predictions = evaluar(
            data_stream,
            classifier_cc,
            0.1,
            logging=logging
        )
        stats_cc.update({"model": model})
        train_stats.append(stats_cc)
        eval_cc = {}
        if true_labels is not None and predictions is not None:
            logging.info("Evaluating...")
            eval_cc = evaluation_metrics(
                true_labels, predictions, stats_cc["start_time"], stats_cc["end_time"])

        eval_cc.update({"model": model})
        eval_stats.append(eval_cc)
        data_stream.restart()

        model = "Label Combination Hoeffding Tree"
        logging.info(model)
        classifier_lcht = LabelCombinationHoeffdingTreeClassifier(
            n_labels=data_stream.n_targets
        )
        stats_lcht, true_labels, predictions = evaluar(
            data_stream,
            classifier_lcht,
            0.1,
            logging=logging,
            train_logs_max=20
        )
        stats_lcht.update({"model": model})
        train_stats.append(stats_lcht)
        eval_lcht = {}
        if true_labels is not None and predictions is not None:
            logging.info("Evaluating...")
            eval_lcht = evaluation_metrics(
                true_labels,
                predictions,
                stats_lcht["start_time"],
                stats_lcht["end_time"]
            )
        eval_lcht.update({"model": model})
        eval_stats.append(eval_lcht)
        data_stream.restart()

        # Limpia memoria
        del x_stream, y_stream, data_stream

        logging.info("Saving training stats...")
        pd.DataFrame.from_dict(train_stats).to_csv(
            os.path.join(
                output_dir, args.dataset + "_train.csv"
            )
        )
        logging.info("Saving evaluation results...")
        pd.DataFrame.from_dict(eval_stats).to_csv(
            os.path.join(
                output_dir, args.dataset + "_eval.csv"
            )
        )

        #### FIN DATASET CLASSIFICATION ######

    #### STREAM ANALYSIS ######

    if args.streams:
        stream_names = args.streamsnames or []
        if len(stream_names) != len(args.streams):
            logging.error(
                "La cantidad de streams y la cantidad de nombres de streams no coinciden.")
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

                #### FIN STREAM ANALYSIS ######

    logging.info("Saving metadata")
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f_p:
        json.dump(metadata, f_p, indent=4)
    logging.info("Files saved in %s", output_dir)


if __name__ == "__main__":
    main()
