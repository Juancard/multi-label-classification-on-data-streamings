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
import uuid
from urllib.error import HTTPError

from common.models import SUPPORTED_MODELS
from common.multi_label_dataset import MultiLabelDataset
from common.helpers import evaluar
from common.my_datasets import (
    available_datasets,
)
from common.evaluation_metrics import evaluation_metrics


TIME_STR = "%Y%m%d_%H%M%S"

DEFAULT_DATASETS = ["enron", "mediamill", "20NG"]

parser = argparse.ArgumentParser("Script to classify streams")
parser.add_argument("-e", "--experiment", help="Description of the experiment")
parser.add_argument(
    "-d",
    "--datasets",
    help="List of skmultilearn datasets (including 20NG)",
    nargs="*",
    default=DEFAULT_DATASETS,
)
parser.add_argument(
    "-m",
    "--models",
    help="List of models to train data",
    nargs="*",
    default=SUPPORTED_MODELS.keys(),
)
parser.add_argument("-s", "--streams", help="Path to stream", nargs="*")
parser.add_argument("-S", "--streamsnames", help="Names of streams", nargs="*")
parser.add_argument("-l", "--labels", type=int, help="Number of labels")
parser.add_argument(
    "-p",
    "--pretrainsize",
    type=float,
    help="Pretrain size proportion (default=0)",
    default=0,
)
parser.add_argument(
    "-c",
    "--copies",
    nargs="*",
    help="Number of copies per instance for each dataset",
    default=[],
)
parser.add_argument(
    "-o", "--output", help="Directory to save output.", default=False
)
parser.add_argument(
    "-v", "--verbose", help="increase output verbosity", action="store_true"
)
parser.add_argument(
    "-f", "--catch", help="Catches errors during training", action="store_true"
)
parser.add_argument(
    "-k",
    "--save-predictions",
    help="Saves predictions for each instance on a csv file",
    action="store_true",
)


def set_logger(verbosity):
    """
    To print logs
    """
    level = logging.DEBUG if verbosity else logging.INFO
    logging.basicConfig(level=level)
    logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
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
    """Validate arguments passed to this script."""
    try:
        all_datasets_available = available_datasets()
    except HTTPError:
        logging.warning(
            "Conection error: skmultilearn datasets are unavailable."
            + " Only local datasets will be available."
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
            "Number of copies per instance of dataset (%s) "
            + "and number of datasets (%s) has to be equal.",
            len(args.copies),
            len(args.datasets),
        )
        return False
    copies_are_digits = reduce(
        lambda are_digits, copy: are_digits
        and copy.isdigit()
        and int(copy) > 0,
        args.copies,
        True,
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


def prepare_dataset(logging, metadata, dataset, copies):
    logging.info("Classifying dataset %s", dataset)
    logging.debug("Loading dataset: %s", dataset)
    multi_label_dataset = MultiLabelDataset(dataset)
    if copies > 1:
        logging.debug(
            "Loading dataset with copies. Copies per instance: {}".format(
                copies
            )
        )
        multi_label_dataset = MultiLabelDataset.withRepetitions(
            dataset, copies=copies
        )
    logging.debug("Loaded dataset: {}".format(multi_label_dataset.name))

    dataset_metadata = multi_label_dataset.metadata()
    dataset_metadata.update({"copies": copies})
    metadata["datasets"].append(dataset_metadata)
    logging.debug(dataset_metadata)
    return multi_label_dataset


def train(logging, pretrain_size, catch_errors, multi_label_dataset, model_id):
    model = SUPPORTED_MODELS[model_id]
    logging.info(model["name"])
    train_data = {
        "model": model["name"],
        "model_id": model_id,
        "dataset_name": multi_label_dataset.name,
        "stream": multi_label_dataset.to_data_stream().name,
    }
    train_stats, true_labels, predictions = evaluar(
        multi_label_dataset.to_data_stream(),
        model["model"](multi_label_dataset.to_data_stream()),
        pretrain_size=pretrain_size,
        ensemble=model["ensemble"],
        catch_errors=catch_errors,
        logging=logging,
        train_logs_max=100000,
        window_size=20,
    )
    train_stats.update(train_data)
    return {
        "train_stats": train_stats,
        "true_labels": true_labels,
        "predictions": predictions,
    }


def evaluate(performance):
    evaluation = {}
    train_ok = performance["true_labels"] and performance["predictions"]
    if train_ok:
        evaluation = evaluation_metrics(
            performance["true_labels"],
            performance["predictions"],
            performance["train_stats"]["start_time"],
            performance["train_stats"]["end_time"],
        )
    performance.update({"evaluation": evaluation})
    return performance


def clean(save_predictions, performance):
    """Removes data from memory if is not needed"""
    if not save_predictions:
        performance.update(
            {
                "true_labels": None,
                "predictions": None,
            }
        )
    return performance


def generate_output_dir(output=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    to_absolute = curry(to_absolute_path)(dir_path)
    default_output_path = "experiments/"
    dest_dir = "{}_classification_{}".format(
        time.strftime(TIME_STR), uuid.uuid1()
    )
    from_dir = output if output else default_output_path
    output_rel = os.path.join(from_dir, dest_dir)
    return pipe(output_rel, to_absolute, create_path_if_not_exists)


def save_prediction(logging, output_dir, performance):
    logging.info("Saving true_vs_pred in a csv...")
    dataset_name = performance["train_stats"]["dataset_name"]
    model_name = performance["train_stats"]["model_id"]
    true_file = "{}_{}_true.csv".format(dataset_name, model_name)
    pred_file = "{}_{}_predicted.csv".format(dataset_name, model_name)
    np.savetxt(
        os.path.join(output_dir, true_file),
        performance["true_labels"],
        delimiter=",",
    )
    np.savetxt(
        os.path.join(output_dir, pred_file),
        performance["predictions"],
        delimiter=",",
    )


def show_evaluation_preview(logging, performance):
    logging.info(performance["evaluation"])
    return performance


def save_experiments_results(
    logging,
    save_predictions,
    output_dir,
    metadata,
    performances,
):
    def merge_train_and_eval(p):
        return {**p["train_stats"], **p["evaluation"]}

    logging.info("Saving results in a csv...")
    results = [merge_train_and_eval(p) for p in performances]
    pd.DataFrame.from_dict(results).to_csv(
        os.path.join(output_dir, "results.csv")
    )

    if save_predictions:
        [save_prediction(logging, output_dir, p) for p in performances]

    logging.info("Saving metadata")
    with open(os.path.join(output_dir, "metadata.json"), "w") as f_p:
        json.dump(metadata, f_p, indent=4)

    logging.info("Files saved in %s", output_dir)


def classify_dataset(
    logging,
    metadata,
    pretrainsize,
    catch,
    save_predictions,
    models,
    dataset,
    copies,
):

    prepare_data = curry(prepare_dataset, logging, metadata, dataset, copies)
    train_dataset_on_model = curry(train, logging, pretrainsize, catch)
    clean_up = curry(clean, save_predictions)
    evaluation_preview = curry(show_evaluation_preview, logging)

    def classify_on_models(multi_label_dataset):
        train_model = curry(train_dataset_on_model, multi_label_dataset)
        return list(
            map(
                lambda model_id: pipe(
                    model_id,
                    train_model,
                    evaluate,
                    clean_up,
                    evaluation_preview,
                ),
                models,
            ),
        )

    def concat_performances(performances):
        return reduce(lambda x, y: x + y, performances)

    return pipe(
        None,
        lambda x: prepare_data(),
        classify_on_models,
        concat_performances,
    )


def main():

    args = parser.parse_args()
    logging = set_logger(args.verbose)
    if not valid_args(args):
        sys.exit(0)

    datasets = args.datasets
    models = [i.lower() for i in args.models]
    copies = [int(i) for i in args.copies]

    metadata = {
        "experimento": args.experiment or "",
        "command": " ".join(sys.argv),
        "date": time.strftime("%Y%m%d%H%M%S"),
        "models": models,
        "copies": copies,
        "datasets": [],
    }
    logging.debug(metadata)

    # DATASET CLASSIFICATION ######
    logging.debug(datasets)

    classify = curry(
        classify_dataset,
        logging,
        metadata,
        args.pretrainsize,
        args.catch,
        args.save_predictions,
        models,
    )

    def call_classify(x):
        dataset_idx, dataset_name = x
        return classify(dataset_name, copies[dataset_idx])

    performances = list(map(call_classify, enumerate(datasets)))

    # STREAM ANALYSIS ######
    if args.streams:
        print("Stream classification. Not yet implemented.")
        sys.exit(0)

    save_experiments_results(
        logging,
        args.save_predictions,
        generate_output_dir(args.output),
        metadata,
        performances,
    )


if __name__ == "__main__":
    main()
