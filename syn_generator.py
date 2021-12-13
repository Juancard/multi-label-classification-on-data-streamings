import sys
import os
import argparse
import time
import csv
import json
import logging
import numpy as np
import pandas as pd
from toolz import pipe, curry
from sklearn.metrics import mean_absolute_error
from skmultiflow.data.data_stream import DataStream
from common.helpers import (
    generate_labels_relationship,
    labels_relationship_graph,
    generate_labels_skew,
    labels_skew_graph,
    top_labels_combinations,
    generate_labels_distribution,
    labels_distribution_graph,
    labels_distribution_mae_graph,
    top_features,
)
from common.my_datasets import (
    load_given_dataset,
    load_moa_stream,
)

PLOT_COLORS = [
    "red",
    "blue",
    "green",
    "orange",
    "violet",
    "yellow",
    "brown",
    "gray",
]
SKEW_TOP_COMBINATIONS = 50
TIME_STR = "%Y%m%d_%H%M%S"

parser = argparse.ArgumentParser(
    "Script to analyze a generated synthetic dataset"
)
parser.add_argument("-e", "--experiment", help="Description of the experiment")
parser.add_argument("-d", "--dataset", help="Name of skmultilearn dataset")
parser.add_argument("-s", "--streams", help="Path to stream", nargs="*")
parser.add_argument("-S", "--streamsnames", help="Names of streams", nargs="*")
parser.add_argument("-l", "--labels", type=int, help="Number of labels")
parser.add_argument(
    "-o", "--output", help="Directory to save output.", default=False
)


def set_logger():
    """
    To print logs
    """
    logging.basicConfig(level=logging.INFO)
    logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    return logging


def to_absolute_path(dir_path, path):
    if os.path.isabs(path):
        return path
    return os.path.join(dir_path, path)


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path
    return path


def filename_path(name, dataset_name, output_dir, ext="csv"):
    filename = "{}_{}.{}".format(dataset_name, name, ext)
    return os.path.join(output_dir, filename)


def save_labels_relationship(
    output_dir, dataset_name, priors, coocurrences, conditional_matrix
):

    with open(
        filename_path("conditional", dataset_name, output_dir), "w"
    ) as f:
        writer = csv.writer(f)
        for row in conditional_matrix:
            writer.writerow(row)

    with open(filename_path("priors", dataset_name, output_dir), "w") as f:
        writer = csv.writer(f)
        writer.writerow(priors)

    with open(
        filename_path("coocurrences", dataset_name, output_dir), "w"
    ) as f:
        writer = csv.writer(f)
        for row in coocurrences:
            writer.writerow(row)


dir_path = os.path.dirname(os.path.realpath(__file__))
to_absolute = curry(to_absolute_path)(dir_path)


def create_output_dir(output_path="experiments/", suffix="syn"):
    dest_dir = "{}_{}".format(time.strftime(TIME_STR), suffix)
    output_rel = os.path.join(output_path, dest_dir)
    return pipe(output_rel, to_absolute, create_path_if_not_exists)


def main():
    logging = set_logger()
    args = parser.parse_args()
    output_dir = create_output_dir(
        output_path=args.output if args.output else None
    )
    metadata = {
        "experimento": args.experiment or "",
        "command": " ".join(sys.argv),
        "date": time.strftime("%Y%m%d%H%M%S"),
    }

    lk_plot_data = []
    ld_plot_data = []
    ld_mae_plot_data = []

    lk_top_labels_combination = {}

    if not args.dataset:
        print("Dataset not provided. Exiting.")
        sys.exit(0)

    # DATASET ANALYSIS

    logging.info("Analyzing dataset %s", args.dataset)
    logging.info("Loading dataset: %s", args.dataset)
    x_stream, y_stream, features_names, label_names = load_given_dataset(
        args.dataset
    )
    data_stream = DataStream(
        data=x_stream.todense(), y=y_stream.todense(), name=args.dataset
    )
    labels = y_stream.shape[1]
    cardinality = (
        sum(np.sum(y_stream.toarray(), axis=1)) / y_stream.toarray().shape[0]
    )
    density = cardinality / labels
    metadata["dataset"] = {
        "name": args.dataset,
        "instances": data_stream.n_remaining_samples(),
        "X_shape": x_stream.shape,
        "y_shape": y_stream.shape,
        "labels": labels,
        "cardinality": cardinality,
        "density": density,
        "label_names": [i[0] for i in label_names],
    }

    logging.info("Analyzing label relationship")
    priors, coocurrences, conditional_matrix = generate_labels_relationship(
        y_stream.toarray(),
        cardinalidad=cardinality,
    )
    save_labels_relationship(
        output_dir, args.dataset, priors, coocurrences, conditional_matrix
    )
    labels_relationship_graph(
        plot_props={"data": conditional_matrix},
        title="",
        # title="Relación entre Etiquetas\n{}".format(
        #    metadata["dataset"]["name"].capitalize()
        # ),
        output=os.path.join(
            output_dir,
            filename_path(
                "relationship_graph", args.dataset, output_dir, ext="png"
            ),
        ),
    )
    data_stream.restart()

    logging.info("Analyzing label skew")
    labels_skew_original = generate_labels_skew(y_stream.toarray())
    labels_skew_original.to_csv(
        os.path.join(output_dir, args.dataset + "_label_skew.csv")
    )
    lk_plot_data.append(
        {
            "x": np.arange(1, SKEW_TOP_COMBINATIONS + 1),
            "y": labels_skew_original.values[:SKEW_TOP_COMBINATIONS],
            "color": "black",
            "join": True,
            "label": "Original",
        }
    )
    lk_top_labels_combination["Original"] = top_labels_combinations(
        label_names, labels_skew_original
    )[:SKEW_TOP_COMBINATIONS]

    logging.info("Analyzing label distribution")
    (
        lbo_not_scaled,
        labels_distribution_original,
    ) = generate_labels_distribution(y_stream.toarray())
    lbo_not_scaled.to_csv(
        os.path.join(output_dir, args.dataset + "_label_distribution.csv")
    )
    ld_plot_data.append(
        {
            "x": labels_distribution_original.index.values,
            "y": labels_distribution_original.values,
            "color": "black",
            "join": True,
            "label": "Original",
        }
    )
    # Mean absolute error - graph
    ld_mae_plot_data.append(
        {
            "x": labels_distribution_original.index.values,
            "y": np.zeros(shape=len(labels_distribution_original)),
            "color": "black",
            "label": "Original",
            "join": True,
        }
    )

    logging.info("Analyzing label space")
    tf = top_features(
        x_stream.toarray(),
        y_stream.toarray(),
        np.array([i[0] for i in label_names]),
        np.array([i[0] for i in features_names]),
    )
    pd.DataFrame(zip(tf[0], tf[1]), columns=["atributo", "peso"]).to_csv(
        os.path.join(output_dir, args.dataset + "_features_space.csv")
    )

    # Limpia memoria
    del x_stream, y_stream, data_stream

    # FIN DATASET ANALYSIS

    # STREAM ANALYSIS

    if args.streams:
        stream_names = args.streamsnames or []
        if len(stream_names) != len(args.streams):
            logging.error(
                "La cantidad de streams y"
                + " la cantidad de nombres de streams no coinciden."
            )
            sys.exit(1)
        metadata["syn_streams"] = []
        for idx, i in enumerate(args.streams):
            stream_path = to_absolute(i)
            stream_name = stream_names[idx]

            logging.info("Analyzing syn stream: %s", stream_name)

            logging.info("Loading syn stream to memory")
            x_syn, y_syn, _, _ = load_moa_stream(stream_path, args.labels)

            labels = y_syn.shape[1]
            cardinality = (
                sum(np.sum(y_syn.toarray(), axis=1)) / y_syn.toarray().shape[0]
            )
            density = cardinality / labels

            logging.info("Analyzing label skew")
            labels_skew_syn = generate_labels_skew(y_syn.toarray())
            labels_skew_syn.to_csv(
                os.path.join(output_dir, stream_name + "_label_skew.csv")
            )
            lk_plot_data.append(
                {
                    "x": np.arange(1, SKEW_TOP_COMBINATIONS + 1),
                    "y": labels_skew_syn.values[:SKEW_TOP_COMBINATIONS],
                    "color": PLOT_COLORS[idx],
                    "join": True,
                    "label": stream_name,
                }
            )
            lk_top_labels_combination[stream_name] = top_labels_combinations(
                label_names, labels_skew_syn
            )[:SKEW_TOP_COMBINATIONS]

            logging.info("Analyzing label distribution")
            (
                lds_not_scaled,
                labels_distribution_syn,
            ) = generate_labels_distribution(y_syn.toarray())
            ld_syn = labels_distribution_syn.reindex(
                np.arange(
                    labels_distribution_original.index.min(),
                    labels_distribution_original.index.max() + 1,
                )
            ).fillna(0)
            ld_syn_not_scaled = lds_not_scaled.reindex(
                np.arange(
                    labels_distribution_original.index.min(),
                    labels_distribution_original.index.max() + 1,
                )
            ).fillna(0)
            ld_plot_data.append(
                {
                    "x": ld_syn.index.values,
                    "y": ld_syn.values,
                    "color": PLOT_COLORS[idx],
                    "join": True,
                    "label": stream_name,
                }
            )
            ld_syn_not_scaled.to_csv(
                os.path.join(
                    output_dir, stream_name + "_label_distribution.csv"
                )
            )
            mae = mean_absolute_error(
                labels_distribution_original.to_numpy(), ld_syn.to_numpy()
            )
            # plot mae
            ld_mae_plot_data.append(
                {
                    "x": labels_distribution_original.index.values,
                    "y": labels_distribution_original.to_numpy()
                    - ld_syn.to_numpy(),
                    "label": stream_name,
                    "color": PLOT_COLORS[idx],
                    "join": True,
                }
            )

            logging.info("Analyzing label relationship")
            (
                priors,
                coocurrences,
                conditional_matrix,
            ) = generate_labels_relationship(
                y_syn.toarray(),
                cardinalidad=cardinality,
            )
            save_labels_relationship(
                output_dir,
                stream_name,
                priors,
                coocurrences,
                conditional_matrix,
            )
            labels_relationship_graph(
                plot_props={"data": conditional_matrix},
                title="",
                # title="Relación entre Etiquetas\n{}".format(stream_name),
                output=os.path.join(
                    output_dir,
                    filename_path(
                        "relationship_graph",
                        stream_name,
                        output_dir,
                        ext="png",
                    ),
                ),
            )

            logging.info("Analyzing label space")
            tf = top_features(
                x_syn.toarray(),
                y_syn.toarray(),
                np.array([i[0] for i in label_names]),
                np.array([i[0] for i in features_names]),
            )
            pd.DataFrame(
                zip(tf[0], tf[1]), columns=["atributo", "peso"]
            ).to_csv(
                os.path.join(output_dir, stream_name + "_features_space.csv")
            )

            metadata["syn_streams"].append(
                {
                    "stream_path": stream_path,
                    "stream_name": stream_name,
                    "y_shape": y_syn.shape,
                    "labels": labels,
                    "cardinality": cardinality,
                    "density": density,
                    "labels_distribution_mean_absolute_error": mae,
                }
            )

    # FIN STREAM ANALYSIS

    logging.info("Plotting Label Skew")
    labels_skew_graph(
        lk_plot_data,
        title="Sesgo de Etiquetas\n{}".format(
            metadata["dataset"]["name"].capitalize()
        ),
        output=os.path.join(output_dir, "label_skew.png"),
    )
    pd.DataFrame.from_dict(lk_top_labels_combination).to_csv(
        os.path.join(output_dir, "label_skew_top_labels_combinations.csv")
    )

    logging.info("Plotting Label Distribution")
    labels_distribution_graph(
        ld_plot_data,
        title="Distribución de Etiquetas\n{}".format(
            metadata["dataset"]["name"].capitalize()
        ),
        output=os.path.join(output_dir, "label_distribution.png"),
    )
    labels_distribution_mae_graph(
        ld_mae_plot_data,
        title="Distribución de Etiquetas - Mean Absolute Error\n{}".format(
            metadata["dataset"]["name"].capitalize()
        ),
        output=os.path.join(output_dir, "ld_mae.png"),
    )

    logging.info("Saving metadata")
    with open(os.path.join(output_dir, "metadata.json"), "w") as fp:
        json.dump(metadata, fp, indent=4)
    logging.info("Files saved in %s", output_dir)


if __name__ == "__main__":
    main()
