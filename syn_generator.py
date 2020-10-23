import sys
import os
import numpy as np
import argparse
from toolz import pipe, curry
import time
import csv
import json
import logging
from sklearn.metrics import mean_absolute_error
from skmultilearn.dataset import load_dataset, load_from_arff
from skmultiflow.data.data_stream import DataStream
from common.helpers import labels_relation_graph, label_skew_graph, labels_distribution_graph

parser = argparse.ArgumentParser(
    "Script to analyze a generated synthetic dataset")
parser.add_argument("-e", "--experiment", help="Description of the experiment")
parser.add_argument("-d", "--dataset", help="Name of skmultilearn dataset")
parser.add_argument("-s", "--streams", help="Path to stream", nargs='*')
parser.add_argument("-S", "--streamsnames", help="Names of streams", nargs='*')
parser.add_argument("-l", "--labels", type=int, help="Number of labels")
parser.add_argument("-o", "--output", help="Directory to save output.",
                    default="experiments/{}_syn".format(time.strftime("%Y%m%d%H%M%S")))


def setLogger():
    """
    To print logs
    """
    logging.basicConfig(level=logging.INFO)
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    return logging


def to_absolute_path(dir_path, path):
    if (os.path.isabs(path)):
        return path
    return os.path.join(dir_path, path)


def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def save_label_relationships(
        output_dir,
        dataset_name,
        priors,
        coocurrences,
        conditional_matrix,
        graph
):
    def filename_path(name, ext="csv"):
        filename = "{}_{}.{}".format(dataset_name, name, ext)
        return os.path.join(output_dir, filename)

    with open(filename_path("conditional"), 'w') as f:
        writer = csv.writer(f)
        for row in conditional_matrix:
            writer.writerow(row)

    with open(filename_path("priors"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(priors)

    with open(filename_path("coocurrences"), 'w') as f:
        writer = csv.writer(f)
        for row in coocurrences:
            writer.writerow(row)

    graph.get_figure().savefig(filename_path("relationship_graph", ext="png"))


def load_20ng_dataset():
    abs_path = os.path.dirname(os.path.realpath(__file__))
    arff_path = "../datasets/20ng/meka/20NG-F.arff"
    N_LABELS = 20
    label_location = "start"
    arff_file_is_sparse = False
    X_mulan, y_mulan, feature_names, label_names = load_from_arff(
        arff_path,
        N_LABELS,
        label_location=label_location,
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )
    return X_mulan, y_mulan, feature_names, label_names


def load_given_dataset(d):
    if (d.lower() == "20ng"):
        return load_20ng_dataset()
    return load_dataset(d, 'undivided')


def load_moa_stream(filepath, labels):
    print("Reading original arff from path")
    with open(filepath) as arff_file:
        arff_file_content = [line.rstrip(",\n") + "\n" for line in arff_file]
        with open("/tmp/stream", "w") as f:
            f.write("".join(arff_file_content))
    del arff_file_content
    print("Reading original arff from tmp")
    arff_path = "/tmp/stream"
    label_location = "start"
    arff_file_is_sparse = False
    return load_from_arff(
        arff_path,
        labels,
        label_location=label_location,
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True
    )


def main():
    logging = setLogger()
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
        "command": sys.argv,
        "date": time.strftime("%Y%m%d%H%M%S"),
    }
    label_skew_plot = None
    label_distribution_plot = None

    if not args.dataset:
        print("Dataset not provided. Exiting.")
        exit(0)

    #### DATASET ANALYSIS ######

    logging.info("Analyzing dataset {}".format(args.dataset))
    logging.info("Loading dataset: {}".format(args.dataset))
    X_stream, y_stream, feature_names, label_names = load_given_dataset(
        args.dataset)
    data_stream = DataStream(data=X_stream.todense(),
                             y=y_stream.todense(), name=args.dataset)
    cardinality = sum(np.sum(y_stream.toarray(), axis=1)
                      ) / y_stream.toarray().shape[0]
    metadata["dataset"] = {
        "name": args.dataset,
        "instances": data_stream.n_remaining_samples(),
        "X_shape": X_stream.shape,
        "y_shape": y_stream.shape,
        "cardinality": cardinality,
        "label_names": [i[0] for i in label_names]
    }

    logging.info("Analyzing label relationship")
    priors, coocurrences, conditional_matrix, graph = labels_relation_graph(
        y_stream.toarray(),
        cardinalidad=cardinality,
        print_coocurrence=False,
        annot=False
    )
    save_label_relationships(
        output_dir,
        args.dataset,
        priors,
        coocurrences,
        conditional_matrix,
        graph
    )
    data_stream.restart()

    logging.info("Analyzing label skew")
    label_skew_plot = label_skew_graph(
        y_stream.toarray(), color="black", plot_labels=500, label="Original")

    logging.info("Analyzing label distribution")
    labels_distribution_original, label_distribution_plot = labels_distribution_graph(
        y_stream.toarray(), color="black", label="Original")
    labels_distribution_original.to_csv(os.path.join(
        output_dir, args.dataset + "_label_distribution.csv"))

    # Limpia memoria
    del X_stream, y_stream, data_stream

    #### FIN DATASET ANALYSIS ######

    #### STREAM ANALYSIS ######

    if args.streams:
        stream_names = args.streamsnames or []
        if len(stream_names) != len(args.streams):
            logging.error(
                "La cantidad de streams y la cantidad de nombres de streams no coinciden.")
            exit(1)
        metadata["syn_streams"] = []
        for idx, i in enumerate(args.streams):
            stream_path = to_absolute(i)
            stream_name = stream_names[idx]

            logging.info("Analyzing syn stream: {}".format(stream_name))

            logging.info("Loading syn stream to memory")
            _, y_syn, _, _ = load_moa_stream(stream_path, args.labels)

            cardinality = sum(
                np.sum(y_syn.toarray(), axis=1)
            ) / y_syn.toarray().shape[0]

            logging.info("Analyzing label skew")
            label_skew_plot = label_skew_graph(
                y_syn.toarray(), plot_labels=500, label=stream_name)

            logging.info("Analyzing label distribution")
            labels_distribution_syn, label_distribution_plot = labels_distribution_graph(
                y_syn.toarray(), label=stream_name)
            labels_distribution_syn.to_csv(os.path.join(
                output_dir, stream_name + "_label_distribution.csv"))
            ld_syn_array = labels_distribution_syn.reindex(
                np.arange(
                    labels_distribution_original.index.min(),
                    labels_distribution_original.index.max() + 1
                )
            ).fillna(0).to_numpy()
            ld_array = labels_distribution_original.to_numpy()
            mae = mean_absolute_error(ld_array, ld_syn_array)

            logging.info("Analyzing label relationship")
            priors, coocurrences, conditional_matrix, graph = labels_relation_graph(
                y_syn.toarray(),
                cardinalidad=cardinality,
                print_coocurrence=False,
                annot=False
            )
            save_label_relationships(
                output_dir,
                stream_name,
                priors,
                coocurrences,
                conditional_matrix,
                graph
            )

            metadata["syn_streams"].append({
                "labels": 53,
                "stream_path": stream_path,
                "stream_name": stream_name,
                "y_shape": y_syn.shape,
                "cardinality": cardinality,
                "labels_distribution_mean_absolute_error": mae
            })

    #### FIN STREAM ANALYSIS ######

    label_skew_plot.get_figure().savefig(os.path.join(output_dir, "label_skew.png"))
    label_distribution_plot.get_figure().savefig(
        os.path.join(output_dir, "label_distribution.png"))

    logging.info("Saving metadata")
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as fp:
        json.dump(metadata, fp, indent=4)
    logging.info("Files saved in {}".format(output_dir))


if __name__ == "__main__":
    main()
