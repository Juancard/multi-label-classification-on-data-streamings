import uuid
from skmultilearn.dataset import (
    load_from_arff,
    load_dataset,
    available_data_sets,
)


def load_from_arff_wrapper(
    arff_path, n_labels, label_location, file_is_sparse=False
):
    return load_from_arff(
        arff_path,
        n_labels,
        label_location=label_location,
        load_sparse=file_is_sparse,
        return_attribute_definitions=True,
    )


def load_custom_dataset(dataset_name, path=None):
    if dataset_name == "20ng":
        arff_path = path if path else "./datasets/20NG-F.arff"
        n_labels = 20
        label_location = "start"
        return load_from_arff_wrapper(arff_path, n_labels, label_location)
    if dataset_name == "test":
        arff_path = path if path else "./datasets/test.arff"
        n_labels = 5
        label_location = "end"
        return load_from_arff_wrapper(arff_path, n_labels, label_location)


def load_moa_stream(filepath, labels):
    print("Reading original arff from path")
    with open(filepath) as arff_file:
        arff_file_content = [line.rstrip(",\n") + "\n" for line in arff_file]
    filename = "stream_{}".format(str(uuid.uuid1()))
    tmp_file = "/tmp/{}".format(filename)
    with open(tmp_file, "w") as opened:
        opened.write("".join(arff_file_content))
    del arff_file_content
    print("Reading original arff from tmp")
    arff_path = tmp_file
    label_location = "start"
    arff_file_is_sparse = False
    return load_from_arff(
        arff_path,
        labels,
        label_location=label_location,
        load_sparse=arff_file_is_sparse,
        return_attribute_definitions=True,
    )


def local_datasets():
    return ["20ng", "test"]


def load_given_dataset(dataset):
    if dataset.lower() in local_datasets():
        return load_custom_dataset(dataset.lower())
    return load_dataset(dataset, "undivided")


def available_datasets():
    available_datasets = {x[0].lower() for x in available_data_sets().keys()}
    for i in local_datasets():
        available_datasets.add(i)
    return available_datasets
