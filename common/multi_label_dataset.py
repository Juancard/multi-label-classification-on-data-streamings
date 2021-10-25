import numpy as np
from toolz import pipe, curry
from skmultiflow.data.data_stream import DataStream
from scipy import sparse
from common.my_datasets import load_given_dataset


class MultiLabelDataset:
    def __init__(self, dataset_name, loader=load_given_dataset):
        self.name = dataset_name
        (self.x, self.y, self.feature_names, self.label_names) = loader(
            dataset_name
        )

    def number_of_instances(self):
        return self.x.shape[0]

    def number_of_labels(self):
        return self.y.shape[1]

    def label_cardinality(self):
        return np.sum(self.y, axis=1).sum() / self.number_of_instances()

    def to_data_stream(self):
        return DataStream(
            data=self.x.toarray(),
            y=self.y.toarray(),
            name=self.name,
        )

    def metadata(self):
        return {
            "name": self.name,
            "instances": self.number_of_instances(),
            "x_shape": self.x.shape,
            "y_shape": self.y.shape,
            "cardinality": self.label_cardinality(),
            "label_names": [i[0] for i in self.label_names],
        }

    @classmethod
    def withRepetitions(
        cls, dataset_name, copies=2, batches=1, loader=load_given_dataset
    ):

        name = "{}_{}_copies_per_instance".format(dataset_name, copies)
        the_loader_for_repetitions = curry(
            loader_with_repetitions, copies, batches, loader
        )
        instance = cls(dataset_name, the_loader_for_repetitions)
        instance.name = name
        return instance


def loader_with_repetitions(copies, batches, loader, dataset_name):
    def repeatInstances(X, y, copies=2, batches=1):
        x_repeat = np.vstack(
            np.array(
                [np.tile(i, (copies, 1)) for i in np.array_split(X, batches)]
            )
        )
        y_repeat = np.vstack(
            np.array(
                [np.tile(i, (copies, 1)) for i in np.array_split(y, batches)]
            )
        )
        return x_repeat, y_repeat

    def fill_repetitions(x, y, feature_names, labels_names):
        x, y = repeatInstances(x.toarray(), y.toarray(), copies, batches)
        return x, y, feature_names, labels_names

    def sparse_data(x, y, feature_names, labels_names):
        return (
            sparse.csr_matrix(x),
            sparse.csr_matrix(y),
            feature_names,
            labels_names,
        )

    return pipe(
        dataset_name,
        loader,
        lambda tuple_data: fill_repetitions(*tuple_data),
        lambda tuple_data: sparse_data(*tuple_data),
    )
