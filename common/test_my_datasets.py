from unittest import TestCase

from common.my_datasets import (
    local_datasets,
    load_given_dataset,
    available_datasets
)


class TestMyDatasets(TestCase):

    def setUp(self) -> None:
        pass

    def assert_dataset_shape(self, name, instances, attributes, labels):
        x, y, _, _ = load_given_dataset(name)
        x_instances, x_attributes = x.shape
        y_instances, y_labels = y.shape
        self.assertEqual(x_instances, instances)
        self.assertEqual(y_instances, instances)
        self.assertEqual(x_attributes, attributes)
        self.assertEqual(y_labels, labels)

    def test_local_datasets_are_the_right_ones(self):
        self.assertEqual(local_datasets(), ["20ng", "test"])

    def test_datasets_include_the_right_ones(self):
        the_right_ones = set([
            "test",
            "20ng",
            "enron",
            "mediamill"
        ])
        self.assertTrue(
            available_datasets().issuperset(the_right_ones)
        )

    def test_loads_test_dataset(self):
        dataset = "test"
        instances = 11
        attributes = 3
        labels = 5
        self.assert_dataset_shape(dataset, instances, attributes, labels)

    def test_loads_20ng(self):
        dataset = "20ng"
        instances = 19300
        attributes = 1006
        labels = 20
        self.assert_dataset_shape(dataset, instances, attributes, labels)

    def test_loads_enron(self):
        dataset = "enron"
        instances = 1702
        attributes = 1001
        labels = 53
        self.assert_dataset_shape(dataset, instances, attributes, labels)

    def test_loads_mediamill(self):
        dataset = "mediamill"
        instances = 43907
        attributes = 120
        labels = 101
        self.assert_dataset_shape(dataset, instances, attributes, labels)
