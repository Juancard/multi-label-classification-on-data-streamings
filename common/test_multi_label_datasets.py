from unittest import TestCase
import numpy as np
from skmultiflow.data.data_stream import DataStream

from common.multi_label_dataset import MultiLabelDataset

TEST_DATASET = "test"


class TestMultiLabelDataset(TestCase):

    NAME = TEST_DATASET
    INSTANCES = 11
    ATTRIBUTES = 3
    LABELS = 5
    CARDINALITY = 1.90909091

    def setUp(self) -> None:
        self.multi_label_dataset = MultiLabelDataset(self.NAME)

    def test_name(self):
        self.assertEqual(self.multi_label_dataset.name, self.NAME)

    def test_x_shape(self):
        self.assertEqual(
            self.multi_label_dataset.x.shape, (self.INSTANCES, self.ATTRIBUTES)
        )

    def test_y_shape(self):
        self.assertEqual(
            self.multi_label_dataset.y.shape, (self.INSTANCES, self.LABELS)
        )

    def test_number_of_instances(self):
        self.assertEqual(
            self.multi_label_dataset.number_of_instances(), self.INSTANCES
        )

    def test_number_of_labels(self):
        self.assertEqual(
            self.multi_label_dataset.number_of_labels(), self.LABELS
        )

    def test_cardinality(self):
        self.assertAlmostEqual(
            self.multi_label_dataset.label_cardinality(), self.CARDINALITY
        )

    def test_converts_to_stream(self):
        data_stream = self.multi_label_dataset.to_data_stream()
        self.assertIsInstance(data_stream, DataStream)

    def test_stream_is_same_shape(self):
        data_stream = self.multi_label_dataset.to_data_stream()
        self.assertEqual(
            self.multi_label_dataset.number_of_instances(),
            data_stream.n_remaining_samples(),
        )
        self.assertEqual(
            self.multi_label_dataset.number_of_labels(), data_stream.n_targets
        )

    def test_stream_is_same_x(self):
        data_stream = self.multi_label_dataset.to_data_stream()
        x, _ = data_stream.next_sample(
            self.multi_label_dataset.number_of_instances()
        )
        self.assertTrue(np.allclose(self.multi_label_dataset.x.toarray(), x))

    def test_stream_is_same_y(self):
        data_stream = self.multi_label_dataset.to_data_stream()
        _, y = data_stream.next_sample(
            self.multi_label_dataset.number_of_instances()
        )
        self.assertTrue(np.allclose(self.multi_label_dataset.y.toarray(), y))


class TestMultiLabelDatasetWithRepetitions(TestMultiLabelDataset):
    COPIES = 2
    BATCHES = 1
    INSTANCES = 11 * 2
    NAME = "{}_{}_copies_per_instance".format(TEST_DATASET, COPIES)

    def setUp(self) -> None:
        self.multi_label_dataset = MultiLabelDataset.withRepetitions(
            TEST_DATASET, self.COPIES, self.BATCHES
        )
