from unittest import TestCase
import numpy as np

from common.evaluation_metrics import accuracy, precision, recall, fscore_1, confusion_matrix_values


class TestEvaluationMetrics(TestCase):

    def setUp(self) -> None:
        pass

    def test_accuracy_1(self):
        vp = 10
        fp = 0
        vn = 10
        fn = 0
        perfect_acc = 1.0
        self.assertEqual(perfect_acc,
                         accuracy(vp, fp, vn, fn))

    def test_accuracy_0(self):
        vp = 0
        fp = 10
        vn = 0
        fn = 10
        acc_0 = 0.0
        self.assertEqual(acc_0,
                         accuracy(vp, fp, vn, fn))

    def test_accuracy_0_5(self):
        vp = 10
        fp = 10
        vn = 10
        fn = 10
        acc_0_5 = 0.5
        self.assertEqual(acc_0_5,
                         accuracy(vp, fp, vn, fn))

    def test_accuracy_no_samples(self):
        vp = 0
        fp = 0
        vn = 0
        fn = 0
        self.assertRaises(ZeroDivisionError,
                          lambda: accuracy(vp, fp, vn, fn))

    def test_precision_1_when_no_fp(self):
        vp = 10
        fp = 0
        vn = 10
        fn = 10
        self.assertEqual(1.0, precision(vp, fp, vn, fn))

    def test_precision_not_1_when_fp(self):
        vp = 10
        fp = 1
        vn = 10
        fn = 10
        self.assertNotEqual(1.0, precision(vp, fp, vn, fn))

    def test_precision_0_5(self):
        vp = 10
        fp = 10
        vn = 0
        fn = 0
        self.assertEqual(0.5, precision(vp, fp, vn, fn))

    def test_precision_0_when_no_vp(self):
        vp = 0
        fp = 10
        vn = 0
        fn = 0
        self.assertEqual(0, precision(vp, fp, vn, fn))

    def test_precision_0_when_no_vp_and_no_fp(self):
        vp = 0
        fp = 0
        vn = 0
        fn = 0
        self.assertEqual(0, precision(vp, fp, vn, fn))

    def test_recall_1_when_no_fn(self):
        vp = 10
        fp = 0
        vn = 10
        fn = 0
        self.assertEqual(1.0, recall(vp, fp, vn, fn))

    def test_recall_not_1_when_fn(self):
        vp = 10
        fp = 1
        vn = 10
        fn = 10
        self.assertNotEqual(1.0, recall(vp, fp, vn, fn))

    def test_recall_0_5(self):
        vp = 10
        fp = 0
        vn = 0
        fn = 10
        self.assertEqual(0.5, recall(vp, fp, vn, fn))

    def test_recall_0_when_no_vp(self):
        vp = 0
        fp = 10
        vn = 0
        fn = 10
        self.assertEqual(0, recall(vp, fp, vn, fn))

    def test_recall_0_when_no_vp_and_no_fn(self):
        vp = 0
        fp = 0
        vn = 0
        fn = 0
        self.assertEqual(0, recall(vp, fp, vn, fn))

    def test_fscore_0_when_precision_and_recall_0(self):
        precision = 0
        recall = 0
        self.assertEqual(0, fscore_1(precision, recall))

    def test_fscore_1_when_precision_and_recall_1(self):
        precision = 1
        recall = 1
        self.assertEqual(1, fscore_1(precision, recall))

    def test_fscore_0_when_precision_1_and_recall_0(self):
        precision = 1
        recall = 0
        self.assertEqual(0, fscore_1(precision, recall))

    def test_fscore_0_when_precision_0_and_recall_1(self):
        precision = 0
        recall = 1
        self.assertEqual(0, fscore_1(precision, recall))

    def test_fscore_0_5_when_precision_0_5_and_recall_0_5(self):
        precision = 0.5
        recall = 0.5
        self.assertEqual(0.5, fscore_1(precision, recall))

    def test_confusion_matrix_example_1(self):
        # pred = np.array([
        # [0, 1, 1, 0],
        # [1, 1, 1, 0],
        # [1, 0, 1, 0],
        # [1, 0, 0, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0]
        # ])
        # true = np.array([
        # [1, 1, 1, 0],
        # [1, 1, 1, 0],
        # [1, 1, 1, 0],
        # [0, 0, 0, 0],
        # [0, 0, 0, 0],
        # [1, 0, 0, 0]
        # ])
        # set_true = set(np.where(true[:, 0])[0])
        # set_pred = set(np.where(pred[:, 0])[0])
        # set_false = set(np.where(true[:, 0] != 1)[0])
        # set_pred_false = set(np.where(pred[:, 0] != 1)[0])
        set_true = {0, 1, 2, 5}
        set_pred = {1, 2, 3}
        set_false = {3, 4}
        set_pred_false = {0, 4, 5}
        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred, set_pred_false)
        self.assertEqual({1, 2}, set_vp)
        self.assertEqual({3}, set_fp)
        self.assertEqual({4}, set_vn)
        self.assertEqual({0, 5}, set_fn)

    def test_confusion_matrix_example_2(self):
        set_true = {0, 1, 2}
        set_pred = {0, 1}
        set_false = {3, 4, 5}
        set_pred_false = {2, 3, 4, 5}
        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred, set_pred_false)
        self.assertEqual({0, 1}, set_vp)
        self.assertEqual(set(), set_fp)
        self.assertEqual({3, 4, 5}, set_vn)
        self.assertEqual({2}, set_fn)

    def test_confusion_matrix_example_3(self):
        set_true = {0, 1, 2}
        set_pred = {0, 1, 2}
        set_false = {3, 4, 5}
        set_pred_false = {3, 4, 5}
        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred, set_pred_false)
        self.assertEqual({0, 1, 2}, set_vp)
        self.assertEqual(set(), set_fp)
        self.assertEqual({3, 4, 5}, set_vn)
        self.assertEqual(set(), set_fn)

    def test_confusion_matrix_example_4(self):
        set_true = set()
        set_pred = set()
        set_false = {0, 1, 2, 3, 4, 5}
        set_pred_false = {0, 1, 2, 3, 4, 5}
        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred, set_pred_false)
        self.assertEqual(set(), set_vp)
        self.assertEqual(set(), set_fp)
        self.assertEqual({0, 1, 2, 3, 4, 5}, set_vn)
        self.assertEqual(set(), set_fn)

    def test_confusion_matrix_example_5(self):
        set_true = {0, 1, 2, 3, 4, 5}
        set_pred = {0, 1, 2, 3, 4, 5}
        set_false = set()
        set_pred_false = set()
        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred, set_pred_false)
        self.assertEqual({0, 1, 2, 3, 4, 5}, set_vp)
        self.assertEqual(set(), set_fp)
        self.assertEqual(set(), set_vn)
        self.assertEqual(set(), set_fn)

    def test_confusion_matrix_example_6(self):
        set_true = {0}
        set_pred = set()
        set_false = set()
        set_pred_false = {0}
        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred, set_pred_false)
        self.assertEqual(set(), set_vp)
        self.assertEqual(set(), set_fp)
        self.assertEqual(set(), set_vn)
        self.assertEqual({0}, set_fn)
