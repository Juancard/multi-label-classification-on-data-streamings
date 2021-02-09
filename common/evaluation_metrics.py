import numpy as np
from scipy import sparse
from sklearn.metrics import (mean_absolute_error, accuracy_score,
                             jaccard_score, hamming_loss,
                             precision_recall_fscore_support, log_loss)
from skmultiflow.metrics import hamming_score, exact_match, j_index
from skmultilearn.utils import measure_per_label


def accuracy_eb(tp, set_true, set_pred):
    if len(set_true) == 0 and len(set_pred) == 0:
        return 1
    return tp / len(set_true.union(set_pred))


def precision_eb(tp, set_true, set_pred):
    if len(set_pred) == 0 and len(set_true) == 0:
        return 1
    if len(set_pred) == 0 and len(set_true) != 0:
        return 0
    return tp / len(set_pred)


def recall_eb(tp, set_true, set_pred):
    if len(set_true) == 0 and len(set_pred) == 0:
        return 1
    if len(set_true) == 0 and len(set_pred) != 0:
        return 0
    return tp / len(set_true)


def f1_eb(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return (2 * precision * recall) / (precision + recall)


def example_based_metrics(y_true, y_pred, normalize=None, sample_weight=None):
    acc_list = []
    prec_list = []
    rec_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tp = len(set_true.intersection(set_pred))
        acc_list.append(accuracy_eb(tp, set_true, set_pred))
        prec_list.append(precision_eb(tp, set_true, set_pred))
        rec_list.append(recall_eb(tp, set_true, set_pred))
    acc = np.mean(acc_list)
    prec = np.mean(prec_list)
    rec = np.mean(rec_list)
    f1 = f1_eb(prec, rec)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def example_based_metrics2(y_true, y_pred, normalize=None, sample_weight=None):
    acc = 0
    prec = 0
    rec = 0
    samples = y_true.shape[0]
    if samples == 0:
        return None
    for i in range(samples):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        tp = len(set_true.intersection(set_pred))
        acc += accuracy_eb(tp, set_true, set_pred) / samples
        prec += precision_eb(tp, set_true, set_pred) / samples
        rec += recall_eb(tp, set_true, set_pred) / samples
    f1 = f1_eb(prec, rec)
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


def evaluation_metrics(true_labels, predictions, start_time, end_time):
    evaluation = {}
    evaluation["accuracy_per_label"] = measure_per_label(
        accuracy_score,
        sparse.csr_matrix(true_labels),
        sparse.csr_matrix(predictions)
    )
    evaluation["hamming_loss"] = hamming_loss(true_labels, predictions)
    evaluation["hamming_score"] = hamming_score(true_labels, predictions)
    evaluation["exact_match"] = exact_match(true_labels,
                                            predictions)
    evaluation["accuracy_subset"] = accuracy_score(
        true_labels, predictions
    )
    evaluation["log_loss"] = log_loss(true_labels, predictions)
    evaluation["jaccard_index_eb_acc"] = j_index(true_labels, predictions)
    eb_metrics = example_based_metrics2(
        np.array(true_labels), np.array(predictions))
    for key, value in eb_metrics.items():
        evaluation["example_based_{}".format(key)] = value
    micro = precision_recall_fscore_support(
        true_labels, predictions, average="micro"
    )
    evaluation.update({
        "micro_precision": micro[0],
        "micro_recall": micro[1],
        "micro_fscore": micro[2],
        "micro_support": micro[3]
    })
    macro = precision_recall_fscore_support(
        true_labels, predictions, average="macro"
    )
    evaluation.update({
        "macro_precision": macro[0],
        "macro_recall": macro[1],
        "macro_fscore": macro[2],
        "macro_support": macro[3]
    })
    evaluation["time_seconds"] = end_time - start_time
    return evaluation
