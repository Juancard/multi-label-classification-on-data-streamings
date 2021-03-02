import numpy as np
from scipy import sparse
from sklearn.metrics import (mean_absolute_error, accuracy_score,
                             jaccard_score, hamming_loss,
                             precision_recall_fscore_support, log_loss)
from skmultiflow.metrics import hamming_score, exact_match, j_index
from skmultilearn.utils import measure_per_label


def accuracy(vp, fp, vn, fn):
    return (vp + vn) / (vp + vn + fp + fn)


def precision(vp, fp, vn, fn):
    if vp + fp + fn == 0:
        return 0
    if vp + fp == 0:
        return 0
    return (vp) / (vp + fp)


def recall(vp, fp, vn, fn):
    if vp + fp + fn == 0:
        return 0
    if vp + fn == 0:
        return 0
    return (vp) / (vp + fn)


def fscore_1(precision, recall):
    if precision == 0 and recall == 0:
        return 0
    return 2.0 * ((precision * recall) / (precision + recall))


def confusion_matrix_values(set_true, set_false, set_pred, set_pred_false):
    set_vp = set_true.intersection(set_pred)
    set_fp = set_pred.difference(set_true)
    set_vn = set_pred_false.intersection(set_false)
    set_fn = set_pred_false.difference(set_false)
    return set_vp, set_fp, set_vn, set_fn


def label_based_metrics(true, pred):
    _, L = true.shape

    macro_acc = 0
    macro_prec = 0
    macro_rec = 0

    micro_vp = 0
    micro_fp = 0
    micro_vn = 0
    micro_fn = 0

    for j in range(L):
        set_true = set(np.where(true[:, j])[0])
        set_pred = set(np.where(pred[:, j])[0])
        set_false = set(np.where(true[:, j] != 1)[0])
        set_pred_false = set(np.where(pred[:, j] != 1)[0])

        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred,  set_pred_false)

        vp = len(set_vp)
        fp = len(set_fp)
        vn = len(set_vn)
        fn = len(set_fn)

        micro_vp += vp
        micro_fp += fp
        micro_vn += vn
        micro_fn += fn

        acc_parcial = accuracy(vp, fp, vn, fn)
        prec_parcial = precision(vp, fp, vn, fn)
        rec_parcial = recall(vp, fp, vn, fn)

        macro_acc += acc_parcial / L
        macro_prec += prec_parcial / L
        macro_rec += rec_parcial / L

    micro_acc = accuracy(micro_vp, micro_fp, micro_vn, micro_fn)
    micro_prec = precision(micro_vp, micro_fp, micro_vn, micro_fn)
    micro_rec = recall(micro_vp, micro_fp, micro_vn, micro_fn)

    return {
        "macro": {
            "accuracy": macro_acc,
            "precision": macro_prec,
            "recall": macro_rec,
            "f1": fscore_1(macro_prec, macro_rec)
        },
        "micro": {
            "accuracy": micro_acc,
            "precision": micro_prec,
            "recall": micro_rec,
            "f1": fscore_1(micro_prec, micro_rec)
        }
    }


def example_based_metrics(true, pred, normalize=None, sample_weight=None):
    acc = 0
    prec = 0
    rec = 0
    samples, _ = true.shape
    if samples == 0:
        return None
    for i in range(samples):
        set_true = set(np.where(true[i])[0])
        set_pred = set(np.where(pred[i])[0])
        set_false = set(np.where(true[i] != 1)[0])
        set_pred_false = set(np.where(pred[i] != 1)[0])

        set_vp, set_fp, set_vn, set_fn = confusion_matrix_values(
            set_true, set_false, set_pred,  set_pred_false)

        vp = len(set_vp)
        fp = len(set_fp)
        vn = len(set_vn)
        fn = len(set_fn)

        acc += accuracy(vp, fp, vn, fn) / samples
        prec += precision(vp, fp, vn, fn) / samples
        rec += recall(vp, fp, vn, fn) / samples
    f1 = fscore_1(prec, rec)
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
    eb_metrics = example_based_metrics(
        np.array(true_labels), np.array(predictions))
    for key, value in eb_metrics.items():
        evaluation["example_based_{}".format(key)] = value
    label_based = label_based_metrics(
        np.array(true_labels), np.array(predictions))
    evaluation.update({
        "micro_precision": label_based["micro"]["precision"],
        "micro_recall": label_based["micro"]["recall"],
        "micro_fscore": label_based["micro"]["f1"],
        "macro_precision": label_based["macro"]["precision"],
        "macro_recall": label_based["macro"]["recall"],
        "macro_fscore": label_based["macro"]["f1"],
    })
    evaluation["time_seconds"] = end_time - start_time
    return evaluation
