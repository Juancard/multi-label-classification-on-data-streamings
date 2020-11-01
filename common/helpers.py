import time
import os
import sys
import math
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from skmultilearn.dataset import load_dataset, load_from_arff
from sklearn.datasets import make_multilabel_classification

from skmultiflow.utils import check_random_state
from skmultiflow.meta import ClassifierChain
from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier
from skmultiflow.core.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import MultilabelGenerator

from sklearn.metrics import (mean_absolute_error, accuracy_score,
                             jaccard_score, hamming_loss, precision_recall_fscore_support, log_loss)
from skmultiflow.metrics import hamming_score, exact_match, j_index
from skmultilearn.utils import measure_per_label


def load_20ng_dataset():
    abs_path = os.path.dirname(os.path.realpath(__file__))
    arff_path = "./datasets/20NG-F.arff"
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


class MultilabelGenerator2(MultilabelGenerator):
    # IGUAL QUE LA FUNCION ORIGINAL PERO NO PERMITE INSTANCIAS SIN ETIQUETAS
    def _prepare_for_use(self):
        print("Preparando Generador Multietiquetas v2")
        self._random_state = check_random_state(self.random_state)
        self.X, self.y = make_multilabel_classification(n_samples=self.n_samples,
                                                        n_features=self.n_features,
                                                        n_classes=self.n_targets,
                                                        n_labels=self.n_labels,
                                                        allow_unlabeled=False,  # SE AGREGA ESTA LINEA
                                                        random_state=self._random_state)
        self.target_names = ["target_" + str(i) for i in range(self.n_targets)]
        self.feature_names = ["att_num_" +
                              str(i) for i in range(self.n_num_features)]
        self.target_values = np.unique(self.y).tolist() if self.n_targets == 1 else \
            [np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)]

    # por alguna razón la clase MultilabelGenerator no implementa el método has_more_samples
    def has_more_samples(self):
        return self.n_remaining_samples() > 0


class ConceptDriftStream2(ConceptDriftStream):
    def prob_drift(self, batch_size=1):
        x = -4.0 * float(self.sample_idx - self.position) / float(self.width)
        probability_drift = 1.0 / (1.0 + np.exp(x))
        x, y = super().next_sample(batch_size)
        return probability_drift

    def next_sample(self, batch_size=1):
        """
        Copio y pego textual de next_sample. 
        Solo quito el planchado realizado sobre la matriz de etiquetas.
        "self.current_sample_y.flatten()" pasa a ser "self.current_sample_y"
        """
        self.current_sample_x = np.zeros((batch_size, self.n_features))
        self.current_sample_y = np.zeros((batch_size, self.n_targets))
        for j in range(batch_size):
            self.sample_idx += 1
            x = -4.0 * float(self.sample_idx - self.position) / \
                float(self.width)
            probability_drift = 1.0 / (1.0 + np.exp(x))
            if self._random_state.rand() > probability_drift:
                X, y = self.stream.next_sample()
            else:
                X, y = self.drift_stream.next_sample()
            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y
        return self.current_sample_x, self.current_sample_y


def label_based_accuracy(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the label-based accuracy for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set(np.where(y_true[i])[0])
        set_pred = set(np.where(y_pred[i])[0])
        #print('\nset_true: {0}'.format(set_true))
        #print('set_pred: {0}'.format(set_pred))
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred)) /\
                float(len(set_true.union(set_pred)))
        #print('tmp_a: {0}'.format(tmp_a))
        acc_list.append(tmp_a)
    return np.mean(acc_list)


def evaluation_metrics(true_labels, predictions, start_time, end_time):
    evaluation = {}
    evaluation["hamming_loss"] = hamming_loss(true_labels, predictions)
    evaluation["exact_match (aka, 0/1-loss)"] = exact_match(true_labels,
                                                            predictions)
    evaluation["accuracy (exampled-based)"] = accuracy_score(true_labels, predictions)
    evaluation["accuracy (label-based): "] = label_based_accuracy(
        np.array(true_labels), np.array(predictions))
    evaluation["accuracy_per_label"] = measure_per_label(
        accuracy_score,
        sparse.csr_matrix(true_labels),
        sparse.csr_matrix(predictions)
    )
    evaluation["jaccard_index"] = j_index(true_labels, predictions)
    evaluation["log_loss"] = log_loss(true_labels, predictions)
    evaluation["precision_recall_fscore_support_samples"] = precision_recall_fscore_support(
        true_labels, predictions, average="samples"
    )
    evaluation["precision_recall_fscore_support_weighted"] = precision_recall_fscore_support(
        true_labels, predictions, average="weighted"
    )
    evaluation["precision_recall_fscore_support_micro"] = precision_recall_fscore_support(
        true_labels, predictions, average="micro"
    )
    evaluation["precision_recall_fscore_support_macro"] = precision_recall_fscore_support(
        true_labels, predictions, average="macro"
    )
    evaluation["time_seconds"] = end_time - start_time
    return evaluation


def evaluar(stream, model, pretrain_size=0.1, window_size=20, logging=None, train_logs_max=5):
    stream.restart()
    stats = {
        "stream_name": stream.name,
        "instances": stream.n_remaining_samples(),
        "model_data": str(model),
        "pretrain_size_prop": pretrain_size,
        "pretrain_size": round(stream.n_remaining_samples() * pretrain_size),
        "window_size": window_size
    }
    stats["train_size"] = stream.n_remaining_samples() - stats["pretrain_size"]
    stats["batch_size"] = math.ceil(stats["train_size"] / window_size)
    stats["train_logs_max"] = train_logs_max
    log_every_iterations = math.ceil(stats["window_size"] / train_logs_max)

    logging.info(stats)

    logging.info("Pretraining...")
    stats["start_time"] = time.time()

    try:
        # Pre training the classifier
        X, y = stream.next_sample(stats["pretrain_size"])
        model.partial_fit(X, y, classes=stream.target_values)

        # Keeping track of sample count, true labels and predictions to later
        # compute the classifier's hamming score
        iterations = 0
        true_labels = []
        predictions = []
        end_time = None

        logging.info("Training...")
        while stream.has_more_samples():
            X, y = stream.next_sample(stats["batch_size"])
            y_pred = model.predict(X)
            model.partial_fit(X, y)
            predictions.extend(y_pred)
            true_labels.extend(y)
            if iterations % log_every_iterations == 0:
                logging.info(
                    "%s / %s trained samples.",
                    (iterations + 1) * stats["batch_size"],
                    stats["train_size"]
                )
            iterations += 1
        end_time = time.time()
        logging.info("All samples trained successfully")
        stats["success"] = True
        stats["error"] = False
    except Exception as e:
        end_time = time.time()
        logging.error(e)
        stats["success"] = False
        stats["error"] = e
        true_labels = None
        predictions = None
    finally:
        stats["end_time"] = end_time
        stats["time_seconds"] = end_time - start_time
        return stats, true_labels, predictions


def evaluate_prequential(stream, model, pretrain_size=0.1, window_size=20, plot=False, output=None):
    stream.restart()
    pretrain_samples = round(stream.n_remaining_samples() * pretrain_size)
    batch_size = round((stream.n_remaining_samples() -
                        pretrain_samples) / window_size)
    print("Pretrain size (examples):", pretrain_samples)
    print("Batch size (examples):", batch_size)
    evaluator = EvaluatePrequential(
        show_plot=plot,
        pretrain_size=pretrain_samples,
        batch_size=batch_size,
        max_samples=1000000,
        metrics=["exact_match", "hamming_score", "hamming_loss",
                 "j_index", "running_time", "model_size"],
        output_file=output
    )
    evaluator.evaluate(stream=stream, model=model)


def generate_labels_skew(y_array, print_top=False):
    df = pd.DataFrame(y_array, columns=[i for i in range(0, y_array.shape[1])])
    labels_set_count = df.groupby(
        df.columns.tolist(), as_index=True).size().sort_values(ascending=False)
    if (print_top):
        print("Top ", print_top, ": \n",
              labels_set_count[:print_top], "\n")
    labels_set_count_scaled = (labels_set_count-labels_set_count.min()) / \
        (labels_set_count.max()-labels_set_count.min())
    return labels_set_count_scaled


def generate_labels_distribution(y_array, print_top=False):
    df = pd.DataFrame(y_array, columns=[i for i in range(0, y_array.shape[1])])
    df_count = df.sum(axis=1).value_counts()
    labels_distribution = df_count.reindex(
        np.arange(df_count.index.min(), df_count.index.max() + 1)).fillna(0)
    if (print_top):
        print("Número de etiquetas por instancia vs frecuencia - ",
              print_top, "\n", labels_distribution[:print_top], "\n")
    labels_distribution_scaled = (labels_distribution-0)/(
        labels_distribution.max()-0)
    #print("Número de etiquetas por instancia vs frecuencia (escalada)\n", labels_distribution_scaled)
    return labels_distribution, labels_distribution_scaled


def labels_distribution_graph(data, title="Label Distribution", output=False):
    f1 = plt.figure(figsize=(16, 8))
    a1 = f1.gca()
    a1.set_title(title)
    a1.set_xlabel('Labels Combinations')
    a1.set_ylabel('Frequency (Scaled)')
    handles = []
    for i in data:
        sns.pointplot(**i, ax=a1)
        handles.append(
            mpatches.Patch(
                color=i.get("color"),
                label=i.get("label")
            )
        )
    plt.legend(handles=handles)
    if (output):
        f1.savefig(output)
    else:
        plt.show()
    plt.cla()
    plt.clf()


def labels_skew_graph(data, title="", output=False):
    f1 = plt.figure(figsize=(16, 8))
    a1 = f1.gca()
    a1.set_title(title)
    a1.set_xlabel('Top Combinations')
    a1.set_ylabel('Frequency (Scaled)')
    handles = []
    for i in data:
        sns.pointplot(**i, ax=a1)
        handles.append(
            mpatches.Patch(
                color=i.get("color"),
                label=i.get("label")
            )
        )
    plt.legend(handles=handles)
    if (output):
        f1.savefig(output)
    else:
        plt.show()
    plt.cla()
    plt.clf()


def labels_distribution_mae_graph(data, title="", output=False):
    f1 = plt.figure(figsize=(16, 8))
    a1 = f1.gca()
    a1.set_title(title)
    a1.set_xlabel('Labels Combinations')
    a1.set_ylabel('Distance from Original Dataset')
    handles = []
    for i in data:
        sns.pointplot(**i, ax=a1)
        handles.append(
            mpatches.Patch(
                color=i.get("color"),
                label=i.get("label")
            )
        )
    plt.legend(handles=handles)
    if (output):
        f1.savefig(output)
    else:
        plt.show()

    plt.cla()
    plt.clf()


def generate_labels_relationship(y_array, cardinalidad=False, print_coocurrence=False):
    # Se calcula la probabilidad condicional P(A|B)
    # p(A|B) = P(A intersect B) / P(B)
    p_b = np.sum(y_array, axis=0)
    if (cardinalidad):
        z = sum(p_b) / cardinalidad
        p_b = [min(1, i) for i in np.divide(p_b, z)]
    coocurrence = np.dot(y_array.T, y_array)
    # np.fill_diagonal(coocurrence,0)
    if (print_coocurrence):
        np.set_printoptions(linewidth=120)
        print("Co-ocurrence matrix")
        print(coocurrence)
        print("\nPrior probabilities: ", p_b)
    p_a_intersection_b = coocurrence / y_array.shape[0]
    conditional_probs = np.divide(
        p_a_intersection_b,
        p_b,
        out=np.zeros(p_a_intersection_b.shape, dtype=float),
        where=p_b != 0
    ).T
    return p_b, coocurrence, conditional_probs


def labels_relationship_graph(plot_props, title="", output=False):
    f1 = plt.figure(figsize=(24, 16))
    a1 = f1.gca()
    a1.set_title(title)
    sns.heatmap(
        linewidths=0,
        cmap=sns.color_palette("Greys_r", n_colors=100),
        ax=a1,
        **plot_props
    )
    if (output):
        f1.savefig(output)
    else:
        plt.show()
    plt.cla()
    plt.clf()


def top_features(X, y, labels_names, features_names, labels=[], top=10):
    idx_instances = np.array(range(0, X.shape[0]))
    for i in labels:
        label_idx = np.where(labels_names == i)[0][0]
        label_only = y[:, label_idx]
        label_only = label_only.toarray() if sparse.issparse(y) else label_only
        found = np.where(label_only > 0)[0]
        idx_instances = np.intersect1d(idx_instances, found)
    features_sum = np.asarray(X[idx_instances].sum(axis=0)).flatten()
    top_idx = features_sum.argsort()[-top:][::-1]
    return features_names[top_idx]


def top_features_df(X, y, labels_names, features_names, labels=[], top=10):
    def tf(labels): return top_features(
        X, y, labels_names, features_names, labels=labels, top=top)
    results = {}
    results["global"] = tf([])
    for l in labels:
        results[l] = tf([l])
    results[";".join(labels)] = tf(labels)
    return pd.DataFrame.from_dict(results)


def repeatInstances(X, y, copies=2, batches=1):
    X_repeat = np.vstack(
        np.array([
            np.tile(i, (copies, 1))
            for i in np.array_split(X, batches)
        ])
    )
    y_repeat = np.vstack(
        np.array([
            np.tile(i, (copies, 1))
            for i in np.array_split(y, batches)
        ])
    )
    return X_repeat, y_repeat
