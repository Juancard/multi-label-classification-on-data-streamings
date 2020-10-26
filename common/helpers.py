from skmultiflow.utils import check_random_state
from skmultilearn.dataset import load_dataset, load_from_arff
from sklearn.datasets import make_multilabel_classification
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from skmultiflow.meta import ClassifierChain
from skmultiflow.trees import LabelCombinationHoeffdingTreeClassifier
from skmultiflow.core.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import MultilabelGenerator
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches


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
    # por alguna razón la clase MultilabelGenerator no implementa el método has_more_classes

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


def evaluar(stream, model, pretrain_size=0.1):
    stream.restart()
    evaluator = EvaluatePrequential(
        show_plot=True,
        pretrain_size=round(stream.n_remaining_samples() * pretrain_size),
        max_samples=1000000,
        metrics=["exact_match", "hamming_score",
                 "hamming_loss", "running_time", "model_size"],
        output_file='results_br_stream_no_drift.csv'
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
    return labels_distribution_scaled


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
    plt.cla()
    plt.clf()
