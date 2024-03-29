import time
import math
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import norm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

from sklearn.datasets import make_multilabel_classification

from skmultiflow.utils import check_random_state
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data import ConceptDriftStream
from skmultiflow.data import MultilabelGenerator
from skmultiflow.utils import calculate_object_size

GRAPH_TITLE_FONTSIZE = 18
GRAPH_AXIS_LABEL_FONTSIZE = 18
GRAPH_LEGEND_FONTSIZE = 12
GRAPH_TICKS_FONTSIZE = 18

class MultilabelGenerator2(MultilabelGenerator):
    # IGUAL QUE LA FUNCION ORIGINAL PERO NO PERMITE INSTANCIAS SIN ETIQUETAS
    def _prepare_for_use(self):
        print("Preparando Generador Multietiquetas v2")
        self._random_state = check_random_state(self.random_state)
        self.X, self.y = make_multilabel_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_classes=self.n_targets,
            n_labels=self.n_labels,
            allow_unlabeled=False,  # SE AGREGA ESTA LINEA
            random_state=self._random_state,
        )
        self.target_names = ["target_" + str(i) for i in range(self.n_targets)]
        self.feature_names = [
            "att_num_" + str(i) for i in range(self.n_num_features)
        ]
        self.target_values = (
            np.unique(self.y).tolist()
            if self.n_targets == 1
            else [
                np.unique(self.y[:, i]).tolist() for i in range(self.n_targets)
            ]
        )

    # por alguna razón la clase MultilabelGenerator no implementa el método
    # has_more_samples
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
            x = (
                -4.0
                * float(self.sample_idx - self.position)
                / float(self.width)
            )
            probability_drift = 1.0 / (1.0 + np.exp(x))
            if self._random_state.rand() > probability_drift:
                X, y = self.stream.next_sample()
            else:
                X, y = self.drift_stream.next_sample()
            self.current_sample_x[j, :] = X
            self.current_sample_y[j, :] = y
        return self.current_sample_x, self.current_sample_y


def evaluar(
    stream,
    model,
    pretrain_size=0.1,
    window_size=20,
    ensemble=False,
    logging=None,
    train_logs_max=5,
    catch_errors=False,
):
    stats = {
        "stream_name": stream.name,
        "instances": stream.n_remaining_samples(),
        "model_data": str(model),
        "ensemble": ensemble(model, stream) if ensemble else ensemble,
        "pretrain_size_prop": pretrain_size,
        "pretrain_size": round(stream.n_remaining_samples() * pretrain_size),
        "window_size": window_size,
    }
    stats["train_size"] = stream.n_remaining_samples() - stats["pretrain_size"]
    stats["batch_size"] = math.ceil(stats["train_size"] / window_size)
    stats["train_logs_max"] = train_logs_max
    log_every_iterations = math.ceil(stats["window_size"] / train_logs_max)

    logging.debug(stats)

    stats["start_time"] = time.time()

    true_labels = []
    predictions = []

    def train():
        # Pre training the classifier
        X, y = stream.next_sample(stats["pretrain_size"])
        do_pretraining = X.shape[0] > 0
        if ensemble:
            if isinstance(model, list):
                if do_pretraining:
                    logging.info("Pre-training models in ensemble...")
                    [
                        m.partial_fit(X, y, classes=stream.target_values[0])
                        for m in model
                    ]
                    model_pretrained = ensemble(model, stream)
                else:
                    model_pretrained = ensemble(model, stream)
            elif (
                type(ensemble(model, stream)).__name__
                == "OzaBaggingMLClassifier"
            ):
                model_pretrained = ensemble(model, stream)
                if do_pretraining:
                    logging.info("Pre-training oza...")
                    model_pretrained.partial_fit(
                        X, y, classes=stream.target_values[0]
                    )
            else:
                if do_pretraining:
                    logging.info("Pre-training model in ensemble...")
                    model.partial_fit(X, y, classes=stream.target_values[0])
                model_pretrained = ensemble(model, stream)
        else:
            if do_pretraining:
                logging.info("Pre-training model...")
                model.partial_fit(X, y, classes=stream.target_values[0])
            model_pretrained = model

        # Keeping track of sample count, true labels and predictions to later
        # compute the classifier's hamming score
        iterations = 0

        logging.info("Training...")
        while stream.has_more_samples():
            X, y = stream.next_sample(stats["batch_size"])
            y_pred = model_pretrained.predict(X)
            model_pretrained.partial_fit(X, y, classes=stream.target_values[0])
            predictions.extend(y_pred)
            true_labels.extend(y)
            if iterations % log_every_iterations == 0:
                logging.info(
                    "%s / %s trained samples.",
                    (iterations + 1) * stats["batch_size"],
                    stats["train_size"],
                )
            iterations += 1
        end_time = time.time()
        logging.info("All samples trained successfully")
        stats["success"] = True
        stats["error"] = False
        stats["end_time"] = end_time
        stats["time_seconds"] = end_time - stats["start_time"]
        stats["model_size_kb"] = calculate_object_size(model_pretrained, "kB")

    def onErrorCatched(error):
        end_time = time.time()
        logging.error(error)
        stats["success"] = False
        stats["error"] = error
        stats["end_time"] = end_time
        stats["time_seconds"] = end_time - stats["start_time"]
        stats["model_size_kb"] = None

    if not catch_errors:
        train()
    else:
        try:
            train()
        except Exception as error:
            onErrorCatched(error)

    return stats, true_labels, predictions


def evaluate_prequential(
    stream, model, pretrain_size=0.1, window_size=20, plot=False, output=None
):
    stream.restart()
    pretrain_samples = round(stream.n_remaining_samples() * pretrain_size)
    batch_size = round(
        (stream.n_remaining_samples() - pretrain_samples) / window_size
    )
    print("Pretrain size (examples):", pretrain_samples)
    print("Batch size (examples):", batch_size)
    evaluator = EvaluatePrequential(
        show_plot=plot,
        pretrain_size=pretrain_samples,
        batch_size=batch_size,
        max_samples=1000000,
        metrics=[
            "exact_match",
            "hamming_score",
            "hamming_loss",
            "j_index",
            "running_time",
            "model_size",
        ],
        output_file=output,
    )
    evaluator.evaluate(stream=stream, model=model)


def generate_labels_skew(y_array, print_top=False):
    dataframe = pd.DataFrame(y_array, columns=list(range(0, y_array.shape[1])))
    labels_set_count = (
        dataframe.groupby(dataframe.columns.tolist(), as_index=True)
        .size()
        .sort_values(ascending=False)
    )
    if print_top:
        print("Top ", print_top, ": \n", labels_set_count[:print_top], "\n")
    labels_set_count_scaled = (labels_set_count - labels_set_count.min()) / (
        labels_set_count.max() - labels_set_count.min()
    )
    return labels_set_count_scaled


def top_labels_combinations(labels_names, labels_skew):
    def active_label_names(x):
        return list(
            map(
                lambda idx: labels_names[idx][0], np.where(np.array(x) == 1)[0]
            )
        )

    return list(labels_skew.index.map(active_label_names))


def generate_labels_distribution(y_array, print_top=False):
    dataframe = pd.DataFrame(y_array, columns=list(range(0, y_array.shape[1])))
    df_count = dataframe.sum(axis=1).value_counts()
    labels_distribution = df_count.reindex(
        np.arange(df_count.index.min(), df_count.index.max() + 1)
    ).fillna(0)
    if print_top:
        print(
            "Número de etiquetas por instancia vs frecuencia - ",
            print_top,
            "\n",
            labels_distribution[:print_top],
            "\n",
        )
    labels_distribution_scaled = (labels_distribution - 0) / (
        labels_distribution.max() - 0
    )
    return labels_distribution, labels_distribution_scaled


def labels_distribution_graph(data, title="Label Distribution", output=False):
    fig = plt.figure(figsize=(16, 8))
    axis = fig.gca()
    axis.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE)
    axis.set_xlabel(
        "Combinaciones de Etiquetas",
        fontsize=GRAPH_AXIS_LABEL_FONTSIZE
    )
    axis.set_ylabel(
        "Frecuencia (escalada)",
        fontsize=GRAPH_AXIS_LABEL_FONTSIZE
    )
    axis.tick_params(axis='both', labelsize=GRAPH_TICKS_FONTSIZE)
    handles = []
    for i in data:
        sns.pointplot(**i, ax=axis)
        handles.append(
            mpatches.Patch(color=i.get("color"), label=i.get("label"))
        )
    plt.legend(handles=handles, prop={"size": GRAPH_LEGEND_FONTSIZE})
    if output:
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.cla()
    plt.clf()


def labels_skew_graph(data, title="", output=False):
    fig = plt.figure(figsize=(16, 8))
    axis = fig.gca()
    axis.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE)
    axis.set_xlabel(
        "Principales Combinaciones",
        fontsize=GRAPH_AXIS_LABEL_FONTSIZE
    )
    axis.set_ylabel(
        "Frecuencia (escalada)",
        fontsize=GRAPH_AXIS_LABEL_FONTSIZE
    )
    axis.tick_params(axis='both', labelsize=GRAPH_TICKS_FONTSIZE)

    max_x = max([max(i["x"]) for i in data])

    handles = []
    for i in data:
        sns.pointplot(**i, ax=axis)
        handles.append(
            mpatches.Patch(color=i.get("color"), label=i.get("label"))
        )
    plt.legend(handles=handles, prop={"size": GRAPH_LEGEND_FONTSIZE})
    axis.set_xticks(np.arange(1, max_x + 1, 2))
    if output:
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()
    plt.cla()
    plt.clf()


def labels_distribution_mae_graph(data, title="", output=False):
    fig = plt.figure(figsize=(16, 8))
    axis = fig.gca()
    axis.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE)
    axis.set_xlabel(
        "Combinaciones de Etiquetas",
        fontsize=GRAPH_AXIS_LABEL_FONTSIZE
    )
    axis.set_ylabel(
        "Distancia con respecto a dataset original",
        fontsize=GRAPH_AXIS_LABEL_FONTSIZE
    )
    axis.tick_params(axis='both', labelsize=GRAPH_TICKS_FONTSIZE)
    handles = []
    for i in data:
        sns.pointplot(**i, ax=axis)
        handles.append(
            mpatches.Patch(color=i.get("color"), label=i.get("label"))
        )
    plt.legend(handles=handles, prop={"size": GRAPH_LEGEND_FONTSIZE})
    if output:
        fig.savefig(output, bbox_inches="tight")
    else:
        plt.show()

    plt.cla()
    plt.clf()


def generate_labels_relationship(
    y_array, cardinalidad=False, print_coocurrence=False
):
    # Se calcula la probabilidad condicional P(A|B)
    # p(A|B) = P(A intersect B) / P(B)
    p_b = np.sum(y_array, axis=0)
    if cardinalidad:
        z_value = sum(p_b) / cardinalidad
        p_b = [min(1, i) for i in np.divide(p_b, z_value)]
    coocurrence = np.dot(y_array.T, y_array)
    # np.fill_diagonal(coocurrence,0)
    if print_coocurrence:
        np.set_printoptions(linewidth=120)
        print("Co-ocurrence matrix")
        print(coocurrence)
        print("\nPrior probabilities: ", p_b)
    p_a_intersection_b = coocurrence / y_array.shape[0]
    conditional_probs = np.divide(
        p_a_intersection_b,
        p_b,
        out=np.zeros(p_a_intersection_b.shape, dtype=float),
        where=p_b != 0,
    ).T
    return p_b, coocurrence, conditional_probs


def labels_relationship_graph(plot_props, title="", output=False):
    fig = plt.figure(figsize=(24, 16))
    axis = fig.gca()
    axis.set_title(title, fontsize=GRAPH_TITLE_FONTSIZE)
    axis.tick_params(axis='both', labelsize=GRAPH_TICKS_FONTSIZE + 16)
    sns.heatmap(
        linewidths=0,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        ax=axis,
        **plot_props
    )
    plt.yticks(rotation=0)
    axis.collections[0].colorbar.ax.tick_params(
            labelsize=GRAPH_TICKS_FONTSIZE + 14)
    if output:
        fig.savefig(output, bbox_inches="tight")
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
    top_freqs = features_sum[np.argsort(features_sum)[-top:]][::-1]
    top_features_name = features_names[top_idx]
    return top_features_name, top_freqs


def top_features_df(X, y, labels_names, features_names, labels=[], top=10):
    def tf_wrapper(labels):
        return top_features(
            X, y, labels_names, features_names, labels=labels, top=top
        )

    results = {}
    freqs = {}
    results["global"], freqs["global"] = tf_wrapper([])
    for label in labels:
        results[label], freqs[label] = tf_wrapper([label])
    labels_names_joined = ";".join(labels)
    results[labels_names_joined], freqs[labels_names_joined] = tf_wrapper(
        labels
    )
    return pd.DataFrame.from_dict(results), pd.DataFrame.from_dict(freqs)


def freqs_per_feature(
    X, y, feature_name, labels_names, features_names, labels=[], top=10
):
    feature_idx = np.where(features_names == feature_name)[0][0]
    idx_instances = np.array(range(0, X.shape[0]))
    for i in labels:
        label_idx = np.where(labels_names == i)[0][0]
        label_only = y[:, label_idx]
        label_only = label_only.toarray() if sparse.issparse(y) else label_only
        found = np.where(label_only > 0)[0]
        idx_instances = np.intersect1d(idx_instances, found)
    freqs_per_feature = (
        X[idx_instances, feature_idx].toarray()
        if sparse.issparse(X)
        else X[idx_instances, feature_idx]
    )
    return np.array(freqs_per_feature.T)[0]


def freqs_per_feature_dict(
    X, y, feature_name, labels_names, features_names, labels=[], top=10
):
    def fpf_wrapper(labels):
        return freqs_per_feature(
            X,
            y,
            feature_name,
            labels_names,
            features_names,
            labels=labels,
            top=top,
        )

    results = {}
    for label in labels:
        results[label] = fpf_wrapper([label])
    labels_names_joined = ";".join(labels)
    results[labels_names_joined] = fpf_wrapper(labels)
    return results
