import numpy as np
import random


def metodo_suma_pesos_legacy(example, labels):
    return [sum(exp["weight"][label] for exp in example)
            for label in range(labels)]


def metodo_suma_pesos_nuevo(example):
    return np.sum([exp["weight"] for exp in example], axis=0)


def metodo_prediccion_legacy(preds, sum_weights, instances, labels):
    aggregate = np.array([[
        np.sum(
            [np.array(p[i][label]) for p in preds] / sum_weights[label],
            axis=0
        ) for label in range(labels)
    ] for i in range(instances)])

    return (np.array(aggregate) + 0.5).astype(int)


def metodo_prediccion_nuevo(preds, sum_weights):
    return (np.sum(preds, axis=0) / sum_weights >= .5).astype(int)


def predict(example):
    instances, labels = example[0]["estimator"]["predict"]("").shape
    preds = [np.array(exp["estimator"]["predict"]("")) * exp["weight"]
             for exp in example]

    sum_weights = metodo_suma_pesos_legacy(example, labels)
    sum_weights2 = metodo_suma_pesos_nuevo(example)
    if not np.array_equal(sum_weights, sum_weights2):
        print("Métodos de suma de pesos NO son iguales!!")

    out = metodo_prediccion_legacy(preds, sum_weights, instances, labels)
    out2 = metodo_prediccion_nuevo(preds, sum_weights2)
    if not np.array_equal(out, out2):
        print("Métodos de calculo de predicción NO son iguales!!")

    # print("estimadores: ", len(example))
    # print("instancias: ", instances)
    # print("etiquetas: ", labels)

    # print("pesos: ", [e["weight"] for e in example])
    # print("predicciones: ", [e["estimator"]["predict"]("") for e in example])

    # print("predicciones ponderadas: ", preds)
    # print("suma de pesos por etiqueta: ", sum_weights)
    # print("aggregate: ", aggregate)

    # print("prediccion final: ", out)
    # print("predicción final 2:", out2)
    # print("*" * 100)

    return out2, out


example_simplest = [{
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 1]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[1, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[1, 1]])
    }
}]

example_less_simple = [{
    "weight": np.array([0.1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[1, 0]])
    }
}, {
    "weight": np.array([0.1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[1, 1]])
    }
}, {
    "weight": np.array([0.1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 1]])
    }
}]

example_with_zero_in_pred = [{
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 1]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 1]])
    }
}]

example_many_instances = [{
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0], [1, 1]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 1], [1, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[1, 0], [0, 1]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[1, 1], [0, 0]])
    }
}]

example_many_instances2 = [{
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0], [1, 1]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 1], [1, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[0, 0], [0, 0]])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([[1, 1], [1, 0]])
    }
}]


def example_random():
    instances = random.randint(1, 10)
    labels = random.randint(1, 10)
    classifiers = random.randint(1, 4)
    return [{
        "weight": np.random.rand(labels),
        "estimator": {
            "predict": lambda _: np.random.rand(instances, labels)
        }
    } for i in range(classifiers)]


examples = [
    example_simplest,
    example_less_simple,
    example_with_zero_in_pred,
    example_many_instances,
    example_many_instances2,
]

[predict(example) for example in examples]

for i in range(1000):
    er = example_random()
    a, b = predict(er)
    if not np.array_equal(a, b):
        print("Distintos!")
        print(a, b)
        print(er)
        break
