import numpy as np
experts = [{
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([0, 0])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([0, 1])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([1, 0])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([1, 1])
    }
}]
labels = 2
preds = [np.array(exp["estimator"]["predict"]("")) * exp["weight"]
         for exp in experts]
sum_weights = [sum(exp["weight"][label] for exp in experts)
               for label in range(labels)]
aggregate = [
    np.sum(
        [np.array(p[label]) for p in preds] / sum_weights[label],
        axis=0
    ) for label in range(labels)
]

out = (np.array(aggregate) + 0.5).astype(int)

print("etiquetas: ", labels)
print("estimadores: ", len(experts))
print("pesos: ", [e["estimator"]["predict"]("") for e in experts])
print("predicciones: ", [e["weight"] for e in experts])
print("predicciones ponderadas: ", preds)
print("suma de pesos: ", sum_weights)
print("aggregate: ", aggregate)
print("prediccion final: ", out)

print("*" * 100)

experts = [{
    "weight": np.array([0.1, 1]),
    "estimator": {
        "predict": lambda _: np.array([1, 0])
    }
}, {
    "weight": np.array([0.1, 1]),
    "estimator": {
        "predict": lambda _: np.array([1, 1])
    }
}, {
    "weight": np.array([0.1, 1]),
    "estimator": {
        "predict": lambda _: np.array([0, 0])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([0, 0])
    }
}, {
    "weight": np.array([1, 1]),
    "estimator": {
        "predict": lambda _: np.array([0, 1])
    }
}]
labels = 2
preds = [np.array(exp["estimator"]["predict"]("")) * exp["weight"]
         for exp in experts]
sum_weights = [sum(exp["weight"][label] for exp in experts)
               for label in range(labels)]
aggregate = [
    np.sum(
        [np.array(p[label]) for p in preds] / sum_weights[label],
        axis=0
    ) for label in range(labels)
]

out = (np.array(aggregate) + 0.5).astype(int)

print("etiquetas: ", labels)
print("estimadores: ", len(experts))
print("pesos: ", [e["weight"] for e in experts])
print("predicciones: ", [e["estimator"]["predict"]("") for e in experts])
print("predicciones ponderadas: ", preds)
print("suma de pesos: ", sum_weights)
print("aggregate: ", aggregate)
print("prediccion final: ", out)
