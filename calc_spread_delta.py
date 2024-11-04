import numpy as np
import pandas as pd

dataset = "100_1"
run_start = 0
run_end = 10
output_file = f"./result/spread/moead/{dataset}.txt"


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def spread_metric(pareto_front):
    pareto_front = sorted(pareto_front, key=lambda x: x[0])
    pareto_front = np.array(pareto_front)

    distances = [
        euclidean_distance(pareto_front[i], pareto_front[i + 1])
        for i in range(len(pareto_front) - 1)
    ]

    d_f = euclidean_distance(pareto_front[0], pareto_front[-1])
    d_l = euclidean_distance(pareto_front[-1], pareto_front[0])

    d_avg = np.mean(distances)

    spread = (d_f + d_l + sum(abs(d - d_avg) for d in distances)) / (
        d_f + d_l + (len(pareto_front) - 1) * d_avg
    )
    return spread


spead_values = []
for run in range(run_start, run_end):
    input_file = f"./result/pareto/moead/{dataset}/{run}.csv"
    pareto_front = np.loadtxt(input_file, dtype=float, delimiter=",")
    spread_value = spread_metric(pareto_front)
    spead_values.append(spread_value)
    print(f"Run {run} Spread (Δ) metric:", spread_value)

with open(output_file, "w") as f:
    for spread_value in spead_values:
        f.write(f"{spread_value}\n")
