import numpy as np
import matplotlib.pyplot as plt

run_start = 0
run_end = 50
base_link = "./result/pareto/moead/"
datasets = ["100_1", "150_1", "200_1", "250_1"]


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def calculate_igd(true_pareto, run_pareto):
    distances = []

    for true_point in true_pareto:
        min_distance = np.min(
            [euclidean_distance(true_point, run_point) for run_point in run_pareto]
        )
        distances.append(min_distance)

    igd = np.mean(distances)

    return igd


igd_values_per_dataset = []

for dataset in datasets:
    igd_values = []
    true_pareto = np.loadtxt(
        f"{base_link}approx_{dataset}.csv", dtype=float, delimiter=","
    )
    true_pareto = sorted(true_pareto, key=lambda x: x[0])
    true_pareto = np.unique(np.array(true_pareto), axis=0)

    f1_max = true_pareto[-1][0]
    f2_max = true_pareto[-1][1]
    f1_min = true_pareto[0][0]
    f2_min = true_pareto[0][1]

    for point in true_pareto:
        point[0] = (point[0] - f1_min) / (f1_max - f1_min)
        point[1] = (point[1] - f2_min) / (f2_max - f2_min)

    for run in range(run_start, run_end):
        input_file = f"{base_link}{dataset}/{run}.csv"
        run_pareto = np.loadtxt(input_file, dtype=float, delimiter=",")

        for point in run_pareto:
            point[0] = (point[0] - f1_min) / (f1_max - f1_min)
            point[1] = (point[1] - f2_min) / (f2_max - f2_min)

        igd_value = calculate_igd(true_pareto, run_pareto)
        igd_values.append(igd_value)

    igd_values_per_dataset.append(igd_values)

plt.boxplot(
    igd_values_per_dataset, labels=["100", "150", "200", "250"], patch_artist=True
)

plt.ylabel("IGD Value")
plt.xlabel("Number of Sensors")
plt.show()
