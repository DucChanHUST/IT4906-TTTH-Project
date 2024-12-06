import glob
import numpy as np
import pandas as pd

datasets = ["100", "150", "200", "250"]
algos = ["moead", "nsga"]


def is_pareto_efficient(costs):
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1

    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


for i in range(0, 10):
    for dataset in datasets:
        for algo in algos:
            file_paths = glob.glob(
                f"./result/pareto/{algo}/{dataset}/{algo}_{dataset}_{i}*.csv"
            )

            all_points = []
            for file_path in file_paths:
                data = pd.read_csv(file_path, header=None, names=["f1", "f2"])
                all_points.append(data)

            combined_data = pd.concat(all_points, ignore_index=True).drop_duplicates()

            points = combined_data[["f1", "f2"]].values
            pareto_mask = is_pareto_efficient(points)
            pareto_front = combined_data[pareto_mask]
            pareto_front_sorted = pareto_front.sort_values(
                by=["f1", "f2"], ascending=[False, True]
            )

            output_path = f"./result/pareto/approx/{algo}_{dataset}_{i}.csv"
            pareto_front_sorted.to_csv(output_path, index=False, header=False, sep=",")
            print("Pareto front saved to:", output_path)
