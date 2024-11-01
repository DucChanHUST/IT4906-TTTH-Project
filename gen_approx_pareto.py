import glob
import numpy as np

dataset = "100_1"


def is_pareto_efficient(points):
    unique_points = np.unique(points, axis=0)
    is_efficient = np.ones(unique_points.shape[0], dtype=bool)

    for index, current_point in enumerate(unique_points):
        if is_efficient[index]:
            other_points = unique_points[is_efficient]
            is_dominated = np.all(other_points > current_point, axis=1)
            is_equal = (other_points == current_point).all(axis=1)
            is_efficient[is_efficient] = ~(is_dominated) | is_equal

    return is_efficient, unique_points


def main():
    file_paths = glob.glob(f"./result/pareto/moead/{dataset}/*.csv")
    print("Found files:", file_paths)
    all_points = []

    for file_path in file_paths:
        print(file_path)
        data = np.loadtxt(file_path, delimiter=",")
        if data.size == 0:
            print(f"No data found in {file_path}")
            continue
        all_points.append(data)

    if len(all_points) == 0:
        print("No data points to process.")
        return

    all_points = np.vstack(all_points)

    pareto_mask, unique_points = is_pareto_efficient(data)
    pareto_points = unique_points[pareto_mask]

    with open(f"./result/pareto/moead/approx_{dataset}.csv", "w") as file:
        pass
    np.savetxt(f"./result/pareto/moead/approx_{dataset}.csv", pareto_points, fmt="%.4f")


if __name__ == "__main__":
    main()
