import numpy as np

def calculate_spread(points):
    # Sắp xếp các điểm theo mục tiêu f1
    points = points[np.argsort(points[:, 0])]

    # Tính khoảng cách giữa các điểm liên tiếp
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    mean_distance = np.mean(distances)

    # Tính D_min và D_max
    D_min = np.linalg.norm(points[0] - np.array([np.min(points[:, 0]), np.min(points[:, 1])]))
    D_max = np.linalg.norm(points[-1] - np.array([np.min(points[:, 0]), np.min(points[:, 1])]))

    # Tính Spread (Δ)
    delta = (D_min + D_max + np.sum(np.abs(distances - mean_distance))) / (D_min + D_max + (len(distances) * mean_distance))

    return delta

# Ví dụ sử dụng
final_generation = np.array([
    [1, 3],
    [2, 2],
    [3, 1],
    [4, 5],
    [5, 4]
])  # Các điểm f1 và f2 trong thế hệ cuối cùng

spread = calculate_spread(final_generation)
print("Spread (Δ):", spread)
