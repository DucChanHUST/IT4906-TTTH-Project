import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Đọc dữ liệu từ tệp tin và phân tích nó


def read_data_from_txt(filename):
    data = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(': ')
                ind = int(parts[0].split(', ')[1].split(' ')[1])
                values = list(map(float, parts[1].split(' ')))
                if ind not in data:
                    data[ind] = []
                data[ind].append(values)
    return data


# Đọc dữ liệu từ tệp tin
filename = 'result.txt'
data = read_data_from_txt(filename)

# Biểu diễn dữ liệu dưới dạng 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tạo bảng màu cho các cá thể khác nhau
colors = list(plt.cm.jet(np.linspace(0, 1, len(data))))

# Vẽ các điểm trên biểu đồ với màu khác nhau cho mỗi cá thể
for ind, values in data.items():
    if (ind % 7 == 3):
        values = np.array(values)
        ax.scatter(values[:, 0], values[:, 1], values[:, 2], c=[
            colors[ind-1]], label=f'Individual {ind}')

        for i in range(len(values) - 1):
            ax.plot([values[i][0], values[i+1][0]],
                    [values[i][1], values[i+1][1]],
                    [values[i][2], values[i+1][2]], c=colors[ind-1], alpha=0.5)


ax.set_xlabel('Power')
ax.set_ylabel('Reliability')
ax.set_zlabel('Fairness')
ax.legend()

plt.show()
