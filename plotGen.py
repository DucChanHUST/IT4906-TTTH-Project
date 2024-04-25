import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Đọc dữ liệu từ tệp tin và phân tích nó


def read_data_from_txt(filename):
    data = {}
    with open(filename, 'r') as file:
        next(file)
        for line in file:
            parts = line.split()
            if line:
                gen = int(parts[0])
                values = list(map(float, parts[2:5]))
                if gen not in data:
                    data[gen] = []
                data[gen].append(values)
    return data


# Đọc dữ liệu từ tệp tin
filename = 'result_MOEAD.txt'
data = read_data_from_txt(filename)

# Biểu diễn dữ liệu dưới dạng 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tạo bảng màu cho các thế hệ khác nhau
colors = plt.cm.jet(np.linspace(0, 1, len(data)))

# Vẽ các điểm trên biểu đồ với màu khác nhau cho mỗi thế hệ
for gen, values in data.items():
    if (gen % 5 == 0):
        values = np.array(values)
        ax.scatter(values[:, 0], values[:, 1], values[:, 2],
                c=[colors[gen-1]], label=f'Gen {gen}')
        
        for i in range(len(values) - 1):
            ax.plot([values[i][0], values[i+1][0]],
                    [values[i][1], values[i+1][1]],
                    [values[i][2], values[i+1][2]], c=colors[gen-1], alpha=0.4)

ax.set_xlabel('Power')
ax.set_ylabel('Reliability')
ax.set_zlabel('Fairness')
ax.legend()

plt.show()
