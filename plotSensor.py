import matplotlib.pyplot as plt
import numpy as np
N = 50  # Số lượng vòng tròn
x_pos = np.array([57, 57, 61, 85, 112, 115, 125, 125, 132, 155, 202, 228, 242, 282, 287, 310, 326, 384, 392, 435, 453, 513, 518, 520, 570, 570, 571, 629, 645, 647, 664, 669, 677, 693, 711, 714, 716, 740, 748, 751, 755, 759, 782, 828, 847, 851, 
865, 947, 951, 956])  # Mảng hoành độ tâm vòng tròn
radii = np.array([57, 0, 0, 0, 0, 1, 0, 0, 0, 7.0, 40.0, 0, 0, 40.0, 0, 0, 54.5, 0, 0, 54.5, 0, 23.5, 0, 0, 0, 29.5, 0, 29.5, 0, 0, 0, 10.5, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 24.0, 0, 0, 0, 50.0, 0, 0, 50.0, 0])  # Mảng bán kính vòng tròn
fig, ax = plt.subplots()

# Vẽ các vòng tròn
for i in range(N):
    circle = plt.Circle((x_pos[i], 0), radii[i], color='blue', alpha=0.5)
    ax.add_patch(circle)
    if radii[i] > 0:
      ax.scatter(x_pos[i], 0, marker='o', color='red', s=10)  # Chấm đỏ có kích thước 10
    else:
      ax.scatter(x_pos[i], 0, marker='x', color='black', s=10)  # Chấm đen có kích thước 10
# Thiết lập trục
ax.set_xlim(0, 1000)  # Thiết lập giới hạn trục X
ax.set_ylim(-radii.max(), radii.max())  # Thiết lập giới hạn trục Y
ax.set_aspect('equal')  # Giữ tỷ lệ khung hình

# Thêm nhãn trục
ax.set_xlabel('X')
ax.set_ylabel('Y')

# Hiển thị đồ thị
plt.show()
