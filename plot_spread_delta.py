import matplotlib.pyplot as plt


def read_data(filename):
    with open(filename, "r") as file:
        return [float(line.strip()) for line in file.readlines()]


base_link = "./result/spread/moead/"
datasets = ["100_1", "150_1", "200_1", "250_1"]

datas = []
for dataset in datasets:
    file_path = f"{base_link}{dataset}.txt"
    data = read_data(file_path)
    datas.append(data)

plt.figure(figsize=(10, 6))
box = plt.boxplot(datas, labels=["100", "150", "200", "250"], patch_artist=True)

color = "#66B3FF"
for patch in box["boxes"]:
    patch.set_facecolor(color)

plt.ylabel("Spread Delta Value")
plt.xlabel("Number of Sensors")
plt.legend()
plt.show()
