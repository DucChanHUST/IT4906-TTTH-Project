import pandas as pd
import matplotlib.pyplot as plt

dataset = "100_1"
run = 6
filename = f"./result/f/moead/{dataset}/{run}.csv"
df = pd.read_csv(filename, sep=" ", header=None, skiprows=1)
df.columns = ["f1", "f2"]
pop_size = 32

# Số thế hệ cụ thể mà bạn muốn biểu diễn
selected_generations = [1, 5, 10, 50, 100, 1000]

colors = ["b", "g", "r", "c", "m", "y", "k"]
markers = ["o", "s", "D", "^", "v", "<", ">"]

plt.figure(figsize=(12, 8))

for i, gen in enumerate(selected_generations):
    if gen * pop_size <= len(df):
        gen_data = df.iloc[(gen - 1) * pop_size : gen * pop_size]
        plt.scatter(
            gen_data["f1"],
            gen_data["f2"],
            label=f"Gen {gen}",
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
        )
        if gen == selected_generations[-1]:
            plt.plot(
                gen_data["f1"],
                gen_data["f2"],
                "--",
                color=colors[i % len(colors)],
                label=f"Gen {gen} (connected)",
            )


plt.xlabel("number of active sensor")
plt.ylabel("total energy consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
