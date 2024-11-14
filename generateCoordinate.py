from operator import index
import random


def is_all_zero(array):
    for i in array:
        if i != 0:
            return False
    return True


def initCoordinate(N):
    x = set()
    while x.__len__() < N:
        x.add(random.randint(0, 1000))
    x = list(x)
    x.sort()
    return x


def save_array_to_txt(array, filename):
    with open(filename, "w") as file:
        for element in array:
            file.write(str(element) + "\n")


array = initCoordinate(N)

N = 150
start_index = 0
end_index = 1
for index in range(start_index, end_index):
    filename = f"./dataset/{N}/{N}_{index}.txt"
    save_array_to_txt(array, filename)

print("Mảng đã được lưu vào tệp tin:", filename)
