import random

N = 200


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
    with open(filename, 'w') as file:
        for element in array:
            file.write(str(element) + '\n')


array = initCoordinate(N)

filename = "./dataset/200_1.txt"
save_array_to_txt(array, filename)

print("Mảng đã được lưu vào tệp tin:", filename)
