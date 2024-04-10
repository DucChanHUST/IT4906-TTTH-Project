import numpy as np
import random

def initCoordinate(N):
  x = []
  for i in range(N):
    x.append(random.randint(1, 1000))
  x.sort()
  return x

def initLambda():
  lamb = []
  # step = 0.1
  for i in range(9):
    for j in range(9):
      a = i + 1
      b = j + 1
      c = 10 - a - b
      if c > 0:
        lamb.append([a/10, b/10, c/10])
  # lamb.__len__() = 36 
  return lamb

def searchNeighbor(lamb, neighborSize):
  B = []
  distance = []
  num = lamb.__len__()
  for i in range(num):
    distance.append([0 for _ in range(num)])

  for i in range(num):
    for j in range(i, num):
      lamb1 = lamb[i]
      lamb2 = lamb[j]
      dis = 0
      for k in range(3):
        dis += (lamb1[k] - lamb2[k]) ** 2
      dis = dis ** 0.5
      distance[i][j] = dis
      distance[j][i] = dis
    temp_list = np.array(distance[i])
    index = np.argsort(temp_list)
    index = index.tolist()
    B.append(index[:neighborSize])
  return B

def initIndividual(N):
  solution = []
  for i in range(N):
    solution.append(random.randint(0, 1))
  return solution

def coverShrink(N, solution, x):
  index = []
  k = 0 # k = number of active sensor
  for i in range(N):
    if solution[i] == 1:
      index.append(i)
      k += 1

  half_dis = []
  half_dis.append(x[index[0]])
  for i in range(0, k-1):
    half_dis.append((x[index[i+1]] - x[index[i]]) / 2)
  half_dis.append(1000 - x[index[-1]])

  r = []
  for i in range(k):
    r.append(max(half_dis[i], half_dis[i+1]))
  
  # shrink
  for i in range(1, k-1):
    if (x[index[i]] - r[i] < x[index[i-1]] + r[i-1]) & (x[index[i]] + r[i] > x[index[i+1]] - r[i+1]):
      r[i] = max(0, min(x[index[i]]-x[index[i-1]]-r[i-1], x[index[i+1]] - x[index[i]] - r[i+1]))
      if r[i] == 0:
        solution[index[i]] = 0
        k -= 1
  for i in range(1, k):
    if r[i] == 0:
      for j in range(i, k):
        r[j] = r[j+1]
  
  return r, k, index

def localSearch():
  pass

def crossover(parent1, parent2):
  pass

def mutation(solution):
  pass

def evaluate(N, individual, X):
  (r, f2, index) = coverShrink(N, individual, X)
  f1 = 0
  f3 = 0
  # r = [r/1000 for r in r]
  for i in range (f2):
    f1 += r[i] ** 2
    if r[i] > f3:
      f3 = r[i]
      
  return f1, f2, f3
  

def main(p_mutation = 0.2, gen = 100, N = 10, neighborSize = 5):
  inf = 1000000000
  lamb = initLambda()
  X = initCoordinate(N)
  # lamb = [[0, 1, 8, 9, 2], [1, 2, 9, 8, 0], [2, 1, 9, 10, 3], ...]
  popSize = lamb.__len__()
  # popSize = 36
  population = []
  for i in range(popSize):
    population.append(initIndividual(N))
  
  neighbor = searchNeighbor(lamb, neighborSize)
  # neighbor = [[0, 1, 8, 9, 2], [1, 2, 9, 8, 0], [2, 1, 9, 10, 3], ...]
  
  # EVALUATE
  f1 = [None] * popSize
  f2 = [None] * popSize
  f3 = [None] * popSize
  z =[inf, inf, inf]
  for i in range(popSize):
    f1[i], f2[i], f3[i] = evaluate(N, population[i], X)
    if f1[i] < z[0]:
      z[0] = f1[i]
    if f2[i] < z[1]:
      z[1] = f2[i]
    if f3[i] < z[2]:
      z[2] = f3[i]
  
  print(z)
  
if __name__ == '__main__':
  main()
