import numpy as np
import random

inf = 1000000000

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
  individual = []
  for i in range(N):
    individual.append(random.randint(0, 1))
  return individual

def coverShrink(N, individual, x):
  index = []
  k = 0 # k = number of active sensor
  for i in range(N):
    if individual[i] == 1:
      index.append(i)
      k += 1
  if k == 0:
    return 0, 0
  
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
        individual[index[i]] = 0
        k -= 1
  for i in range(1, k):
    if r[i] == 0:
      for j in range(i, k):
        r[j] = r[j+1]
  
  return r, k

def localSearch():
  pass

def crossover(parent1, parent2):
  index = random.randint(1, parent1.__len__()-1)
  child = parent1[:index] + parent2[index:]
  return child

def mutation(individual):
  newIndividual = individual
  while True:
    index = random.randint(0, individual.__len__()-1)
    if newIndividual[index] == 1:
      newIndividual[index] = 0
      break
    while True:
      index = random.randint(0, individual.__len__()-1)
      if newIndividual[index] == 0:
        newIndividual[index] = 1
        break
  return newIndividual

def evaluate(N, individual, X):
  (r, f2) = coverShrink(N, individual, X)
  if r == 0:
    return inf, inf, inf
  f1 = 0
  f3 = 0
  # r = [r/1000 for r in r]
  for i in range (f2):
    f1 += r[i] ** 2
    if r[i] > f3:
      f3 = r[i]
      
  return f1, f2, f3
  
def calFitness(f1, f2, f3, z, lamb_i):
  fitness = 0
  fitness += (f1 - z[0]) / lamb_i[0]
  fitness += (f2 - z[1]) / lamb_i[1]
  fitness += (f3 - z[2]) / lamb_i[2]
  return fitness

def updateZ(f1, f2, f3, z):
  if f1 < z[0]:
    z[0] = f1
  if f2 < z[1]:
    z[1] = f2
  if f3 < z[2]:
    z[2] = f3
  return z

def main(p_mutation = 0.2, gen = 100, N = 10, neighborSize = 5):
  X = initCoordinate(N)
  print("X: ", X)
  lamb = initLambda()
  # lamb = [[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.1, 0.3, 0.6], ...]
  popSize = lamb.__len__()
  # popSize = 36
  population = []
  for i in range(popSize):
    population.append(initIndividual(N))
    
  for i in range(popSize):
    print(evaluate(N, population[i], X))
  
  neighbor = searchNeighbor(lamb, neighborSize)
  # neighbor = [[0, 1, 8, 9, 2], [1, 2, 9, 8, 0], [2, 1, 9, 10, 3], ...]
  
  # EVALUATE
  f1 = [None] * popSize
  f2 = [None] * popSize
  f3 = [None] * popSize
  z =[inf, inf, inf]
  for i in range(popSize):
    f1[i], f2[i], f3[i] = evaluate(N, population[i], X)
    z = updateZ(f1[i], f2[i], f3[i], z)
  # 
  for i in range(gen):
    for j in range(popSize):
      # select 1 neighbor to crossover
      neighborIndex = random.randint(1, neighborSize-1)
      # crossover
      child = crossover(population[j], population[neighbor[j][neighborIndex]])
      f1j, f2j, f3j = evaluate(N, population[j], X)
      childFit = calFitness(f1j, f2j, f3j, z, lamb[j])
      parentFit = calFitness(f1[j], f2[j], f3[j], z, lamb[j])
      
      if childFit < parentFit:
        population[j] = child
        z = updateZ(f1j, f2j, f3j, z)
      
      # mutation
      if random.random() < p_mutation:
        newIndividual = mutation(population[j])
        f1j, f2j, f3j = evaluate(N, newIndividual, X)
        newFit = calFitness(f1j, f2j, f3j, z, lamb[j])
        if newFit < parentFit:
          population[j] = newIndividual
          f1[j], f2[j], f3[j] = evaluate(N, population[j], X)
          z = updateZ(f1[j], f2[j], f3[j], z)      
      # local search
  
  print("Result: ")
  for i in range(popSize):
    print(evaluate(N, population[i], X))
  
if __name__ == '__main__':
  main()
