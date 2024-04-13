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

def is_all_zero(array):
    for i in array:
      if i != 0:
        return False
    return True

def initValidIndividual(N):
    individual = initIndividual(N)
    while is_all_zero(individual):
        individual = initIndividual(N)
    return individual

def coverShrink(N, individual, x):
  index = [i for i in range(N) if individual[i] == 1]
  k = len(index)
  
  half_dis = [x[index[0]], *[(x[index[i+1]] - x[index[i]]) / 2 for i in range(k-1)], 1000 - x[index[-1]]]

  r = [max(half_dis[0], half_dis[1]), max(half_dis[k-1], half_dis[k])]
  for i in range(1, k-1):
      r.insert(-1, max(half_dis[i], half_dis[i+1]))

  c_r = []
  r_index = 0
  for i in range(N):
    if individual[i] == 1:
      c_r.append(r[r_index])
      r_index += 1
    else:
      c_r.append(0)
  # Ex: c_r = [0, 100, 0, 0, 59, 74, 0,]; r = [100, 59, 74]


  # shrink - TODO: NEED A HELP
  '''
  shrink_iterations = 0
  while shrink_iterations < 1:
    changed = False 
    for i in range(1, k-1):
      if (x[index[i]] - r[i] < x[index[i-1]] + r[i-1]) and (x[index[i]] + r[i] > x[index[i+1]] - r[i+1]):
        c_r[index[i]] = max(0, min(x[index[i]] - x[index[i-1]] - r[i-1], x[index[i+1]] - x[index[i]] - r[i+1]))
        changed = True
        if c_r[index[i]] == 0:
          individual[index[i]] = 0
          k -= 1
    if not changed:
        break
    shrink_iterations += 1 
  '''
  
  return c_r, k

def crossover(parent1, parent2):
  index = random.randint(1, parent1.__len__()-1)
  child = parent1[:index] + parent2[index:]
  while is_all_zero(child):
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
    return inf, inf, inf, 0
  f1 = 0
  f3 = 0
  # r = [r/1000 for r in r]
  for i in range (N):
    if individual[i] == 1:
      f1 += r[i] ** 2
    if r[i] > f3:
      f3 = r[i]
      
  return f1, f2, f3, r
  
def calFitness(f1, f2, f3, z, zNad, lamb_i):
  fitness = 0
  fitness += ((f1 - z[0]) / (zNad[0] - z[0])) / lamb_i[0]
  fitness += ((f2 - z[1]) / (zNad[1] - z[1])) / lamb_i[1] 
  fitness += ((f3 - z[2]) / (zNad[2] - z[2])) / lamb_i[2]
  return fitness

def updateZ(f1, f2, f3, z, zNad):
  if f1 < z[0]:
    z[0] = f1
  if f2 < z[1]:
    z[1] = f2
  if f3 < z[2]:
    z[2] = f3
  if f1 > zNad[0]:
    zNad[0] = f1
  if f2 > zNad[1]:
    zNad[1] = f2
  if f3 > zNad[2]:
    zNad[2] = f3
  return z, zNad

def main(p_mutation = 0.2, gen = 50, N = 50, neighborSize = 5):
  # X = initCoordinate(N)
  X = [57, 57, 61, 85, 112, 115, 125, 125, 132, 155, 202, 228, 242, 282, 287, 310, 326, 384, 392, 435, 453, 513, 518, 520, 570, 570, 571, 629, 645, 647, 664, 669, 677, 693, 711, 714, 716, 740, 748, 751, 755, 759, 782, 828, 847, 851, 
865, 947, 951, 956]
  print("X: ", X)
  lamb = initLambda()
  # lamb = [[0.1, 0.1, 0.8], [0.1, 0.2, 0.7], [0.1, 0.3, 0.6], ...]
  popSize = lamb.__len__()
  # popSize = 36
  population = []
  for i in range(popSize):
    population.append(initValidIndividual(N))
    
  # for i in range(popSize):
  #   print(evaluate(N, population[i], X))
  
  neighbor = searchNeighbor(lamb, neighborSize)
  # neighbor = [[0, 1, 8, 9, 2], [1, 2, 9, 8, 0], [2, 1, 9, 10, 3], ...]
  
  # EVALUATE
  f1 = [None] * popSize
  f2 = [None] * popSize
  f3 = [None] * popSize
  r = [None] * popSize
  z = [inf, inf, inf]
  zNad = [0, 0, 0]
  for i in range(popSize):
    f1[i], f2[i], f3[i], r[i] = evaluate(N, population[i], X)
    z, zNad = updateZ(f1[i], f2[i], f3[i], z, zNad)
  # 
  for i in range(gen):
    print("Generation: ", i)
    for j in range(popSize):
      minFit = calFitness(f1[j], f2[j], f3[j], z, zNad, lamb[j])
      
      # select 1 neighbor to crossover
      neighborIndex = random.randint(1, neighborSize-1)
      # crossover
      child = crossover(population[j], population[neighbor[j][neighborIndex]])
      f1j, f2j, f3j, r_j = evaluate(N, child, X)
      childFit = calFitness(f1j, f2j, f3j, z, zNad, lamb[j])
      
      if childFit < minFit:
        minFit = childFit
        population[j] = child
        z, zNad = updateZ(f1j, f2j, f3j, z, zNad)
        f1[j], f2[j], f3[j], r[j] = f1j, f2j, f3j, r_j
      
      # mutation
      if random.random() < p_mutation:
        newIndividual = mutation(population[j])
        f1j, f2j, f3j, r_j = evaluate(N, newIndividual, X)
        newFit = calFitness(f1j, f2j, f3j, z, zNad, lamb[j])
        if newFit < minFit:
          minFit = newFit
          population[j] = newIndividual
          z, zNad = updateZ(f1[j], f2[j], f3[j], z, zNad)
          f1[j], f2[j], f3[j], r[j] = f1j, f2j, f3j, r_j

      # local search
      r_max, r_min = 0, inf
      index_max, index_min = -1, -1
      for k in range (0, N-4):
        if (population[j][k] == 0 and population[j][k+1] == 1 and population[j][k+2] == 0):
          if r[j][k+1] > r_max:
            index_max = k+1
            r_max = r[j][index_max]
        elif (population[j][k] == 1 and population[j][k+1] == 0 and population[j][k+2] == 1):
          if r[j][k+1] < r_min:
            index_min = k+1
            r_min = r[j][index_min]

      if index_max != -1:
        forward = population[j]
        forward[index_max-1:index_max+2] = [1, 0, 1]
        f1j, f2j, f3j, r_j = evaluate(N, forward, X)
        forwardFit = calFitness(f1j, f2j, f3j, z, zNad, lamb[j])
        if forwardFit < minFit:
          minFit = forwardFit
          population[j] = forward
          z, zNad = updateZ(f1[j], f2[j], f3[j], z, zNad)
          f1[j], f2[j], f3[j], r[j] = f1j, f2j, f3j, r_j

      if index_min != -1:
        backward = population[j]
        backward[index_min-1:index_min+2] = [0, 1, 0]
        f1j, f2j, f3j, r_j = evaluate(N, backward, X)
        backwardFit = calFitness(f1j, f2j, f3j, z, zNad, lamb[j])
        if backwardFit < minFit:
          minFit = backwardFit
          population[j] = backward
          z, zNad = updateZ(f1[j], f2[j], f3[j], z, zNad)
          f1[j], f2[j], f3[j], r[j] = f1j, f2j, f3j, r_j
      

  print("Result: ")
  for i in range(popSize):
    print(i, r[i])
    print("Fitness: ", calFitness(f1[i], f2[i], f3[i], z, zNad, lamb[i]))
    print(f1[i], f2[i], f3[i])
    print("\n")

  print("Z: ", z, zNad)
  
if __name__ == '__main__':
  main()
